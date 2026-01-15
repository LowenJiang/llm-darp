import copy
import logging
from typing import Any, Callable, Dict, Optional, Union

import torch
import torch.nn as nn


log = logging.getLogger(__name__)


class RewardScaler:
    """Simple running statistic tracker used to normalize or scale rewards."""

    def __init__(self, scale: Optional[Union[str, int]] = None):
        self.scale = scale
        self.count = 0
        self.mean = 0
        self.M2 = 0

    def __call__(self, scores: torch.Tensor) -> torch.Tensor:
        if self.scale is None:
            return scores
        if isinstance(self.scale, int):
            return scores / self.scale

        self.update(scores)
        tensor_to_kwargs = dict(dtype=scores.dtype, device=scores.device)
        std = (self.M2 / (self.count - 1)).float().sqrt()
        scaling = std.to(**tensor_to_kwargs) + torch.finfo(scores.dtype).eps
        if self.scale == "norm":
            scores = (scores - self.mean.to(**tensor_to_kwargs)) / scaling
        elif self.scale == "scale":
            scores = scores / scaling
        else:
            raise ValueError(f"Unknown scaling operation requested: {self.scale}")
        return scores

    @torch.no_grad()
    def update(self, batch: torch.Tensor) -> None:
        flat = batch.reshape(-1)
        self.count += len(flat)
        delta = flat - self.mean
        self.mean += (delta / self.count).sum()
        delta2 = flat - self.mean
        self.M2 += (delta * delta2).sum()


class REINFORCEBaseline(nn.Module):
    """Minimal baseline interface used by the distilled REINFORCE trainer."""

    def wrap_dataset(self, dataset: Any, *args, **kwargs) -> Any:  # pragma: no cover - passthrough
        return dataset

    def setup(self, *args, **kwargs) -> None:  # pragma: no cover - optional hook
        pass

    def epoch_callback(self, *args, **kwargs) -> None:  # pragma: no cover - optional hook
        pass

    def eval(self, state: Any, reward: torch.Tensor, env: Any = None, **kwargs):
        raise NotImplementedError


class NoBaseline(REINFORCEBaseline):
    """No baseline: always returns zero baseline and zero baseline loss."""

    def eval(self, state: Any, reward: torch.Tensor, env: Any = None):
        return torch.zeros_like(reward), torch.tensor(0.0, device=reward.device)


class SharedBaseline(REINFORCEBaseline):
    """Shared baseline: mean reward along a dimension (defaults to dim=1)."""

    def __init__(self, on_dim: int = 1):
        super().__init__()
        self.on_dim = on_dim

    def eval(self, state: Any, reward: torch.Tensor, env: Any = None):
        return reward.mean(dim=self.on_dim, keepdim=True), torch.tensor(
            0.0, device=reward.device
        )


class ExponentialBaseline(REINFORCEBaseline):
    """Exponential moving average baseline."""

    def __init__(self, beta: float = 0.8):
        super().__init__()
        self.beta = beta
        self.v = None

    def eval(self, state: Any, reward: torch.Tensor, env: Any = None):
        if self.v is None:
            value = reward.mean()
        else:
            value = self.beta * self.v + (1.0 - self.beta) * reward.mean()
        self.v = value.detach()
        return self.v, torch.tensor(0.0, device=reward.device)


class MeanBaseline(ExponentialBaseline):
    """Baseline equivalent to the running mean (beta=0)."""

    def __init__(self):
        super().__init__(beta=0.0)


BASELINE_REGISTRY: Dict[str, Callable[..., REINFORCEBaseline]] = {
    "no": NoBaseline,
    "shared": SharedBaseline,
    "exponential": ExponentialBaseline,
    "mean": MeanBaseline,
}


def get_reinforce_baseline(name: Optional[str], **kwargs) -> REINFORCEBaseline:
    """Instantiate a baseline from a name."""
    if name is None:
        name = "no"
    try:
        return BASELINE_REGISTRY[name](**kwargs)
    except KeyError:
        raise ValueError(f"Unknown baseline '{name}'. Available: {list(BASELINE_REGISTRY)}")


def _clone_state(x: Any) -> Any:
    """Clone a nested structure that may contain tensors."""
    if hasattr(x, "clone"):
        try:
            return x.clone()
        except TypeError:
            pass
    if isinstance(x, dict):
        return {k: _clone_state(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        cloned = [_clone_state(v) for v in x]
        return type(x)(cloned)
    return copy.deepcopy(x)


class REINFORCE:
    """Self-contained REINFORCE trainer.

    The class expects two callables:
    - env.reset(batch) -> state object consumed by the policy
    - policy(state, env, ...) -> dict with at least `reward` and `log_likelihood`
    """

    def __init__(
        self,
        env: Any,
        policy: nn.Module,
        baseline: Union[REINFORCEBaseline, str, None] = "mean",
        baseline_kwargs: Optional[Dict[str, Any]] = None,
        reward_scale: Optional[Union[str, int]] = None,
    ):
        self.env = env
        self.policy = policy
        baseline_kwargs = baseline_kwargs or {}

        if isinstance(baseline, str) or baseline is None:
            baseline = get_reinforce_baseline(baseline, **baseline_kwargs)
        elif baseline_kwargs:
            log.warning("baseline_kwargs ignored because a baseline instance was provided.")

        self.baseline = baseline
        self.advantage_scaler = RewardScaler(reward_scale)

    def post_setup_hook(
        self, batch_size: int = 64, device: Union[str, torch.device] = "cpu", dataset_size: Optional[int] = None
    ) -> None:
        """Optional setup hook to initialize the baseline."""
        try:
            self.baseline.setup(self.policy, self.env, batch_size=batch_size, device=device, dataset_size=dataset_size)
        except TypeError:
            # Baseline may not use all arguments; ignore unexpected ones.
            self.baseline.setup(self.policy, self.env)

    def shared_step(self, batch: Any, phase: str = "train", dataloader_idx: Optional[int] = None) -> Dict[str, Any]:
        state = self.env.reset(batch) if hasattr(self.env, "reset") else batch
        initial_state = _clone_state(state) if phase == "train" else None

        outputs = self.policy(state, self.env, phase=phase, select_best=phase != "train")
        if phase == "train":
            outputs = self.calculate_loss(initial_state, batch, outputs)

        return {"loss": outputs.get("loss"), **self._collect_metrics(outputs, phase, dataloader_idx)}

    def calculate_loss(
        self,
        initial_state: Any,
        batch: Any,
        policy_out: Dict[str, Any],
        reward: Optional[torch.Tensor] = None,
        log_likelihood: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        reward = policy_out["reward"] if reward is None else reward
        log_likelihood = policy_out["log_likelihood"] if log_likelihood is None else log_likelihood

        extra = batch.get("extra", None) if isinstance(batch, dict) else None
        actions = policy_out.get("actions", None)
        if actions is not None and hasattr(self.env, "validate_actions"):
            valid_mask = self.env.validate_actions(actions)
            num_invalid = (~valid_mask).sum().item()
            if num_invalid > 0:
                log.warning(
                    "Found %d/%d invalid instances. Excluding from loss computation.",
                    num_invalid,
                    valid_mask.shape[0],
                )
                if not valid_mask.any():
                    zero = torch.tensor(0.0, device=reward.device, requires_grad=True)
                    policy_out.update(
                        {"loss": zero, "reinforce_loss": zero, "bl_loss": torch.tensor(0.0, device=reward.device), "bl_val": torch.zeros_like(reward)}
                    )
                    return policy_out

                reward = reward[valid_mask]
                log_likelihood = log_likelihood[valid_mask]
                if initial_state is not None and hasattr(initial_state, "__getitem__"):
                    initial_state = initial_state[valid_mask]
                if extra is not None and hasattr(extra, "__getitem__"):
                    extra = extra[valid_mask]

        if extra is None:
            baseline_value, baseline_loss = self.baseline.eval(initial_state, reward, self.env)
        else:
            baseline_value, baseline_loss = extra, torch.tensor(0.0, device=reward.device)

        advantage = reward - baseline_value
        advantage = self.advantage_scaler(advantage)
        reinforce_loss = -(advantage * log_likelihood).mean()
        loss = reinforce_loss + baseline_loss

        policy_out.update(
            {
                "loss": loss,
                "reinforce_loss": reinforce_loss,
                "bl_loss": baseline_loss,
                "bl_val": baseline_value,
            }
        )
        return policy_out

    def wrap_dataset(self, dataset: Any, batch_size: int = 64, device: Union[str, torch.device] = "cpu") -> Any:
        """Allow baselines to preprocess datasets (e.g., add cached rewards)."""
        return self.baseline.wrap_dataset(dataset, self.env, batch_size=batch_size, device=device)

    def set_decode_type_multistart(self, phase: str) -> None:
        """Helper to prepend `multistart_` to `train/val/test` decode type attributes."""
        attribute = f"{phase}_decode_type"
        attr_value = getattr(self.policy, attribute, None)
        if attr_value is None:
            log.error("Decode type for %s is None. Cannot prepend `multistart_`.", phase)
            return
        if "multistart" not in attr_value:
            setattr(self.policy, attribute, f"multistart_{attr_value}")

    def _collect_metrics(self, outputs: Dict[str, Any], phase: str, dataloader_idx: Optional[int]) -> Dict[str, Any]:
        metrics = {}
        reward = outputs.get("reward")
        if reward is not None:
            metrics[f"{phase}_reward_mean"] = reward.detach().mean()
        reinforce_loss = outputs.get("reinforce_loss")
        if reinforce_loss is not None:
            metrics[f"{phase}_reinforce_loss"] = reinforce_loss.detach()
        baseline_loss = outputs.get("bl_loss")
        if baseline_loss is not None:
            metrics[f"{phase}_baseline_loss"] = baseline_loss.detach()
        if dataloader_idx is not None:
            metrics["dataloader_idx"] = dataloader_idx
        return metrics


def main() -> None:
    """Sanity-check REINFORCE with PDPTWEnv + PDPTWAttentionPolicy."""
    from pathlib import Path

    from env import PDPTWEnv
    from generator import SFGenerator
    from policy import PDPTWAttentionPolicy

    torch.manual_seed(0)

    csv_path = Path(__file__).with_name("traveler_trip_types_res_7.csv")
    ttm_path = Path(__file__).with_name("travel_time_matrix_res_7.csv")
    generator = SFGenerator(
        csv_path=csv_path,
        travel_time_matrix_path=ttm_path,
        num_customers=2,
        perturbation=0,
        vehicle_capacity=8,
    )

    env = PDPTWEnv(generator=generator)
    policy = PDPTWAttentionPolicy()
    policy.temperature = 2.0
    trainer = REINFORCE(env, policy, baseline="mean")

    batch = generator(batch_size=[4])
    state = env.reset(batch)
    policy.train()

    outputs = policy(state, env, phase="train", decode_type="sampling")
    outputs = trainer.calculate_loss(_clone_state(state), batch, outputs)
    loss = outputs["loss"]
    loss.backward()

    print(f"Loss: {loss.item():.4f}")
    print(f"Reward mean: {outputs['reward'].mean().item():.4f}")
    if "actions" in outputs:
        print(f"Actions shape: {tuple(outputs['actions'].shape)}")


__all__ = [
    "REINFORCE",
    "RewardScaler",
    "REINFORCEBaseline",
    "NoBaseline",
    "SharedBaseline",
    "ExponentialBaseline",
    "MeanBaseline",
    "get_reinforce_baseline",
]


if __name__ == "__main__":
    main()
