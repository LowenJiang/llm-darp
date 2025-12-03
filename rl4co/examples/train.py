import torch
#from inner_loop.rl4co.envs import DARPEnv
import os, sys
print(os.getcwd())
# Should show: /home/jiangwolin/rl4co/examples
import sys
sys.path.append("../../")
sys.path.insert(0, "../")
sys.path.append(os.path.abspath(".."))

from rl4co.envs.routing import PDPTWGenerator, CVRPTWGenerator, SFGenerator
from rl4co.envs.routing import PDPTWEnv, CVRPTWEnv
from rl4co.models import AttentionModelPolicy, REINFORCE
from rl4co.models.zoo.pomo.model import POMO
from rl4co.utils.trainer import RL4COTrainer
from lightning.pytorch.callbacks import ModelCheckpoint, RichModelSummary
from rl4co.models.zoo import AttentionModel
from lightning.pytorch.loggers import WandbLogger

import wandb

wandb.login()

env = PDPTWEnv()

# Policy: neural network, in this case with encoder-decoder architecture
policy = AttentionModelPolicy(env_name=env.name,
                              embed_dim=128,
                              num_encoder_layers=3,
                              num_heads=8,
                              temperature=1.2  # Increase exploration (default is 1.0)
                            )


model = REINFORCE(
    env,
    policy=policy,            # or omit to use the default AttentionModelPolicy
    baseline="rollout",
    batch_size=512,
    train_data_size=10_000,#100_000,
    val_data_size=1000,
    optimizer_kwargs={"lr": 1e-4},
    metrics={
        "train": ["loss", "reward", "cost", "unresolved_penalty"],
        "val": ["reward", "cost", "unresolved_penalty"],
        "test": ["reward", "cost", "unresolved_penalty"],
    },
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = WandbLogger(project="rl4co", name="h3-7-run5")

checkpoint_callback = ModelCheckpoint(  dirpath="checkpoints/sf_newenv_5", # save to checkpoints/
                                        filename="epoch_{epoch:03d}",  # save as epoch_XXX.ckpt
                                        save_top_k=1, # save only the best model
                                        save_last=True, # save the last model
                                        monitor="val/reward", # monitor validation reward
                                        mode="max") # maximize validation reward

rich_model_summary = RichModelSummary(max_depth=3)
callbacks = [checkpoint_callback, rich_model_summary]

trainer = RL4COTrainer(
    max_epochs=120,
    accelerator="cpu",#"gpu",
    devices=1,#-1,
    logger=logger,
    callbacks=callbacks,
)

trainer.fit(model) #trainer.fit(model, ckpt_path="checkpoints/sf_newenv_2/epoch=048.ckpt")
