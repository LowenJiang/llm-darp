import torch
#from rl4co.envs import DARPEnv
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
                              num_heads=8
                            )


model = REINFORCE(
    env,
    policy=policy,            # or omit to use the default AttentionModelPolicy
    baseline="rollout",
    batch_size=512,
    train_data_size=100_000,
    val_data_size=5000,
    optimizer_kwargs={"lr": 1e-4},
    metrics={
        "train": ["loss", "reward", "vehicles_used"],
        "val": ["reward", "vehicles_used"],
        "test": ["reward", "vehicles_used"],
    },
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = WandbLogger(project="rl4co", name="sf-newenv-2", id="s73fo521", resume='allow')

checkpoint_callback = ModelCheckpoint(  dirpath="checkpoints/sf_newenv_2", # save to checkpoints/
                                        filename="epoch_{epoch:03d}",  # save as epoch_XXX.ckpt
                                        save_top_k=1, # save only the best model
                                        save_last=True, # save the last model
                                        monitor="val/reward", # monitor validation reward
                                        mode="max") # maximize validation reward

rich_model_summary = RichModelSummary(max_depth=3)
callbacks = [checkpoint_callback, rich_model_summary]

trainer = RL4COTrainer(
    max_epochs=80,
    accelerator="cpu",#"gpu",
#    devices=-1,
    logger=logger,
    callbacks=callbacks,
)

trainer.fit(model) #trainer.fit(model, ckpt_path="checkpoints/sf_newenv_2/epoch=048.ckpt")
