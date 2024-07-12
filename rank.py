# Simple test to verify that multi-node NeMo ranks are set up successfuly.
# Test succeeds if all nodes print Success message
# Test fails is any node prints Error message

import platform
import sys

import torch.distributed as dist
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy
from nemo.utils.get_rank import get_rank
from nemo.core.config import hydra_runner
from omegaconf import DictConfig

# Trainer needs some function; this empty function allows
# us to initialize NeMo by running a trainer that does nothing
def do_nothing():
    return

# The hydra_runner reads configure options from a file 
# and allows overrides on the command line
# (e.g., trainer.num_nodes=2)
@hydra_runner(config_path="conf", config_name="rank")
def main(cfg: DictConfig) -> None:
    hostname = platform.node()

    # Explicitly check for mpi4py so we can fail if it's not present
    # Otherwise, if it's missing, NeMo will hang during startup
    try:
        import mpi4py
    except ImportError as ex:
        print(f"[hostname={hostname}] Error: unable to load mpi4py: {ex}")
        return 2

    # Create a Trainer using the config from config/rank.yaml and
    # command-line overrides
    trainer = Trainer(strategy=NLPDDPStrategy(), **cfg.trainer)
    if trainer.strategy.launcher is not None:
        # Run a trivial Trainer that does nothing on each rank
        trainer.strategy.launcher.launch(do_nothing, trainer=trainer)
        # Ensure rank variables are set
        trainer.strategy.setup_environment()
    # Wait for Trainer to finish on all ranks
    dist.barrier()

    # NeMo and PyTorch have their own notions of rank which must agree
    nemo_rank = get_rank()
    torch_rank = dist.get_rank()
    print(f"[hostname={hostname}] nemo_rank={nemo_rank}, torch_rank={torch_rank} Hello world!")

    # Check whether NeMo and PyTorch rank numbers agree
    # This will fail if NeMo is launched other than through srun or Open MPI's mpirun
    # unless $SLURM_PROCID or $OMPI_COMM_WORLD_RANK are set in the environment
    if(nemo_rank != torch_rank):
        print(f"[hostname={hostname}] Error: NeMo and PyTorch don't agree on rank number")
        return 1
    else:
        print(f"[hostname={hostname}] Success: NeMo and PyTorch agree on rank number")
        return 0

if __name__ == '__main__':
    sys.exit(main())

