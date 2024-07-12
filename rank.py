import platform
import sys

import torch.distributed as dist
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy
from nemo.utils.get_rank import get_rank
from nemo.core.config import hydra_runner
from omegaconf import DictConfig

def do_nothing():
    return

@hydra_runner(config_path="conf", config_name="rank")
def main(cfg: DictConfig) -> None:
    hostname = platform.node()

    try:
        import mpi4py
    except ImportError as ex:
        print(f"[hostname={hostname}] Error: unable to load mpi4py: {ex}")
        return 2

    trainer = Trainer(strategy=NLPDDPStrategy(), **cfg.trainer)
    if trainer.strategy.launcher is not None:
        trainer.strategy.launcher.launch(do_nothing, trainer=trainer)
        trainer.strategy.setup_environment()
    dist.barrier()

    nemo_rank = get_rank()
    torch_rank = dist.get_rank()
    print(f"[hostname={hostname}] nemo_rank={nemo_rank}, torch_rank={torch_rank} Hello world!")

    if(nemo_rank != torch_rank):
        print(f"[hostname={hostname}] Error: NeMo and PyTorch don't agree on rank number")
        return 1
    else:
        print(f"[hostname={hostname}] Success: NeMo and PyTorch agree on rank number")
        return 0

if __name__ == '__main__':
    sys.exit(main())

