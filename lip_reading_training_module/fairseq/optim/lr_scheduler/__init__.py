"""isort:skip_file"""

import importlib
import os

from fairseq import registry
from fairseq.optim.lr_scheduler.fairseq_lr_scheduler import (  # noqa
    FairseqLRScheduler,
    LegacyFairseqLRScheduler,
)
from omegaconf import DictConfig


(
    build_lr_scheduler_,
    register_lr_scheduler,
    LR_SCHEDULER_REGISTRY,
    LR_SCHEDULER_DATACLASS_REGISTRY,
) = registry.setup_registry(
    "--lr-scheduler", base_class=FairseqLRScheduler, default="fixed"
)


def build_lr_scheduler(cfg: DictConfig, optimizer):
    return build_lr_scheduler_(cfg, optimizer)


# polynomial_decay is also registered to read AV-HuBERT pretraining metadata.
from . import tri_stage_lr_scheduler, cosine_lr_scheduler, fixed_schedule, polynomial_decay_schedule
