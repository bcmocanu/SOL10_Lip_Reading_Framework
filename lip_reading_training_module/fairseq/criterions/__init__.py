"""isort:skip_file"""

import importlib
import os

from fairseq import registry
from fairseq.criterions.fairseq_criterion import (  # noqa
    FairseqCriterion,
    LegacyFairseqCriterion,
)
from omegaconf import DictConfig


(
    build_criterion_,
    register_criterion,
    CRITERION_REGISTRY,
    CRITERION_DATACLASS_REGISTRY,
) = registry.setup_registry(
    "--criterion", base_class=FairseqCriterion, default="cross_entropy"
)


def build_criterion(cfg: DictConfig, task):
    return build_criterion_(cfg, task)


from . import ctc, cross_entropy
