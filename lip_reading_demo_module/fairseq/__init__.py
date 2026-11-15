import os
import sys
from .version import __version__
__all__ = ["pdb"]
from fairseq.distributed import utils as distributed_utils
from fairseq.logging import meters, metrics, progress_bar
sys.modules["fairseq.distributed_utils"] = distributed_utils
sys.modules["fairseq.meters"] = meters
sys.modules["fairseq.metrics"] = metrics
sys.modules["fairseq.progress_bar"] = progress_bar
from fairseq.dataclass.initialize import hydra_init
hydra_init()
import fairseq.models
import fairseq.modules
import fairseq.tasks
