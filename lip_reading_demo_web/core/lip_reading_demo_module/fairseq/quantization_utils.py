

import logging

from fairseq.modules.quantization import scalar
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

def quantize_model_scalar(model, model_cfg: DictConfig):
    quant_noise_scalar = getattr(model_cfg, "quant_noise_scalar", 0) or 0
    if quant_noise_scalar > 0:

        scalar.quantize_model_(model, p=quant_noise_scalar, bits=8, update_step=1000)
    return model

