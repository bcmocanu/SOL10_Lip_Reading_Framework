from dataclasses import dataclass, field
from typing import List, Optional

from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass


# Retained for loading legacy AV-HuBERT Namespace configurations.
# Stage 1 and stage 2 never build this pretraining criterion.
@dataclass
class AVHubertCriterionConfig(FairseqDataclass):
    pred_masked_weight: float = field(
        default=1.0,
        metadata={"help": "weight for predictive loss for masked frames"},
    )
    pred_nomask_weight: float = field(
        default=0.0,
        metadata={"help": "weight for predictive loss for unmasked frames"},
    )
    loss_weights: Optional[List[float]] = field(
        default=None,
        metadata={"help": "weights for additional loss terms (not first one)"},
    )
    log_keys: List[str] = field(
        default_factory=lambda: [],
        metadata={"help": "output keys to log"},
    )


@register_criterion("av_hubert", dataclass=AVHubertCriterionConfig)
class AVHubertCriterion(FairseqCriterion):
    def __init__(self, task, pred_masked_weight, pred_nomask_weight, loss_weights=None, log_keys=None):
        super().__init__(task)
        self.pred_masked_weight = pred_masked_weight
        self.pred_nomask_weight = pred_nomask_weight
        self.loss_weights = loss_weights
        self.log_keys = [] if log_keys is None else log_keys

    def forward(self, model, sample, reduce=True, log_pred=False):
        raise NotImplementedError(
            "AV-HuBERT pretraining is not included. This registration only supports "
            "legacy pretrained checkpoint configuration loading. Use stage 1 or stage 2."
        )
