

from typing import Optional
from dataclasses import dataclass, field
from fairseq.tasks import register_task
from avhubert.hubert_pretraining import AVHubertPretrainingConfig, AVHubertPretrainingTask

@dataclass
class AVRomanConfig(AVHubertPretrainingConfig):
    time_mask: bool = field(default=False)
    random_erase: bool = field(default=False)
    target_dictionary: Optional[str] = field(
        default=None,
        metadata={
            "help": "override default dictionary location"
        }
    )
    label_rate: int = field(default=-1)

@register_task("av-roman-task", dataclass=AVRomanConfig)
class AV_RomanTask(AVHubertPretrainingTask):
    cfg: AVRomanConfig

    def __init__(
        self,
        cfg: AVRomanConfig,
    ) -> None:
        super().__init__(cfg)

        self.blank_symbol = "<s>"

