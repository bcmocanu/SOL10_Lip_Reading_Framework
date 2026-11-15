

from typing import List, Optional
from dataclasses import dataclass, field
from fairseq.data import Dictionary
from fairseq.tasks import register_task
from omegaconf import MISSING, II

from avhubert.hubert_pretraining import AVHubertPretrainingConfig, AVHubertPretrainingTask

@dataclass
class Zero_AVSRConfig(AVHubertPretrainingConfig):
    time_mask: bool = field(default=False)
    random_erase: bool = field(default=False)
    target_dictionary: Optional[str] = field(
        default=None,
        metadata={
            "help": "override default dictionary location"
        }
    )
    label_rate: int = field(default=-1)
    llm_path: str = field(
        default=MISSING, metadata={"help": "path to llama checkpoint"}
    )

    zero_shot: bool = field(
        default=True, metadata={"help": "select zero shot setting"}
    )
    zero_shot_lang: str = field(
        default='deu', metadata={"help": "ara, deu, ell, spa, fra, ita, por, rus"}
    )

@register_task("zero-avsr-task", dataclass=Zero_AVSRConfig)
class Zero_AVSR_TrainingTask(AVHubertPretrainingTask):
    cfg: Zero_AVSRConfig

    def __init__(
        self,
        cfg: Zero_AVSRConfig,
    ) -> None:
        super().__init__(cfg)

        self.blank_symbol = "<s>"

    @property
    def source_dictionary(self) -> Optional[Dictionary]:
        return None

    @property
    def target_dictionary(self) -> Optional[Dictionary]:
        return None

    @property
    def dictionaries(self) -> List[Dictionary]:
        return None

    @classmethod
    def setup_task(
        cls, cfg: Zero_AVSRConfig, **kwargs
    ) -> "Zero_AVSR_TrainingTask":
        if cfg.pdb:
            import pdb
            pdb.set_trace()
        return cls(cfg)

