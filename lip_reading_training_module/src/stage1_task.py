import logging
import os
from typing import Optional

from dataclasses import dataclass, field
from fairseq.data import Dictionary
from fairseq.tasks import register_task


from avhubert.hubert_pretraining import AVHubertPretrainingConfig, AVHubertPretrainingTask, LabelEncoder
logger = logging.getLogger(__name__)


from .stage1_dataset import AVRomanDataset


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

        logger.debug(f"current directory is {os.getcwd()}")
        logger.debug(f"AV_RomanTask Config {cfg}")

        self.blank_symbol = "<s>"
        self.state.add_factory("target_dictionary", self.load_target_dictionary)

    def load_target_dictionary(self):
        if self.cfg.labels:
            from .training_config import DICTIONARY_PATH
            dict_path = str(DICTIONARY_PATH)
            logger.info('Using dict_path : {}'.format(dict_path))
            return Dictionary.load(dict_path)
        return None

    def load_dataset(self, split: str, **kwargs) -> None:
        manifest = f"{self.cfg.data}/{split}.tsv"
        dictionaries = [self.target_dictionary]
        pad_list = [dictionary.pad() for dictionary in dictionaries]
        eos_list = [dictionary.eos() for dictionary in dictionaries]

        procs = [LabelEncoder(dictionary) for dictionary in dictionaries]
        paths = [
            f"{self.get_label_dir()}/{split}.{l}" for l in self.cfg.labels
        ]
        image_aug = self.cfg.image_aug if split == 'train' else False

        self.datasets[split] = AVRomanDataset(
            manifest,
            sample_rate=self.cfg.sample_rate,
            label_paths=paths,
            label_rates=self.cfg.label_rate,
            pad_list=pad_list,
            eos_list=eos_list,
            label_processors=procs,
            max_keep_sample_size=self.cfg.max_sample_size,
            min_keep_sample_size=self.cfg.min_sample_size,
            max_sample_size=self.cfg.max_trim_sample_size,
            pad_audio=self.cfg.pad_audio,
            normalize=self.cfg.normalize,
            store_labels=True,
            random_crop=self.cfg.random_crop,
            single_target=self.cfg.single_target,
            stack_order_audio=self.cfg.stack_order_audio,
            skip_verify=self.cfg.skip_verify,
            image_mean=self.cfg.image_mean,
            image_std=self.cfg.image_std,
            image_crop_size=self.cfg.image_crop_size,
            image_aug=image_aug,
            modalities=self.cfg.modalities,
            is_s2s=self.cfg.is_s2s,
            time_mask=self.cfg.time_mask,
            random_erase=self.cfg.random_erase,
        )
