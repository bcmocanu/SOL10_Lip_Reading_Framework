import importlib
import os

from fairseq import registry


build_tokenizer, register_tokenizer, TOKENIZER_REGISTRY, _ = registry.setup_registry(
    "--tokenizer",
    default=None,
)


build_bpe, register_bpe, BPE_REGISTRY, _ = registry.setup_registry(
    "--bpe",
    default=None,
)


from . import sentencepiece_bpe
