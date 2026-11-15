

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

for file in sorted(os.listdir(os.path.dirname(__file__))):
    if file.endswith(".py") and not file.startswith("_"):
        module = file[: file.find(".py")]
        importlib.import_module("fairseq.data.encoders." + module)
