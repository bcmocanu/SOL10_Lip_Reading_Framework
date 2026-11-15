"""Local training components."""
import os

for _name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE",
              "HF_HUB_DISABLE_TELEMETRY", "DO_NOT_TRACK"):
    os.environ[_name] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
