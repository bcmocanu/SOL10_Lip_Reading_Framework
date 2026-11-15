import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import faulthandler
faulthandler.enable()
from src.main_lip_reading import main

if __name__ == "__main__":
    main()
