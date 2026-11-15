#!/usr/bin/env python3
from __future__ import annotations

import logging
import sys


def main() -> int:
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("setup")
    try:
        logger.info("Pregatire framework...")
        from src.training_setup import setup_framework
        setup_framework()
        return 0
    except KeyboardInterrupt:
        logger.error("Pregatirea a fost intrerupta. Poate fi reluata cu aceeasi comanda.")
        return 130
    except (ImportError, ValueError, OSError, RuntimeError) as exc:
        logger.error("%s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
