import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)

_DEFAULT_PATH = Path(__file__).parent / "settings.yaml"


@lru_cache(maxsize=1)
def load_config(path: str = str(_DEFAULT_PATH)) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    logger.debug("Configuration loaded from %s", path)
    return cfg
