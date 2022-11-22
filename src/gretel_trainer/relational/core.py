from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


class MultiTableException(Exception):
    pass


# rdb_config: Dict[str, Any]
#   table_data: Dict[str, pd.DataFrame]
#   table_files: Dict[str, str]
#   primary_keys: Dict[str, str]
#   relationships: List[List[Tuple[str, str]]]
