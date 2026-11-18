"""数据根目录单一真源: 启动时显式注入(TDR_DATA_DIR), 未注入回退相对路径"data"。
所有服务的 data/ 路径一律经此模块, 禁止再出现第二处硬编码。"""

import os
from pathlib import Path

DATA_ROOT = Path(os.environ.get("TDR_DATA_DIR", "data"))
