"""
对比评测（仓库入口）。实现已迁入 ``laya_thalamus.eval.compare``，
pip 安装后请用::

    thalamus compare
    python -m laya_thalamus.eval.compare
"""

from __future__ import annotations

import sys

from laya_thalamus.eval.compare import main

if __name__ == "__main__":
    sys.exit(main())
