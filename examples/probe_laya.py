"""
探测真实 Laya 是否可用（仓库入口）?
pip 安装后请?:

    thalamus probe-laya
    python -m laya_thalamus.probe
"""

from __future__ import annotations

import sys

from laya_thalamus.probe import main

if __name__ == "__main__":
    raise SystemExit(main())
