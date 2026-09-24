"""Fine-tune Laya on Thalamus gold traces (or custom JEV-like JSON).

**Unstable** -needs ``pip install "laya-thalamus[laya]"``. Entry: ``thalamus finetune``.
"""

from laya_thalamus.train.cli import main

__all__ = ["main"]
