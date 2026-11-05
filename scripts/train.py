"""
Script ejecutable de entrenamiento y validación de la red neuronal MammoCNN.
Uso:
    python scripts/train.py --tamano 512 --epochs 10 --batch-size 16
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.models import trainer

if __name__ == "__main__":
    trainer.main()
