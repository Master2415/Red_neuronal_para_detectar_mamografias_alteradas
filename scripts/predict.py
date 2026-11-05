"""
Script ejecutable de inferencia y diagnóstico clínico.
Uso:
    python scripts/predict.py --carpeta "D:\\RESULTADOS_SGIRD\\imageneologia\\65552684"
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src import predict

if __name__ == "__main__":
    predict.main()
