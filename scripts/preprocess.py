"""
Script ejecutable de preprocesamiento, recorte de ROI y partición estratificada por paciente.
Uso:
    python scripts/preprocess.py --tamano 512
"""

import sys
import argparse
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.data.preprocessor import procesar_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tamano", type=int, default=512, help="Resolución en píxeles (default: 512)")
    args = parser.parse_args()
    procesar_dataset(tamano=args.tamano)
