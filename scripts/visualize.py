"""
Script ejecutable para regenerar todas las gráficas científicas a 300 DPI.
Uso:
    python scripts/visualize.py
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.utils import visualization

if __name__ == "__main__":
    visualization.graficar_curvas_entrenamiento()
    visualization.graficar_matriz_confusion()
    visualization.graficar_distribucion_dataset()
    visualization.graficar_distribucion_probabilidades()
    print("[OK] Gráficas generadas exitosamente en graficas/.")
