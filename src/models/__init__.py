"""
Módulo de Modelos y Entrenamiento de Redes Neuronales.
"""

from .dataset import MammographyDataset
from .architectures import MammoCNN, crear_modelo_resnet
from .trainer import entrenar_epoca, evaluar

