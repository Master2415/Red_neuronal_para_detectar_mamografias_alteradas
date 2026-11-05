"""
Arquitecturas de Redes Neuronales para Clasificación de Mamografías.
Soporta:
1. Transfer Learning con ResNet18 / ResNet50 adaptado a mamografía.
2. CNN Especializada ligera (MammoCNN) optimizada para inferencia y entrenamiento rápido en CPU.
"""

import torch
import torch.nn as nn
import torchvision.models as models

class MammoCNN(nn.Module):
    """
    Red Neuronal Convolucional personalizada y eficiente para mamografía en escala de grises.
    Diseñada con bloques convolucionales residuales, MaxPooling, Dropout y Batch Normalization.
    """
    def __init__(self, num_clases: int = 2, in_channels: int = 1):
        super(MammoCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Bloque 1: Entrada 512x512 -> 256x256
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Bloque 2: 256x256 -> 128x128
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Bloque 3: 128x128 -> 64x64
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Bloque 4: 64x64 -> 32x32
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Bloque 5: 32x32 -> 16x16
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Adaptive pooling para fijar tamaño sin importar resolución de entrada
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_clases)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x

def crear_modelo_resnet(nombre: str = "resnet18", num_clases: int = 2, in_channels: int = 1, pretrained: bool = True):
    """
    Crea un modelo ResNet adaptado para mamografías.
    Modifica la primera capa para aceptar 1 canal en escala de grises y la cabeza final.
    """
    if nombre == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
    else:
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        
    # Adaptar primera capa convolucional para escala de grises
    if in_channels != 3:
        original_conv = model.conv1
        new_conv = nn.Conv2d(
            in_channels,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None
        )
        if pretrained:
            with torch.no_grad():
                # Promediar los pesos RGB para inicializar el canal único
                new_conv.weight[:] = original_conv.weight.sum(dim=1, keepdim=True) / 3.0
        model.conv1 = new_conv
        
    # Reemplazar la capa fully connected final
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_clases)
    )
    
    return model

