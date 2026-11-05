"""
Dataset Loader personalizado para PyTorch sobre el manifiesto de Mamografías.
"""

import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class MammographyDataset(Dataset):
    """
    Dataset PyTorch que carga imágenes de mamografía a partir de un manifiesto CSV.
    """
    def __init__(self, manifest_csv: str, modo: str = "train", transform=None, objetivo: str = "binario"):
        self.df = pd.read_csv(manifest_csv)
        # Filtrar rutas válidas
        col_ruta = "ruta_procesada" if "ruta_procesada" in self.df.columns else "ruta_imagen"
        self.df = self.df[self.df[col_ruta].apply(lambda p: os.path.exists(str(p)))].reset_index(drop=True)
        self.col_ruta = col_ruta
        self.modo = modo
        self.objetivo = objetivo
        
        # Transformaciones por defecto si no se especifican
        if transform is not None:
            self.transform = transform
        else:
            if modo == "train":
                self.transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=8),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.25], std=[0.25])
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.25], std=[0.25])
                ])
                
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        ruta = row[self.col_ruta]
        
        # Cargar imagen en escala de grises
        with Image.open(ruta) as img:
            img_gray = img.convert("L")
            
        tensor_img = self.transform(img_gray)
        
        if self.objetivo == "binario":
            etiqueta = int(row["etiqueta_binaria"])
        else:
            # Multi-clase BI-RADS: mapear a enteros 0..5
            det = str(row["detalle_resultado"]).upper()
            if "BI-RADS 1" in det: etiqueta = 1
            elif "BI-RADS 2" in det: etiqueta = 2
            elif "BI-RADS 3" in det: etiqueta = 3
            elif "BI-RADS 4" in det: etiqueta = 4
            elif "BI-RADS 5" in det: etiqueta = 5
            else: etiqueta = 0 # BI-RADS 0 u otro
            
        return tensor_img, torch.tensor(etiqueta, dtype=torch.long)

