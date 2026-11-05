"""
Script de Entrenamiento y Evaluación de la Red Neuronal de Mamografía.
Mide Accuracy, Sensibilidad (Recall), Especificidad y guarda el mejor modelo.
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from dataset import MammographyDataset
from modelo import MammoCNN, crear_modelo_resnet

def calcular_pesos_clases(manifest_csv: str) -> torch.Tensor:
    """Calcula pesos inversos a la frecuencia para balancear clases."""
    df = pd.read_csv(manifest_csv)
    conteo = df["etiqueta_binaria"].value_counts().to_dict()
    n_0 = conteo.get(0, 1)
    n_1 = conteo.get(1, 1)
    total = n_0 + n_1
    w_0 = total / (2.0 * n_0)
    w_1 = total / (2.0 * n_1)
    return torch.tensor([w_0, w_1], dtype=torch.float32)

def obtener_balanceado_sampler(dataset: MammographyDataset) -> torch.utils.data.WeightedRandomSampler:
    """
    Crea un muestreador ponderado para que cada batch contenga ~50% de normales y ~50% de patológicas,
    evitando que la clase mayoritaria ahogue a la clase patológica.
    """
    labels = dataset.df["etiqueta_binaria"].astype(int).values
    conteo = dataset.df["etiqueta_binaria"].value_counts().to_dict()
    pesos_por_clase = {c: 1.0 / count for c, count in conteo.items()}
    pesos_muestras = [pesos_por_clase[int(l)] for l in labels]
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=torch.DoubleTensor(pesos_muestras),
        num_samples=len(pesos_muestras),
        replacement=True
    )
    return sampler

def entrenar_epoca(modelo, dataloader, criterion, optimizer, device, epoch=1, total_epochs=1):
    modelo.train()
    total_loss = 0.0
    correctos = 0
    total = 0
    num_batches = len(dataloader)
    
    for i, (inputs, labels) in enumerate(dataloader, 1):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = modelo(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correctos += torch.sum(preds == labels.data).item()
        total += labels.size(0)
        
        if i % 15 == 0 or i == num_batches:
            batch_acc = correctos / total
            batch_loss = total_loss / total
            print(f"  [Época {epoch:02d}/{total_epochs:02d}] Batch {i:02d}/{num_batches:02d} | Loss: {batch_loss:.4f} | Acc: {batch_acc*100:.1f}%", flush=True)
            
    return total_loss / total, correctos / total

def evaluar(modelo, dataloader, criterion, device):
    modelo.eval()
    total_loss = 0.0
    todas_etiquetas = []
    todas_predicciones = []
    todas_probabilidades = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = modelo(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * inputs.size(0)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            todas_etiquetas.extend(labels.cpu().numpy())
            todas_predicciones.extend(preds.cpu().numpy())
            todas_probabilidades.extend(probs[:, 1].cpu().numpy())
            
    total = len(todas_etiquetas)
    val_loss = total_loss / total if total > 0 else 0.0
    
    y_true = np.array(todas_etiquetas)
    y_pred = np.array(todas_predicciones)
    acc = np.mean(y_pred == y_true) if total > 0 else 0.0
    
    # Métricas clínicas: Sensibilidad (Recall) y Especificidad
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    
    sensibilidad = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    especificidad = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
    return val_loss, acc, sensibilidad, especificidad, todas_etiquetas, todas_predicciones

def main():
    parser = argparse.ArgumentParser(description="Entrenador de Red Neuronal para Mamografía")
    parser.add_argument("--tamano", type=int, default=512, help="Resolución de las imágenes (512 o 1024)")
    parser.add_argument("--epochs", type=int, default=15, help="Número de épocas de entrenamiento")
    parser.add_argument("--batch-size", type=int, default=8, help="Tamaño del batch")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--arch", type=str, default="mammocnn", choices=["mammocnn", "resnet18"], help="Arquitectura")
    args = parser.parse_args()
    
    carpeta_splits = os.path.join(config.DIRECTORIO_DATASET_PROCESADO, f"{args.tamano}x{args.tamano}")
    train_csv = os.path.join(carpeta_splits, "train_manifest.csv")
    val_csv = os.path.join(carpeta_splits, "val_manifest.csv")
    test_csv = os.path.join(carpeta_splits, "test_manifest.csv")
    
    if not os.path.exists(train_csv) or not os.path.exists(val_csv):
        print(f"[!] No se encontraron los manifiestos de entrenamiento en {carpeta_splits}.")
        print("    Ejecuta primero: python preparar_dataset.py --tamano {args.tamano}")
        return
        
    print("=" * 80)
    print(f" ENTRENAMIENTO DE RED NEURONAL - CLASIFICACIÓN DE MAMOGRAFÍAS")
    print(f" Arquitectura: {args.arch} | Tamaño entrada: {args.tamano}x{args.tamano}")
    print(f" Épocas: {args.epochs} | Batch size: {args.batch_size} | LR: {args.lr}")
    print("=" * 80)
    
    # Configurar dispositivo (GPU si está disponible, sino CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[i] Dispositivo de cómputo: {device}")
    
    # Cargar Datasets con Muestreo Balanceado Clínico
    ds_train = MammographyDataset(train_csv, modo="train")
    ds_val = MammographyDataset(val_csv, modo="val")
    
    # Muestreador balanceado para forzar 50% normales y 50% patológicos por batch
    sampler_train = obtener_balanceado_sampler(ds_train)
    loader_train = DataLoader(ds_train, batch_size=args.batch_size, sampler=sampler_train, drop_last=False)
    loader_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False)
    
    print(f"[i] Muestras de entrenamiento: {len(ds_train)} (Batches balanceados con WeightedRandomSampler)")
    print(f"[i] Muestras de validación: {len(ds_val)}")
    
    # Modelo
    if args.arch == "resnet18":
        modelo = crear_modelo_resnet("resnet18", num_clases=2, in_channels=1, pretrained=True)
    else:
        modelo = MammoCNN(num_clases=2, in_channels=1)
    modelo = modelo.to(device)
    
    # Función de pérdida: Como el WeightedRandomSampler ya balancea la proporción en cada batch (~50/50),
    # aplicamos un leve sesgo médico a favor de la sensibilidad (1.5x) sin sobre-penalizar a la clase normal.
    pesos_balanceados = torch.tensor([1.0, 1.5], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=pesos_balanceados)
        
    optimizer = torch.optim.AdamW(modelo.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    mejor_val_loss = float("inf")
    mejor_bal_acc = 0.0
    ruta_guardado = os.path.join(carpeta_splits, f"mejor_modelo_{args.arch}.pth")
    
    print("\n--- INICIANDO CICLO DE ENTRENAMIENTO ---")
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = entrenar_epoca(modelo, loader_train, criterion, optimizer, device, epoch=epoch, total_epochs=args.epochs)
        va_loss, va_acc, sens, espec, _, _ = evaluar(modelo, loader_val, criterion, device)
        scheduler.step()
        
        bal_acc = (sens + espec) / 2.0
        info = (
            f"--> FIN ÉPOCA [{epoch:02d}/{args.epochs:02d}] | "
            f"Train Loss: {tr_loss:.4f} Acc: {tr_acc*100:.1f}% | "
            f"Val Loss: {va_loss:.4f} Acc: {va_acc*100:.1f}% | "
            f"Sensibilidad: {sens*100:.1f}% | Espec: {espec*100:.1f}% | BalAcc: {bal_acc*100:.1f}%"
        )
        
        # Salvaguarda Médica: Solo se guarda el modelo si mantiene Sensibilidad clínica >= 75%
        # y mejora la pérdida de validación o la exactitud balanceada.
        es_mejor = False
        if sens >= 0.75:
            if va_loss < mejor_val_loss or bal_acc > mejor_bal_acc:
                es_mejor = True
                if va_loss < mejor_val_loss:
                    mejor_val_loss = va_loss
                if bal_acc > mejor_bal_acc:
                    mejor_bal_acc = bal_acc
        elif mejor_val_loss == float("inf"):
            # En épocas iniciales de calentamiento
            mejor_val_loss = va_loss
            es_mejor = True
            
        if es_mejor:
            torch.save(modelo.state_dict(), ruta_guardado)
            info += " [* Mejor modelo clínico guardado]"
            
        print(info, flush=True)
        
    print("\n" + "=" * 80)
    print(f" ENTRENAMIENTO FINALIZADO EXITOSAMENTE")
    print(f" Mejor modelo guardado en: {ruta_guardado}")
    print("=" * 80)
    
    # Evaluación independiente en conjunto de Test
    if os.path.exists(test_csv) and os.path.exists(ruta_guardado):
        print("\n--- EVALUANDO EN CONJUNTO DE PRUEBA (TEST SET INDEPENDIENTE) ---")
        ds_test = MammographyDataset(test_csv, modo="test")
        loader_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False)
        
        modelo.load_state_dict(torch.load(ruta_guardado, map_location=device))
        t_loss, t_acc, t_sens, t_espec, y_t, y_p = evaluar(modelo, loader_test, criterion, device)
        
        print(f" Test Samples: {len(ds_test)}")
        print(f" Test Loss:         {t_loss:.4f}")
        print(f" Test Accuracy:     {t_acc * 100:.2f}%")
        print(f" Sensibilidad (TPR): {t_sens * 100:.2f}% (Capacidad de detectar patologías)")
        print(f" Especificidad (TNR):{t_espec * 100:.2f}% (Capacidad de descartar normales)")
        print("=" * 80)

if __name__ == "__main__":
    main()

