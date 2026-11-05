"""
Módulo de Preprocesamiento y Creación de Splits (Train / Val / Test)
para el Dataset de Mamografías.
- Recorta automáticamente el fondo negro (Detección de ROI del tejido mamario).
- Normaliza y redimensiona a resolución óptima (ej. 512x512 o 1024x1024).
- Realiza partición estratificada A NIVEL DE PACIENTE para evitar fuga de datos (Data Leakage).
"""

import os
import csv
import json
import random
import argparse
import numpy as np
import pandas as pd
from PIL import Image

import config

PROYECCIONES_VALIDAS_MAMOGRAFIA = {
    "R-CC", "L-CC", "R-MLO", "L-MLO",
    "RCC", "LCC", "RMLO", "LMLO",
    "RL", "LL", "RL_2", "LL_2", "RL_3", "LL_3", "RL_4", "LL_4",
    "R-CC_2", "L-CC_2", "R-MLO_2", "L-MLO_2", "R-MLO_3", "L-MLO_3",
    "ML", "LM", "XCCL", "AT", "CV"
}

def recortar_tejido_mamario(img: Image.Image, umbral: int = 15, margen: int = 10) -> Image.Image:
    """
    Elimina el fondo negro exterior y recorta el rectángulo que contiene el tejido mamario.
    """
    arr = np.array(img)
    # Máscara binaria donde el valor del píxel es mayor al umbral de fondo
    mascara = arr > umbral
    
    filas = np.any(mascara, axis=1)
    cols = np.any(mascara, axis=0)
    
    if not np.any(filas) or not np.any(cols):
        return img  # Imagen completamente oscura o atípica
        
    ymin, ymax = np.where(filas)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    
    # Aplicar margen respetando límites
    ymin = max(0, ymin - margen)
    ymax = min(arr.shape[0], ymax + margen)
    xmin = max(0, xmin - margen)
    xmax = min(arr.shape[1], xmax + margen)
    
    recortada = img.crop((xmin, ymin, xmax, ymax))
    return recortada

def procesar_dataset(
    manifest_csv: str = config.MANIFEST_MAESTRO_CSV,
    directorio_salida: str = config.DIRECTORIO_DATASET_PROCESADO,
    tamano: int = 512,
    split_train: float = 0.70,
    split_val: float = 0.15,
    split_test: float = 0.15,
    semilla: int = 42
):
    """
    Procesa todas las imágenes del manifiesto, realiza splits y guarda el dataset listo para entrenar.
    """
    if not os.path.exists(manifest_csv):
        print(f"[!] No existe el manifiesto maestro: {manifest_csv}")
        return
        
    df = pd.read_csv(manifest_csv)
    # Filtrar solo registros cuyas imágenes existan en disco y sean proyecciones mamográficas legítimas
    df = df[df["ruta_imagen"].apply(lambda p: os.path.exists(str(p)) and os.path.getsize(str(p)) > 10000)].copy()
    
    # Filtro estricto de mamografía
    df = df[df["proyeccion"].isin(PROYECCIONES_VALIDAS_MAMOGRAFIA)].copy()
    df = df[~df["proyeccion"].astype(str).str.contains(r"PA|AP|TORAX", case=False, regex=True)].copy()
    
    if len(df) == 0:
        print("[!] No hay imágenes válidas descargadas en el manifiesto.")
        return
        
    print("=" * 80)
    print(f" PREPROCESANDO DATASET DE MAMOGRAFÍAS ({tamano}x{tamano})")
    print(f" Total de imágenes válidas encontradas: {len(df)}")
    print(f" Total de pacientes únicos: {df['identificacion'].nunique()}")
    print("=" * 80)
    
    # Directorios de salida
    subcarpeta_tamano = os.path.join(directorio_salida, f"{tamano}x{tamano}")
    carpeta_imagenes = os.path.join(subcarpeta_tamano, "imagenes")
    os.makedirs(carpeta_imagenes, exist_ok=True)
    
    # 1. Preprocesar y guardar imágenes en resolución objetivo
    rutas_procesadas = []
    print("\n[>] Recortando fondo y redimensionando imágenes...")
    
    for idx, row in df.iterrows():
        ruta_orig = row["ruta_imagen"]
        cedula = str(row["identificacion"])
        cod_arch = str(row["codigo_archivo"])
        proy = str(row["proyeccion"])
        sop_uid = str(row.get("sop_uid", "")).split(".")[-1]
        
        nombre_salida = f"{cedula}_{cod_arch}_{proy}_{sop_uid}.png"
        ruta_salida = os.path.join(carpeta_imagenes, nombre_salida)
        
        if not os.path.exists(ruta_salida):
            try:
                with Image.open(ruta_orig) as img:
                    img_gray = img.convert("L")
                    # Recorte de fondo
                    img_recortada = recortar_tejido_mamario(img_gray)
                    # Resize de alta calidad
                    img_final = img_recortada.resize((tamano, tamano), Image.Resampling.LANCZOS)
                    img_final.save(ruta_salida, format="PNG", optimize=True)
            except Exception as e:
                print(f"  [!] Error procesando {ruta_orig}: {e}")
                rutas_procesadas.append(None)
                continue
                
        rutas_procesadas.append(ruta_salida)
        
    df["ruta_procesada"] = rutas_procesadas
    df = df[df["ruta_procesada"].notnull()].copy()
    
    # 2. Partición Estratificada a Nivel de Paciente (Patient-Level Split)
    # Evita que vistas del mismo paciente caigan simultáneamente en train y test
    random.seed(semilla)
    pacientes_unicos = df[["identificacion", "etiqueta_binaria"]].drop_duplicates()
    
    pacientes_normales = pacientes_unicos[pacientes_unicos["etiqueta_binaria"] == 0]["identificacion"].tolist()
    pacientes_alterados = pacientes_unicos[pacientes_unicos["etiqueta_binaria"] == 1]["identificacion"].tolist()
    
    random.shuffle(pacientes_normales)
    random.shuffle(pacientes_alterados)
    
    def particionar(lista, p_train, p_val):
        n = len(lista)
        if n == 0:
            return [], [], []
        if n == 1:
            return lista, [], []
        if n == 2:
            return [lista[0]], [], [lista[1]]
        if n == 3:
            return [lista[0]], [lista[1]], [lista[2]]
            
        n_va = max(1, int(round(n * p_val)))
        n_te = max(1, int(round(n * (1.0 - p_train - p_val))))
        n_tr = max(1, n - n_va - n_te)
        
        return lista[:n_tr], lista[n_tr:n_tr + n_va], lista[n_tr + n_va:]
        
    tr_norm, va_norm, te_norm = particionar(pacientes_normales, split_train, split_val)
    tr_alt, va_alt, te_alt = particionar(pacientes_alterados, split_train, split_val)
    
    train_pacs = set(tr_norm + tr_alt)
    val_pacs = set(va_norm + va_alt)
    test_pacs = set(te_norm + te_alt)
    
    def asignar_split(pac):
        if pac in train_pacs: return "train"
        if pac in val_pacs: return "val"
        return "test"
        
    df["split"] = df["identificacion"].apply(asignar_split)
    
    # 3. Guardar Manifiestos de cada partición
    ruta_manifest_proc = os.path.join(subcarpeta_tamano, "manifest_procesado.csv")
    df.to_csv(ruta_manifest_proc, index=False, encoding="utf-8")
    
    df_train = df[df["split"] == "train"]
    df_val = df[df["split"] == "val"]
    df_test = df[df["split"] == "test"]
    
    df_train.to_csv(os.path.join(subcarpeta_tamano, "train_manifest.csv"), index=False, encoding="utf-8")
    df_val.to_csv(os.path.join(subcarpeta_tamano, "val_manifest.csv"), index=False, encoding="utf-8")
    df_test.to_csv(os.path.join(subcarpeta_tamano, "test_manifest.csv"), index=False, encoding="utf-8")
    
    print("\n" + "=" * 80)
    print(" RESUMEN DEL DATASET GENERADO")
    print(f" Train: {len(df_train)} imágenes ({df_train['identificacion'].nunique()} pacientes)")
    print(f" Val:   {len(df_val)} imágenes ({df_val['identificacion'].nunique()} pacientes)")
    print(f" Test:  {len(df_test)} imágenes ({df_test['identificacion'].nunique()} pacientes)")
    print(f"\n Manifiestos guardados en: {subcarpeta_tamano}")
    print("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tamano", type=int, default=512, help="Resolución en píxeles (ej. 512 o 1024)")
    args = parser.parse_args()
    procesar_dataset(tamano=args.tamano)
