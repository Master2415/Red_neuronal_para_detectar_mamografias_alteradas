"""
Script de Generación de Gráficas Científicas y Métricas Clínicas.
Produce figuras de alta resolución basadas en los principios de las diapositivas:
1. Curvas de Aprendizaje (Pérdida y Accuracy) con detección de Overfitting y Early Stopping.
2. Matriz de Confusión Clínica (Test Set Ciego) con Sensibilidad y Especificidad.
3. Distribución y Balance del Dataset (BI-RADS, Vistas Anatómicas, Particiones).
4. Distribución de Probabilidades de Riesgo en Inferencia Clínica.
"""

import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
from torch.utils.data import DataLoader

try:
    from src import config
    from src.models.architectures import MammoCNN
    from src.models.dataset import MammographyDataset
    from src.models.trainer import evaluar
except ImportError:
    import config
    from modelo import MammoCNN
    from dataset import MammographyDataset
    from entrenar import evaluar

# Configurar estilo visual limpio y profesional
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

CARPETA_GRAFICAS = r"D:\RESULTADOS_SGIRD\imageneologia\graficas"
CARPETA_ARTIFACTS = r"C:\Users\USUARIO\.gemini\antigravity\brain\27346eb3-d703-4182-8f22-474e923d7ba0"
os.makedirs(CARPETA_GRAFICAS, exist_ok=True)

# -------------------------------------------------------------------------------------
# 1. CURVAS DE ENTRENAMIENTO Y EARLY STOPPING (Slide P67-77)
# -------------------------------------------------------------------------------------
def graficar_curvas_entrenamiento():
    epocas = np.arange(1, 6)
    
    # Datos registrados del ciclo de entrenamiento real sobre 2,424 imágenes (Task 813)
    train_loss = [0.4288, 0.3960, 0.3507, 0.3588, 0.3127]
    val_loss   = [0.8305, 0.4311, 0.2721, 0.3087, 0.4497]
    
    train_acc  = [79.9, 79.6, 82.8, 83.1, 86.6]
    val_acc    = [47.8, 80.4, 91.2, 88.4, 78.2]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    
    # Gráfica de Pérdida (Loss)
    ax1.plot(epocas, train_loss, 'o-', color='#1f77b4', linewidth=2.2, label='Train Loss (Entrenamiento)')
    ax1.plot(epocas, val_loss, 's--', color='#d62728', linewidth=2.2, label='Val Loss (Validación)')
    ax1.axvline(x=3, color='#2ca02c', linestyle=':', linewidth=2, label='Early Stopping / Mejor Modelo (Época 3)')
    ax1.annotate('Punto Óptimo\n(Val Loss: 0.2721\nVal Acc: 91.2%)', xy=(3, 0.2721), xytext=(3.3, 0.55),
                 arrowprops=dict(facecolor='#2ca02c', shrink=0.08, width=1.5, headwidth=7),
                 fontsize=10, fontweight='bold', color='#1e7e34')
    ax1.set_title('Evolución de la Función de Coste (Dataset 2,424 Mamografías)', fontsize=12, fontweight='bold', pad=12)
    ax1.set_xlabel('Época', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.set_xticks(epocas)
    ax1.legend(frameon=True, facecolor='white', framealpha=0.9)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Gráfica de Exactitud (Accuracy)
    ax2.plot(epocas, train_acc, 'o-', color='#1f77b4', linewidth=2.2, label='Train Accuracy')
    ax2.plot(epocas, val_acc, 's--', color='#2ca02c', linewidth=2.2, label='Val Accuracy')
    ax2.axvline(x=3, color='#2ca02c', linestyle=':', linewidth=2, label='Mejor Modelo (Val Acc: 91.2%)')
    ax2.set_title('Evolución de la Exactitud (Accuracy)', fontsize=12, fontweight='bold', pad=12)
    ax2.set_xlabel('Época', fontsize=11)
    ax2.set_ylabel('Exactitud (%)', fontsize=11)
    ax2.set_xticks(epocas)
    ax2.set_ylim(40, 100)
    ax2.legend(frameon=True, facecolor='white', framealpha=0.9)
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    ruta_guardado = os.path.join(CARPETA_GRAFICAS, "curvas_entrenamiento_early_stopping.png")
    fig.savefig(ruta_guardado, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Curvas de entrenamiento guardadas en: {ruta_guardado}")
    return ruta_guardado

# -------------------------------------------------------------------------------------
# 2. MATRIZ DE CONFUSIÓN Y MÉTRICAS CLÍNICAS (Slide P57)
# -------------------------------------------------------------------------------------
def graficar_matriz_confusion():
    # Resultados reales del test set ciego (364 imágenes / 82 pacientes no vistos):
    # Total patológicas reales (clase 1) = 27 -> TP = 19 (70.37% Sensibilidad), FN = 8
    # Total normales reales (clase 0) = 337 -> TN = 297 (88.13% Especificidad), FP = 40
    # Test Accuracy Global = 86.81%
    cm = np.array([
        [297, 40], # Normales
        [8, 19]    # Patológicos
    ])
    
    fig, ax = plt.subplots(figsize=(7.5, 6), dpi=300)
    cax = ax.matshow(cm, cmap='Blues', alpha=0.85)
    
    # Añadir valores y etiquetas explicativas
    classes = ['Normal / Benigno\n(BI-RADS 1, 2, 3)', 'Sospechoso\n(BI-RADS 4, 5)']
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(classes, fontsize=11, fontweight='bold')
    ax.set_yticklabels(classes, fontsize=11, fontweight='bold')
    
    # Textos dentro de cada celda
    labels_desc = [
        [f"Verdaderos Negativos (TN)\n{cm[0,0]}\n(Descarte correcto: 88.1%)", f"Falsos Positivos (FP)\n{cm[0,1]}\n(Sobre-alerta benigna)"],
        [f"Falsos Negativos (FN)\n{cm[1,0]}\n(Subestimación)", f"Verdaderos Positivos (TP)\n{cm[1,1]}\n(Detección exitosa: 70.4%)"]
    ]
    
    for i in range(2):
        for j in range(2):
            color_txt = "white" if cm[i, j] > 100 else "#111111"
            ax.text(j, i, labels_desc[i][j], ha="center", va="center", color=color_txt, fontsize=11, fontweight='bold')
            
    fig.colorbar(cax, shrink=0.8)
    ax.set_title("Matriz de Confusión Clínica - Test Ciego Independiente\n(364 Mamografías / 82 Pacientes No Vistos | Acc: 86.8%)",
                 fontsize=12, fontweight='bold', pad=20)
    ax.set_xlabel("Predicción de la Red Neuronal (MammoCNN)", fontsize=11, labelpad=10)
    ax.set_ylabel("Diagnóstico Radiológico Real (Ground Truth)", fontsize=11, labelpad=10)
    
    plt.tight_layout()
    ruta_guardado = os.path.join(CARPETA_GRAFICAS, "matriz_confusion_clinica.png")
    fig.savefig(ruta_guardado, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Matriz de confusión guardada en: {ruta_guardado}")
    return ruta_guardado

# -------------------------------------------------------------------------------------
# 3. DISTRIBUCIÓN CLÍNICA DEL DATASET (Slide P20-29 KDD/CRISP-DM)
# -------------------------------------------------------------------------------------
def graficar_distribucion_dataset():
    manifest_csv = config.MANIFEST_MAESTRO_CSV
    df = pd.read_csv(manifest_csv)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    
    # 1. Distribución por BI-RADS
    conteo_birads = df['detalle_resultado'].value_counts()
    colores_birads = ['#2ca02c', '#33a02c', '#ff7f0e', '#d62728', '#e31a1c', '#9467bd', '#7f7f7f', '#1f77b4']
    bars1 = ax1.bar(conteo_birads.index, conteo_birads.values, color=colores_birads[:len(conteo_birads)], edgecolor='#333333', alpha=0.85)
    ax1.set_title('Distribución de Imágenes por Clasificación BI-RADS', fontsize=12, fontweight='bold', pad=12)
    ax1.set_xlabel('Categoría Diagnóstica', fontsize=11)
    ax1.set_ylabel('Número de Imágenes', fontsize=11)
    ax1.tick_params(axis='x', rotation=30)
    
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 5, f"{int(yval)}", ha='center', va='bottom', fontsize=9, fontweight='bold')
        
    # 2. Distribución de Partición Estratificada a Nivel Paciente (Lectura Dinámica)
    carpeta_tamano = os.path.join(config.DIRECTORIO_DATASET_PROCESADO, "512x512")
    df_tr = pd.read_csv(os.path.join(carpeta_tamano, "train_manifest.csv"))
    df_va = pd.read_csv(os.path.join(carpeta_tamano, "val_manifest.csv"))
    df_te = pd.read_csv(os.path.join(carpeta_tamano, "test_manifest.csv"))
    
    splits = ['Entrenamiento (Train 70%)', 'Validación (Val 15%)', 'Prueba Ciega (Test 15%)']
    conteo_splits = [len(df_tr), len(df_va), len(df_te)]
    colores_split = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    bars2 = ax2.bar(splits, conteo_splits, color=colores_split, edgecolor='#333333', alpha=0.85, width=0.55)
    ax2.set_title('Partición Estratificada a Nivel de Paciente (Zero Leakage)', fontsize=12, fontweight='bold', pad=12)
    ax2.set_ylabel('Número de Imágenes', fontsize=11)
    
    pacs_text = [f"{df_tr['identificacion'].nunique()} Pacientes",
                 f"{df_va['identificacion'].nunique()} Pacientes",
                 f"{df_te['identificacion'].nunique()} Pacientes"]
    for i, bar in enumerate(bars2):
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + (max(conteo_splits)*0.02), f"{int(yval)} imgs\n({pacs_text[i]})",
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
                 
    ax2.set_ylim(0, max(conteo_splits) * 1.2)
    plt.tight_layout()
    ruta_guardado = os.path.join(CARPETA_GRAFICAS, "distribucion_dataset_estratificado.png")
    fig.savefig(ruta_guardado, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Distribución de dataset guardada en: {ruta_guardado}")
    return ruta_guardado

# -------------------------------------------------------------------------------------
# 4. DISTRIBUCIÓN DE RIESGO PREDICHO EN EL TEST SET
# -------------------------------------------------------------------------------------
def graficar_distribucion_probabilidades():
    test_csv = os.path.join(config.DIRECTORIO_DATASET_PROCESADO, "512x512", "test_manifest.csv")
    pesos_pth = os.path.join(config.DIRECTORIO_DATASET_PROCESADO, "512x512", "mejor_modelo_mammocnn.pth")
    
    device = torch.device("cpu")
    modelo = MammoCNN(num_clases=2, in_channels=1)
    modelo.load_state_dict(torch.load(pesos_pth, map_location=device))
    modelo.eval()
    
    ds_test = MammographyDataset(test_csv, modo="test")
    loader_test = DataLoader(ds_test, batch_size=8, shuffle=False)
    
    probs_clase1 = []
    etiquetas = []
    with torch.no_grad():
        for x, y in loader_test:
            out = modelo(x)
            pr = torch.softmax(out, dim=1)[:, 1].numpy()
            probs_clase1.extend(pr * 100.0)
            etiquetas.extend(y.numpy())
            
    df_res = pd.DataFrame({'riesgo': probs_clase1, 'etiqueta': etiquetas})
    riesgo_normales = df_res[df_res['etiqueta'] == 0]['riesgo']
    riesgo_patologicos = df_res[df_res['etiqueta'] == 1]['riesgo']
    
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    ax.hist(riesgo_normales, bins=15, alpha=0.65, color='#2ca02c', label=f'Casos Normales/Benignos (Mediana: {riesgo_normales.median():.1f}%)', edgecolor='white')
    ax.hist(riesgo_patologicos, bins=15, alpha=0.75, color='#d62728', label=f'Casos Sospechosos/Patológicos (Mediana: {riesgo_patologicos.median():.1f}%)', edgecolor='white')
    
    ax.axvline(x=50.0, color='#333333', linestyle='--', linewidth=1.5, label='Umbral de Decisión (50%)')
    ax.set_title('Distribución de Probabilidad de Riesgo Asignada por MammoCNN\n(Evaluación Ciega en Test Set)', fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel('Probabilidad de Sospecha / Malignidad (%)', fontsize=11)
    ax.set_ylabel('Frecuencia de Imágenes', fontsize=11)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    ruta_guardado = os.path.join(CARPETA_GRAFICAS, "distribucion_riesgo_predicho.png")
    fig.savefig(ruta_guardado, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Distribución de riesgo guardada en: {ruta_guardado}")
    return ruta_guardado

def copiar_a_artifacts(rutas):
    for r in rutas:
        destino = os.path.join(CARPETA_ARTIFACTS, os.path.basename(r))
        shutil.copy2(r, destino)
        print(f"[+] Copiada gráfica a carpeta de artifacts: {destino}")

if __name__ == "__main__":
    r1 = graficar_curvas_entrenamiento()
    r2 = graficar_matriz_confusion()
    r3 = graficar_distribucion_dataset()
    r4 = graficar_distribucion_probabilidades()
    copiar_a_artifacts([r1, r2, r3, r4])

