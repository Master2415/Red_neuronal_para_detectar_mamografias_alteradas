"""
Script de Inferencia Clínica para Mamografías.
Permite evaluar una imagen individual o una carpeta completa de un paciente.
Retorna la predicción (Normal vs. Sospechoso/Patológico) con porcentaje de certeza.
"""

import os
import sys
import argparse
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

try:
    from src import config
    from src.models.architectures import MammoCNN
    from src.data.preprocessor import recortar_tejido_mamario
except ImportError:
    import config
    from modelo import MammoCNN
    from preparar_dataset import recortar_tejido_mamario

def cargar_modelo(ruta_pesos: str, device: torch.device):
    modelo = MammoCNN(num_clases=2, in_channels=1)
    if not os.path.exists(ruta_pesos):
        raise FileNotFoundError(f"No se encontró el archivo de pesos: {ruta_pesos}")
    modelo.load_state_dict(torch.load(ruta_pesos, map_location=device))
    modelo.to(device)
    modelo.eval()
    return modelo

def predecir_imagen(ruta_imagen: str, modelo, device, tamano: int = 512):
    """Procesa una imagen y retorna probabilidades y clase predicha."""
    if not os.path.exists(ruta_imagen):
        return None
        
    with Image.open(ruta_imagen) as img:
        img_gray = img.convert("L")
        img_recortada = recortar_tejido_mamario(img_gray)
        img_final = img_recortada.resize((tamano, tamano), Image.Resampling.LANCZOS)
        
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.25], std=[0.25])
    ])
    
    tensor_img = transform(img_final).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = modelo(tensor_img)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        pred_clase = int(np.argmax(probs))
        
    prob_normal = probs[0] * 100.0
    prob_sospechoso = probs[1] * 100.0
    
    return {
        "ruta": ruta_imagen,
        "clase_predicha": pred_clase,
        "etiqueta": "Sospechoso / Patológico (BI-RADS 4/5)" if pred_clase == 1 else "Normal / Benigno (BI-RADS 1/2/3)",
        "prob_normal": prob_normal,
        "prob_sospechoso": prob_sospechoso
    }

def main():
    ruta_local = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "mejor_modelo_mammocnn.pth")
    ruta_dataset = r"D:\RESULTADOS_SGIRD\imageneologia\dataset_preprocesado\512x512\mejor_modelo_mammocnn.pth"
    modelo_defecto = ruta_local if os.path.exists(ruta_local) else ruta_dataset

    parser = argparse.ArgumentParser(description="Inferencia y Diagnóstico Asistido por IA en Mamografías")
    parser.add_argument("--imagen", type=str, help="Ruta a una imagen individual (.jpg, .png)")
    parser.add_argument("--carpeta", type=str, help="Ruta a una carpeta de paciente con múltiples proyecciones")
    parser.add_argument("--modelo", type=str, default=modelo_defecto, help="Ruta al archivo .pth del modelo")
    parser.add_argument("--tamano", type=int, default=512, help="Resolución de la imagen")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 75)
    print(" DIAGNÓSTICO ASISTIDO POR IA - INFERENCIA DE MAMOGRAFÍA")
    print(f" Dispositivo: {device} | Modelo: {os.path.basename(args.modelo)}")
    print("=" * 75)

    try:
        modelo = cargar_modelo(args.modelo, device)
    except Exception as e:
        print(f"[!] Error cargando el modelo: {e}")
        return

    rutas_a_evaluar = []
    if args.imagen:
        if not os.path.exists(args.imagen):
            print(f"[!] Archivo de imagen no encontrado: {args.imagen}")
            return
        rutas_a_evaluar.append(args.imagen)
    elif args.carpeta:
        if not os.path.exists(args.carpeta):
            print(f"[!] Carpeta no encontrada: {args.carpeta}")
            return
        for root, _, files in os.walk(args.carpeta):
            for f in files:
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    rutas_a_evaluar.append(os.path.join(root, f))
    else:
        print("Por favor especifica una imagen con --imagen o una carpeta con --carpeta.")
        return

    if not rutas_a_evaluar:
        print("[!] No se encontraron imágenes para analizar.")
        return

    print(f"\nAnalizando {len(rutas_a_evaluar)} proyección(es)...")
    print("-" * 75)

    paciente_sospechoso = False
    max_riesgo = 0.0

    for r in rutas_a_evaluar:
        res = predecir_imagen(r, modelo, device, tamano=args.tamano)
        if res:
            nombre = os.path.basename(r)
            riesgo = res["prob_sospechoso"]
            if riesgo > max_riesgo:
                max_riesgo = riesgo
            if res["clase_predicha"] == 1:
                paciente_sospechoso = True
                
            estado_tag = "[ALERTA]" if res["clase_predicha"] == 1 else "[NORMAL]"
            print(f" {estado_tag} {nombre:20s} | {res['etiqueta']:35s} | Riesgo: {riesgo:5.1f}%")

    print("-" * 75)
    print(" CONCLUSIÓN DEL ESTUDIO:")
    if paciente_sospechoso:
        print(f"  --> RIESGO ELEVADO DETECTADO (Riesgo máximo en vistas: {max_riesgo:.1f}%)")
        print("      Se recomienda revisión prioritaria por radiólogo especialista.")
    else:
        print(f"  --> ESTUDIO CON ALTA PROBABILIDAD DE SER BENIGNO / NORMAL (Riesgo máximo: {max_riesgo:.1f}%)")
    print("=" * 75)

if __name__ == "__main__":
    main()

