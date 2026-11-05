"""
Script de Purga y Saneamiento del Dataset.
Detecta y elimina imágenes y registros que no correspondan a estudios mamográficos
(por ejemplo, radiografías de tórax PA/AP, columna, etc.).
"""

import os
import shutil
import pandas as pd

import config

PROYECCIONES_VALIDAS_MAMOGRAFIA = {
    "R-CC", "L-CC", "R-MLO", "L-MLO",
    "RCC", "LCC", "RMLO", "LMLO",
    "RL", "LL", "RL_2", "LL_2", "RL_3", "LL_3", "RL_4", "LL_4",
    "R-CC_2", "L-CC_2", "R-MLO_2", "L-MLO_2", "R-MLO_3", "L-MLO_3",
    "ML", "LM", "XCCL", "AT", "CV"
}

def purgar():
    manifest_path = config.MANIFEST_MAESTRO_CSV
    if not os.path.exists(manifest_path):
        print("[!] No existe el manifiesto.")
        return
        
    df = pd.read_csv(manifest_path)
    total_antes = len(df)
    
    # Identificar filas no mamográficas
    filtro_invalidos = (
        (~df["proyeccion"].isin(PROYECCIONES_VALIDAS_MAMOGRAFIA)) |
        (df["proyeccion"].isnull()) |
        (df["proyeccion"].str.strip() == "") |
        (df["proyeccion"].str.contains(r"PA|AP|TORAX", case=False, regex=True))
    )
    
    invalidos = df[filtro_invalidos].copy()
    validos = df[~filtro_invalidos].copy()
    
    print(f"Total registros analizados: {total_antes}")
    print(f"Total registros no mamográficos detectados: {len(invalidos)}")
    
    # Eliminar archivos en disco de los casos inválidos
    pacientes_afectados = set()
    for _, row in invalidos.iterrows():
        ruta_img = str(row["ruta_imagen"])
        cedula = str(row["identificacion"])
        pacientes_afectados.add(cedula)
        
        # Eliminar imagen raw si existe
        if os.path.exists(ruta_img):
            try:
                os.remove(ruta_img)
                print(f"  [ELIMINADO RAW] {ruta_img}")
            except Exception as e:
                print(f"  [!] Error eliminando {ruta_img}: {e}")
                
    # Eliminar carpeta completa del paciente si quedó vacía o corrupta
    for cedula in pacientes_afectados:
        dir_pac = os.path.join(config.DIRECTORIO_RAIZ_IMAGENES, cedula)
        if os.path.exists(dir_pac):
            try:
                shutil.rmtree(dir_pac)
                print(f"  [CARPETA PURGADA] {dir_pac}")
            except Exception as e:
                print(f"  [!] Error borrando carpeta {dir_pac}: {e}")
                
    # Eliminar imágenes preprocesadas si existen
    carpeta_proc = os.path.join(config.DIRECTORIO_DATASET_PROCESADO, "512x512", "imagenes")
    if os.path.exists(carpeta_proc):
        for f in os.listdir(carpeta_proc):
            for cedula in pacientes_afectados:
                if f.startswith(f"{cedula}_") or "_PA_" in f or "_AP_" in f:
                    ruta_p = os.path.join(carpeta_proc, f)
                    try:
                        os.remove(ruta_p)
                        print(f"  [ELIMINADO PROCESADO] {ruta_p}")
                    except Exception:
                        pass
                        
    # Sobrescribir manifest maestro solo con datos válidos
    validos.to_csv(manifest_path, index=False, encoding="utf-8")
    print(f"\n[OK] Manifiesto maestro purgado: {len(validos)} registros legítimos restantes.")

if __name__ == "__main__":
    purgar()

