"""
Configuración central para el pipeline de descarga de mamografías,
construcción del dataset y entrenamiento de la red neuronal.
Soporta lectura segura de variables de entorno mediante .env para GitHub.
"""

import os
from pathlib import Path

# Cargar variables de entorno desde .env si existe
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
except ImportError:
    pass

# Configuración de Base de Datos Institucional (Opcional - Módulo Privado)
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS", "")
DB_NAME = os.getenv("DB_NAME", "base_datos_hospital")
DB_SCHEMA = os.getenv("DB_SCHEMA", "public")

# Endpoints PACS Institucionales (Opcional - Módulo Privado)
BASE_URL = os.getenv("PACS_BASE_URL", "https://pacs.hospital-local.internal")
CONSULTA_URL = f"{BASE_URL}/consulta"
DOCUMENT_URL = f"{BASE_URL}/document"
VIEWER_FORM_URL = f"{BASE_URL}/document/viewer"
VISOR_BASE_URL = f"{BASE_URL}/visor"

USER_AGENT = os.getenv(
    "HTTP_USER_AGENT",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# Rutas de Almacenamiento Local (Configurable por variable de entorno)
DIRECTORIO_RAIZ_IMAGENES = os.getenv("DATASET_DIR", r"D:\RESULTADOS_SGIRD\imageneologia")
DIRECTORIO_DATASET_PROCESADO = os.path.join(DIRECTORIO_RAIZ_IMAGENES, "dataset_preprocesado")
MANIFEST_MAESTRO_CSV = os.path.join(DIRECTORIO_RAIZ_IMAGENES, "dataset_manifest.csv")
CARPETA_GRAFICAS = os.path.join(DIRECTORIO_RAIZ_IMAGENES, "graficas")

# Asegurar directorios base si existe el disco
try:
    os.makedirs(DIRECTORIO_RAIZ_IMAGENES, exist_ok=True)
    os.makedirs(CARPETA_GRAFICAS, exist_ok=True)
except Exception:
    pass
