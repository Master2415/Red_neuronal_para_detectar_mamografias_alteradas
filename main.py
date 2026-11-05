"""
Punto de Entrada Unificado (CLI) para el Sistema MammoCNN.
Permite ejecutar inferencia, entrenamiento, preprocesamiento y visualización.
"""

import os
import sys
import argparse

def imprimir_guia_rapida():
    print("=" * 75)
    print(" MammoCNN: Sistema de Clasificación y Triaje de Mamografías por IA")
    print("=" * 75)
    print("El modelo ya se encuentra entrenado (pesos en models/mejor_modelo_mammocnn.pth).")
    print("No necesitas descargar ningún dataset para utilizarlo.\n")
    print("MODO DE USO RÁPIDO (DIAGNÓSTICO EN TIEMPO REAL):")
    print("  1. Diagnosticar una imagen individual:")
    print("     python main.py --imagen \"ruta/a/la/mamografia.jpg\"")
    print("     o directamente:")
    print("     python main.py \"ruta/a/la/mamografia.jpg\"\n")
    print("  2. Diagnosticar un estudio completo (carpeta con vistas CC y MLO):")
    print("     python main.py --carpeta \"ruta/a/la/carpeta_paciente\"")
    print("     o directamente:")
    print("     python main.py \"ruta/a/la/carpeta_paciente\"\n")
    print("OTROS COMANDOS DISPONIBLES:")
    print("  python main.py train        -> Reentrenar la red neuronal")
    print("  python main.py preprocess   -> Segmentar fondo negro (ROI) y particionar")
    print("  python main.py visualize    -> Regenerar gráficas científicas a 300 DPI")
    print("=" * 75)

def normalizar_argumentos():
    """Permite ejecutar directamente 'python main.py ruta' o 'python main.py --imagen ruta'"""
    if len(sys.argv) == 1:
        imprimir_guia_rapida()
        sys.exit(0)

    primer_arg = sys.argv[1].strip()

    # Si el usuario pasa directamente --imagen o --carpeta sin escribir predict
    if primer_arg in ("--imagen", "--carpeta", "-i", "-c"):
        sys.argv.insert(1, "predict")
        return

    # Subcomandos conocidos
    subcomandos_conocidos = {"predict", "train", "preprocess", "download", "visualize", "-h", "--help"}
    if primer_arg in subcomandos_conocidos:
        return

    # Si el primer argumento es una ruta existente en disco
    if os.path.exists(primer_arg):
        if os.path.isdir(primer_arg):
            sys.argv = [sys.argv[0], "predict", "--carpeta", primer_arg] + sys.argv[2:]
        else:
            sys.argv = [sys.argv[0], "predict", "--imagen", primer_arg] + sys.argv[2:]
        return

    # Si parece una ruta de archivo de imagen
    if primer_arg.lower().endswith((".jpg", ".jpeg", ".png", ".dcm", ".dicom")):
        sys.argv = [sys.argv[0], "predict", "--imagen", primer_arg] + sys.argv[2:]
        return

def main():
    normalizar_argumentos()

    parser = argparse.ArgumentParser(
        prog="MammoCNN",
        description="Sistema de Triaje y Diagnóstico Asistido por IA en Mamografía"
    )
    subparsers = parser.add_subparsers(dest="comando", help="Comandos disponibles")

    # 1. Subcomando: predict
    p_predict = subparsers.add_parser("predict", help="Diagnosticar una imagen o estudio de paciente")
    p_predict.add_argument("--imagen", type=str, help="Ruta a una imagen individual (.jpg, .png)")
    p_predict.add_argument("--carpeta", type=str, help="Ruta a la carpeta de un paciente con múltiples vistas")
    p_predict.add_argument("--modelo", type=str, default=None, help="Ruta opcional al archivo .pth del modelo")
    p_predict.add_argument("--tamano", type=int, default=512, help="Resolución en píxeles (default: 512)")

    # 2. Subcomando: train
    p_train = subparsers.add_parser("train", help="Entrenar la red neuronal MammoCNN")
    p_train.add_argument("--tamano", type=int, default=512, help="Resolución de las imágenes (512 o 1024)")
    p_train.add_argument("--epochs", type=int, default=10, help="Número de épocas de entrenamiento")
    p_train.add_argument("--batch-size", type=int, default=16, help="Tamaño del batch")
    p_train.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    p_train.add_argument("--arch", type=str, default="mammocnn", choices=["mammocnn", "resnet18"], help="Arquitectura")

    # 3. Subcomando: preprocess
    p_prep = subparsers.add_parser("preprocess", help="Segmentar fondo negro (ROI) y particionar dataset")
    p_prep.add_argument("--tamano", type=int, default=512, help="Resolución objetivo (default: 512)")

    # 4. Subcomando: download
    p_down = subparsers.add_parser("download", help="Descargar mamografías desde PACS institucional (módulo interno/privado)")
    p_down.add_argument("--limit", type=int, default=None, help="Límite de pacientes a descargar")
    p_down.add_argument("--workers", type=int, default=6, help="Número de hilos en paralelo (default: 6)")
    p_down.add_argument("--solo-normales", action="store_true", help="Descargar únicamente casos normales (BI-RADS 1 y 2)")
    p_down.add_argument("--solo-patologicos", action="store_true", help="Descargar únicamente casos patológicos (BI-RADS 4 y 5)")

    # 5. Subcomando: visualize
    p_viz = subparsers.add_parser("visualize", help="Generar gráficas científicas y curvas de convergencia (300 DPI)")

    args = parser.parse_args()

    if args.comando == "predict":
        from src import predict as pred_mod
        sys.argv = [sys.argv[0]]
        if args.imagen: sys.argv.extend(["--imagen", args.imagen])
        if args.carpeta: sys.argv.extend(["--carpeta", args.carpeta])
        if args.modelo: sys.argv.extend(["--modelo", args.modelo])
        sys.argv.extend(["--tamano", str(args.tamano)])
        pred_mod.main()

    elif args.comando == "train":
        from src.models import trainer as train_mod
        sys.argv = [
            sys.argv[0],
            "--tamano", str(args.tamano),
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--lr", str(args.lr),
            "--arch", args.arch
        ]
        train_mod.main()

    elif args.comando == "preprocess":
        from src.data.preprocessor import procesar_dataset
        procesar_dataset(tamano=args.tamano)

    elif args.comando == "download":
        try:
            from src.data import downloader as down_mod
            sys.argv = [sys.argv[0]]
            if args.limit: sys.argv.extend(["--limit", str(args.limit)])
            sys.argv.extend(["--workers", str(args.workers)])
            if args.solo_normales: sys.argv.append("--solo-normales")
            if args.solo_patologicos: sys.argv.append("--solo-patologicos")
            down_mod.main()
        except (ImportError, AttributeError):
            print("\n[AVISO] El módulo de descarga directa e ingesta hospitalaria requiere credenciales")
            print("institucionales y se distribuye en un repositorio privado por razones de seguridad perimetral.")
            print("Para entrenar o inferir en este repositorio público, utilice un dataset existente en DATASET_DIR.")

    elif args.comando == "visualize":
        from src.utils import visualization as viz_mod
        viz_mod.graficar_curvas_entrenamiento()
        viz_mod.graficar_matriz_confusion()
        viz_mod.graficar_distribucion_dataset()
        viz_mod.graficar_distribucion_probabilidades()
        print("\n[OK] Todas las gráficas científicas han sido regeneradas con éxito.")

if __name__ == "__main__":
    main()
