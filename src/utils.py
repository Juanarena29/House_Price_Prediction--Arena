import logging
import os


def setup_logging(level=logging.INFO):
    """Configura logging básico si aún no está configurado."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )


def ensure_directory(path: str) -> str:
    """Crea el directorio si no existe y devuelve la ruta."""
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path
