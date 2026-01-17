"""
Configuración de pytest para que pueda importar módulos desde src/
"""
import sys
import os

# Agregar el directorio src/ al PYTHONPATH
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'src')))
