"""
Tests para el módulo train_model.py

Cubre lo necesario:
- create_preprocessing_pipeline()
- train_multiple_models()
"""

from src.train_model import create_preprocessing_pipeline, train_multiple_models  # pylint: disable=import-error,wrong-import-position
import os
import sys

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

# Agregar src/ al path para poder importar
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'src')))


# =============================================================================
# TESTS PARA create_preprocessing_pipeline()
# =============================================================================


def test_create_preprocessing_pipeline_builds_pipeline_for_numeric_and_categorical():
    """
    Test: Verifica que se construya un Pipeline con preprocesador numérico y categórico.

    Razón:
    El pipeline debe detectar tipos y preparar transformaciones correctas.
    """
    # ARRANGE
    df_input = pd.DataFrame({
        "LotArea": [1000, 2000, 3000],
        "Neighborhood": ["A", "B", "A"],
        "HasGarage": [True, False, True]
    })

    # ACT
    pipeline = create_preprocessing_pipeline(df_input)

    # ASSERT
    assert isinstance(pipeline, Pipeline), "No devolvió un Pipeline"
    assert "preprocessor" in pipeline.named_steps, "Falta el paso 'preprocessor'"

    transformers = pipeline.named_steps["preprocessor"].transformers
    transformer_names = [t[0] for t in transformers]
    assert "num" in transformer_names, "No se detectaron columnas numéricas"
    assert "cat" in transformer_names, "No se detectaron columnas categóricas"


def test_create_preprocessing_pipeline_raises_when_no_columns():
    """
    Test: Si no hay columnas, debe lanzar ValueError.

    Razón:
    Sin columnas no hay nada que transformar.
    """
    # ARRANGE
    df_empty = pd.DataFrame()

    # ACT / ASSERT
    try:
        create_preprocessing_pipeline(df_empty)
        assert False, "Debería haber lanzado ValueError pero no lo hizo"
    except ValueError as error:
        assert "No se encontraron" in str(error)


# =============================================================================
# TESTS PARA train_multiple_models()
# =============================================================================


def test_train_multiple_models_returns_models_with_grid_info():
    """
    Test: Verifica que entrene múltiples modelos y adjunte info de GridSearchCV.

    Razón:
    La función principal debe devolver modelos entrenados con metadatos de CV.
    """
    # ARRANGE: datos mínimos y rápidos
    rng = np.random.default_rng(123)
    X_train_prepared = rng.normal(size=(30, 2))
    y_train = 2 * X_train_prepared[:, 0] + 0.5 * X_train_prepared[:, 1]

    models_config = [
        {
            "name": "RandomForest",
            "type": "random_forest",
                    "param_grid": {"n_estimators": [5], "max_depth": [None], "min_samples_split": [2]},
                    "cv": 2
        },
        {
            "name": "GradientBoosting",
            "type": "gradient_boosting",
                    "param_grid": {"n_estimators": [10], "learning_rate": [0.1], "max_depth": [3]},
                    "cv": 2
        }
    ]

    # ACT
    trained_models = train_multiple_models(
        X_train_prepared, y_train, models_config, cv=2)

    # ASSERT
    assert "RandomForest" in trained_models, "No devolvió RandomForest entrenado"
    assert "GradientBoosting" in trained_models, "No devolvió GradientBoosting entrenado"

    for model_name, model in trained_models.items():
        assert hasattr(
            model, "grid_search_info_"), f"{model_name} sin grid_search_info_"
        assert "best_params" in model.grid_search_info_, f"{model_name} sin best_params"
        assert "best_score" in model.grid_search_info_, f"{model_name} sin best_score"


def test_train_multiple_models_raises_when_param_grid_missing():
    """
    Test: Si falta param_grid en una config, debe lanzar ValueError.

    Razón:
    Esta función SIEMPRE usa GridSearchCV, por lo tanto param_grid es obligatorio.
    """
    # ARRANGE
    X_train_prepared = np.array([[1, 2], [3, 4], [5, 6]])
    y_train = np.array([1, 2, 3])

    models_config = [
        {
            "name": "RandomForest",
            "type": "random_forest"
        }
    ]

    # ACT / ASSERT
    try:
        train_multiple_models(X_train_prepared, y_train, models_config, cv=2)
        assert False, "Debería haber lanzado ValueError por falta de param_grid"
    except ValueError as error:
        assert "param_grid" in str(error), f"Error inesperado: {error}"
