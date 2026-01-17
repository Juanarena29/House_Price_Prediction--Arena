"""
Tests para el módulo evaluate.py

Lo justo y necesario:
- evaluate_model()
- evaluate_multiple_models()
"""

from src.evaluate import evaluate_model, evaluate_multiple_models  # pylint: disable=import-error,wrong-import-position
import os
import sys

import numpy as np
import pandas as pd

# Agregar src/ al path para poder importar
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'src')))


class DummyModel:
    """Modelo simple para tests: siempre devuelve el mismo valor."""

    def __init__(self, y_pred_value):
        self.y_pred_value = y_pred_value

    def predict(self, X):
        return np.full(shape=(len(X),), fill_value=self.y_pred_value, dtype=float)


# =============================================================================
# TESTS PARA evaluate_model()
# =============================================================================


def test_evaluate_model_returns_expected_structure_and_metrics():
    """
    Test: Estructura y métricas básicas de evaluate_model().

    En criollo: quiero el dict completo y que con predicción perfecta
    las métricas den 0 de error y R²=1.
    """
    # ARRANGE
    X_train = np.zeros((3, 2))
    X_test = np.zeros((2, 2))
    y_train = np.array([10, 10, 10], dtype=float)
    y_test = np.array([10, 10], dtype=float)

    model = DummyModel(y_pred_value=10)

    # ACT
    result = evaluate_model(model, X_train, y_train,
                            X_test, y_test, model_name="Dummy")

    # ASSERT estructura
    assert "model_name" in result, "Falta model_name"
    assert "train" in result, "Falta train"
    assert "test" in result, "Falta test"
    assert "overfitting" in result, "Falta overfitting"
    assert "r2_difference" in result, "Falta r2_difference"

    # ASSERT métricas (predicción perfecta)
    assert result["train"]["MAE"] == 0, "MAE train debería ser 0"
    assert result["test"]["MAE"] == 0, "MAE test debería ser 0"
    assert result["train"]["RMSE"] == 0, "RMSE train debería ser 0"
    assert result["test"]["RMSE"] == 0, "RMSE test debería ser 0"
    assert result["train"]["R2"] == 1.0, "R2 train debería ser 1"
    assert result["test"]["R2"] == 1.0, "R2 test debería ser 1"


# =============================================================================
# TESTS PARA evaluate_multiple_models()
# =============================================================================


def test_evaluate_multiple_models_returns_sorted_dataframe():
    """
    Test: Evalúa varios modelos y ordena por Test_R2.

    Si está bien ordenado, el mejor debería quedar primero.
    """
    # ARRANGE
    X_train = np.zeros((3, 2))
    X_test = np.zeros((2, 2))
    y_train = np.array([10, 10, 10], dtype=float)
    y_test = np.array([10, 10], dtype=float)

    models = {
        "Perfect": DummyModel(10),
        "Bad": DummyModel(0)
    }

    # ACT
    results_df = evaluate_multiple_models(
        models, X_train, y_train, X_test, y_test)

    # ASSERT
    assert isinstance(results_df, pd.DataFrame), "No devolvió DataFrame"
    assert list(results_df.columns) == [
        "Modelo", "Train_MAE", "Train_R2", "Train_RMSE",
        "Test_MAE", "Test_R2", "Test_RMSE", "Overfitting", "R2_Diff"
    ], "Columnas inesperadas"

    # El mejor modelo (Perfect) debe estar primero
    assert results_df.iloc[0]["Modelo"] == "Perfect", "No ordenó por Test_R2"
