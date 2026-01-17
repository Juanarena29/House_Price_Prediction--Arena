"""
Tests para data_preprocessing.py

Lo básico que necesito cubrir:
- _clean_missing_values()
- load_and_clean_data()
- prepare_new_data()
"""

from src.data_preprocessing import _clean_missing_values, load_and_clean_data, prepare_new_data  # pylint: disable=import-error,wrong-import-position
import os
import sys
import tempfile

import pandas as pd
import numpy as np

# Agregar src/ al path para poder importar
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'src')))


def test_clean_missing_values_fills_areas_with_zero():
    """
    Test: Las columnas de área con NaN se vuelven 0.

    En simple: si una parte no existe, el área es 0.
    No tiene sentido usar promedio acá.
    """
    # ARRANGE (Preparar datos de entrada)
    df_input = pd.DataFrame({
        'GarageArea': [500.0, np.nan, 300.0],
        '2ndFlrSF': [np.nan, 250.0, 300.0],
        'MasVnrArea': [np.nan, np.nan, 100.0],
        'OpenPorchSF': [np.nan, 300.0, 300.0],
        'WoodDeckSF': [np.nan, np.nan, np.nan],
        'EnclosedPorch': [100.0, 100.0, 300.0],
        'TotalBsmtSF': [np.nan, 400.0, np.nan],
        'BsmtFinSF1': [np.nan, 500.0, 600.0],
    })

    # ACT (Ejecutar la función)
    df_result = _clean_missing_values(df_input.copy())

    # ASSERT (Verificar resultados)
    # Lista de columnas de área que deben ser tratadas
    area_columns = ['GarageArea', '2ndFlrSF', 'MasVnrArea', 'OpenPorchSF',
                    'WoodDeckSF', 'EnclosedPorch', 'TotalBsmtSF', 'BsmtFinSF1']

    # Verificación 1: No quedan NaN en NINGUNA columna de área
    for col in area_columns:
        if col in df_result.columns:
            assert df_result[col].isna().sum() == 0, \
                f"{col} todavía tiene valores NaN después de limpiar"

    # Verificación 2: Los NaN fueron reemplazados por 0 (verificar casos específicos)
    assert df_result['GarageArea'].iloc[1] == 0, \
        f"GarageArea: Esperado 0, pero obtuvo {df_result['GarageArea'].iloc[1]}"
    assert df_result['2ndFlrSF'].iloc[0] == 0, \
        f"2ndFlrSF: Esperado 0, pero obtuvo {df_result['2ndFlrSF'].iloc[0]}"
    assert df_result['WoodDeckSF'].iloc[0] == 0, \
        f"WoodDeckSF: Esperado 0 (100% NaN), pero obtuvo {df_result['WoodDeckSF'].iloc[0]}"

    # Verificación 3: Los valores originales (no NaN) no cambiaron
    assert df_result['GarageArea'].iloc[0] == 500.0, \
        "GarageArea[0] cambió inesperadamente"
    assert df_result['2ndFlrSF'].iloc[1] == 250.0, \
        "2ndFlrSF[1] cambió inesperadamente"

    # Verificación 4: Columnas NO tratadas (LotArea) no fueron modificadas
    assert df_result["EnclosedPorch"].equals(df_input["EnclosedPorch"]), \
        "LotArea fue modificado cuando no debería (es una columna de control)"


def test_clean_missing_values_fills_years_with_median():
    """
    Test: Años con NaN se completan con la mediana.

    La mediana es más estable que la media cuando hay outliers.
    """
    df_input = pd.DataFrame({
        "YearBuilt": [1933, np.nan, 1992],
        "YearRemodAdd": [np.nan, 1970, 1977],
        "GarageYrBlt": [np.nan, np.nan, np.nan]
    })

    median_YearBuilt = pd.Series([1933, 1992]).median()
    median_yearRemodAdd = pd.Series([1970, 1977]).median()

    df_result = _clean_missing_values(df_input.copy())

    year_columns = ["YearBuilt", "YearRemodAdd", "GarageYrBlt"]

    # Verificacion general nans
    for col in year_columns:
        if col in df_result.columns:
            assert df_result[col].isna().sum() == 0, f"{col} tiene NaN"

    # verifico casos especificos (todo nan, nan principio, nan medio)
    assert df_result["YearBuilt"].iloc[
        1] == median_YearBuilt, f"{df_result["YearBuilt"]} no da resultado esperado"
    assert df_result["YearRemodAdd"].iloc[
        0] == median_yearRemodAdd, f"{df_result["YearRemodAdd"]} no da resultado esperado"
    assert df_result["GarageYrBlt"].iloc[1] == 0, f"{df_result["GarageYrBlt"]} no da resultado esperado"
    # como GarageYrBlt tiene todos nan, en preprocessing lleno con 0 entonces su media es 0. (evito media nan)

    # verifico que no se cambien valores no nans
    assert df_result["YearBuilt"].iloc[0] == 1933, f"{df_result["YearBuilt"]} cambio uno de sus valores originales"
    assert df_result["YearRemodAdd"].iloc[
        1] == 1970, f"{df_result["YearRemodAdd"]} cambio uno de sus valores originales"


def test_clean_missing_values_fills_continuous_with_mean():
    """
    Test: Continuas con NaN se completan con la media.

    Si la columna está 100% NaN, cae a 0.
    """
    df_input = pd.DataFrame({
        "LotFrontage": [200, np.nan, 300],
        "1stFlrSF": [np.nan, 200, 200],
        "GrLivArea": [np.nan, np.nan, np.nan],
        "LotArea": [200, 100, 300]
    })

    mean_LotFrontage = pd.Series([200, 300]).mean()
    mean_1stFlrSF = pd.Series([200, 200]).mean()
    mean_GrLivArea = 0
    mean_LotArea = pd.Series([200, 100, 300]).mean()

    df_result = _clean_missing_values(df_input.copy())

    cont_columns = ["LotFrontage", "1stFlrSF", "GrLivArea", "LotArea"]
    # verifico general nans
    for col in cont_columns:
        if col in df_result.columns:
            assert df_result[col].isna().sum() == 0, f"{col} tiene nans"

    # verifico casos especiales? x las dudas
    assert df_result["LotFrontage"].iloc[
        1] == mean_LotFrontage, f"{df_result["LotFrontage"]} no relleno bien el nan"
    assert df_result["1stFlrSF"].iloc[
        0] == mean_1stFlrSF, f"{df_result["1stFlrSF"]} no relleno bien el nan"
    assert df_result["GrLivArea"].iloc[
        0] == mean_GrLivArea, f"{df_result["1stFlrSF"]} no relleno bien el nan"
    assert df_result["LotArea"].iloc[
        0] == mean_LotArea, f"{df_result["LotArea"]} falló columna sin nans"

    # verifico que no se cambien valores no nans
    assert df_result["LotFrontage"].iloc[
        2] == 300, f"{df_result["LotFrontage"]} cambio un valor de nan original"
    assert df_result["LotArea"].iloc[1] == 100, f"{df_result["LotFrontage"]} cambio un valor de nan original"

    # verifico que no se modifiquen columnas que no fueron procesadas
    assert df_result["LotArea"].equals(
        df_input["LotArea"]), f"{df_result["LotArea"]} se modifico columna no procesada"


def test_clean_missing_values_fills_categoricals_with_values():
    """
    Test: Categóricas con NaN se llenan con valores de negocio.

    No hay promedios acá, son valores con sentido (ej: "No", "None").
    """
    df_input = pd.DataFrame({
        'BsmtExposure': ["Gd", np.nan, "Av"],
        'BsmtFinType1': [np.nan, "GLQ", "Rec"],
        'GarageFinish': [np.nan, np.nan, np.nan],
        'KitchenQual': ["Rec", "ALQ", np.nan]
    })

    value_BsmtExposure = "No"
    value_BsmtFinType1 = 'None'
    value_GarageFinish = 'None'
    value_KitchenQual = "TA"
    df_result = _clean_missing_values(df_input.copy())

    categorical_columns = ["BsmtExposure",
                           "BsmtFinType1", "GarageFinish", "KitchenQual"]

    # verificacion general nans
    for col in categorical_columns:
        if col in df_result.columns:
            assert df_result[col].isna().sum() == 0

    # verificacion casos especiales x las dudas
    assert df_result["BsmtExposure"].iloc[1] == value_BsmtExposure, "BsmtExposure llena mal el nan"
    assert df_result["BsmtFinType1"].iloc[0] == value_BsmtFinType1, "BsmtFinType1 llena mal el nan"
    assert df_result["GarageFinish"].iloc[1] == value_GarageFinish, "GarageFinish llena mal el nan"
    assert df_result["KitchenQual"].iloc[2] == value_KitchenQual, "KitchenQual llena mal el nan"

    # verifico que no se cambien valores no nans
    assert df_result["BsmtExposure"].iloc[0] == "Gd", "Se ha cambiado un valor no nulo"
    assert df_result["BsmtFinType1"].iloc[1] == "GLQ", "Se ha cambiado un valor no nulo"
    assert df_result["KitchenQual"].iloc[0] == "Rec", "Se ha cambiado un valor no nulo"

    # verifico que columna 100% NaN se llenó correctamente
    assert df_result["GarageFinish"].iloc[
        2] == value_GarageFinish, "GarageFinish no se llenó correctamente (100% NaN)"


def test_load_and_clean_data_raises_error_if_file_not_exists():
    """
    Test: Si el archivo no existe, debe lanzar FileNotFoundError.
    """
    # Creo un path q no existe
    non_existent_path = "path/that/does/not/exist.csv"
    # aca testeo el cargarlo (deberia tirar filenotfounder)
    try:
        load_and_clean_data(non_existent_path)
        assert False, "Debería haber lanzado FileNotFoundError pero no lo hizo"

    except FileNotFoundError as error:
        assert non_existent_path in str(error), \
            f"El error no menciona el archivo faltante: {error}"


def test_load_and_clean_data_validates_required_columns():
    """
    Test: Si faltan columnas requeridas, debe lanzar ValueError.
    """
    # Creo df sin SalePrice que es requerida
    df_incomplete = pd.DataFrame({
        'LotArea': [1000, 2000, 3000],
        'YearBuilt': [1990, 2000, 2010],
        'GarageArea': [200, 300, 400]
    })

    # Guardo en archivo temporal
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
        df_incomplete.to_csv(tmp_file.name, index=False)
        tmp_path = tmp_file.name

    try:
        # Intento cargar csv sin la columna SalePrice, si falla metodo, pasa test.
        try:
            load_and_clean_data(tmp_path, target_col="SalePrice")
            assert False, "Debería haber lanzado ValueError por columna faltante"

        except ValueError as error:
            # ASSERT: verificar que menciona la columna faltante
            assert "SalePrice" in str(error), \
                f"El error no menciona 'SalePrice': {error}"
            # Si llegamos aquí, el test PASA ✓

    finally:
        # Limpio ese archivo temporal
        os.unlink(tmp_path)


def test_prepare_new_data_removes_target_if_exists():
    """
    Test: Si aparece SalePrice en predicción, lo borra.

    Es por seguridad: no quiero que se use como feature.
    """
    # ARRANGE: crear CSV con SalePrice (caso de error del usuario)
    df_with_target = pd.DataFrame({
        'SalePrice': [200000, 180000, 250000],
        'LotArea': [1000, 2000, 3000],
        'YearBuilt': [1990, 2000, 2010]
    })

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
        df_with_target.to_csv(tmp_file.name, index=False)
        tmp_path = tmp_file.name

    try:
        # ACT
        df_result = prepare_new_data(tmp_path)

        # ASSERT: 'SalePrice' debe haber sido eliminado
        assert 'SalePrice' not in df_result.columns, \
            "'SalePrice' no fue eliminado del DataFrame de predicción"
        assert 'LotArea' in df_result.columns, \
            "LotArea fue eliminado incorrectamente"
        assert len(df_result.columns) == 2, \
            f"Número incorrecto de columnas: esperado 2, obtenido {len(df_result.columns)}"

    finally:
        os.unlink(tmp_path)


def test_prepare_new_data_works_without_target_column():
    """
    Test: Funciona bien cuando el CSV no trae SalePrice (caso normal).
    """
    # ARRANGE: CSV sin SalePrice (caso normal de predicción)
    df_prediction = pd.DataFrame({
        'LotArea': [1000, 2000, 3000],
        'YearBuilt': [1990, 2000, 2010],
        'GarageArea': [200, 300, 400]
    })

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
        df_prediction.to_csv(tmp_file.name, index=False)
        tmp_path = tmp_file.name

    try:
        # ACT
        df_result = prepare_new_data(tmp_path)

        # ASSERT
        assert df_result is not None, "prepare_new_data devolvió None"
        assert len(df_result) == 3, "Número incorrecto de filas"
        assert 'SalePrice' not in df_result.columns, \
            "'SalePrice' no debería estar presente"
        assert list(df_result.columns) == ['LotArea', 'YearBuilt', 'GarageArea'], \
            "Columnas incorrectas en resultado"

    finally:
        os.unlink(tmp_path)
