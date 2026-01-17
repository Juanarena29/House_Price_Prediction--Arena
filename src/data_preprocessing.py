import os
import pandas as pd


def _clean_missing_values(df):
    """
    Limpia NaN con las reglas del negocio.

    Lo uso igual para entrenamiento y para predicción.
    Devuelve el mismo DF pero con los faltantes resueltos.
    """
    # Para áreas/superficies → 0 tiene sentido (no existe esa área)
    area_columns = ['2ndFlrSF', 'GarageArea', 'MasVnrArea', 'BsmtFinSF1', 'TotalBsmtSF',
                    'OpenPorchSF', 'EnclosedPorch', 'WoodDeckSF', 'BedroomAbvGr', 'BsmtUnfSF']
    for col in area_columns:
        if col in df.columns and df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(0)

    # Para años → uso la mediana, si toda la columna es nan, lleno con 0.
    year_columns = ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']
    for col in year_columns:
        if col in df.columns:
            if df[col].isnull().all():
                df[col] = df[col].fillna(0)
            else:
                df[col] = df[col].fillna(df[col].median())

    # Para medidas continuas → uso mean()
    continuous_columns = ['LotFrontage', '1stFlrSF', 'GrLivArea', 'LotArea']
    for col in continuous_columns:
        if col in df.columns:
            if df[col].isnull().all():
                df[col] = df[col].fillna(0)
            else:
                df[col] = df[col].fillna(df[col].mean())

    # Manejo personalizado de categóricas
    categorical_fills = {
        'BsmtExposure': 'No',      # No hay exposición = No hay sótano
        'BsmtFinType1': 'None',     # No tiene
        'GarageFinish': 'None',     # No tiene garage
        'KitchenQual': 'TA'  # kitchen qual avg
    }
    for col, fill_value in categorical_fills.items():
        if col in df.columns and df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(fill_value)

    return df


def load_and_clean_data(path_csv, target_col="SalePrice", required_cols=None):
    """
    Carga el CSV y lo deja listo para usar.

    Si hay target, valida que exista. Si no hay, no molesta.
    Después aplica la limpieza de NaN.
    """
    if not os.path.exists(path_csv):
        raise FileNotFoundError(f"Archivo no encontrado: {path_csv}")

    df = pd.read_csv(path_csv)

    # Solo validar columnas si se especifican o si hay target_col
    if required_cols:
        expected_cols = required_cols
    elif target_col:
        expected_cols = [target_col, "LotArea", "YearBuilt"]
    else:
        # Predicción sin target: solo validar que tenga columnas
        expected_cols = []

    if expected_cols:
        missing_cols = set(expected_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Faltan columnas: {missing_cols}")

    df = _clean_missing_values(df)
    return df


def prepare_new_data(path_csv):
    """
    Prepara datos nuevos para predicción (sin target).

    Carga, limpia y si aparece SalePrice la borra por seguridad.
    """
    df = load_and_clean_data(path_csv, target_col=None, required_cols=None)

    # Eliminar 'SalePrice' si existe (por seguridad)
    if "SalePrice" in df.columns:
        df = df.drop(columns=["SalePrice"])

    return df
