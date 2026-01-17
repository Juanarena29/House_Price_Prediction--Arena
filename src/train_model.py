import joblib
import logging
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

logger = logging.getLogger(__name__)


def create_preprocessing_pipeline(X):
    """
    Arma el pipeline de preprocesado a partir del dataframe.
    Detecta numéricas y categóricas y prepara todo para transformar.
    """
    numeric_features = X.select_dtypes(
        include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(
        include=["object", "category", "bool"]).columns.tolist()

    transformers = []
    if numeric_features:
        numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
        transformers.append(("num", numeric_transformer, numeric_features))
    if categorical_features:
        categorical_transformer = Pipeline(
            steps=[("encoder", OneHotEncoder(handle_unknown="ignore"))]
        )
        transformers.append(
            ("cat", categorical_transformer, categorical_features))

    if not transformers:
        raise ValueError(
            "No se encontraron columnas numéricas ni categóricas para preprocesar")

    preprocessor = ColumnTransformer(transformers=transformers)
    return Pipeline(steps=[("preprocessor", preprocessor)])


def _get_base_model(model_type):
    """
    Devuelve el modelo base (sin entrenar)
    Si el tipo no existe, avisa con error.
    """
    base_models = {
        "random_forest": RandomForestRegressor(),
        "decision_tree": DecisionTreeRegressor(),
        "gradient_boosting": GradientBoostingRegressor(),
        "linear": LinearRegression()
    }

    if model_type not in base_models:
        raise ValueError(
            f"Modelo no soportado. Usa: {list(base_models.keys())}")

    return base_models[model_type]


def train_multiple_models(X_train_prepared, y_train, models_config, cv=5, scoring='r2'):
    """
    Entrena varios modelos con GridSearchCV.

    Recibe los datos ya preprocesados y una lista de configuraciones.
    Te devuelve cada modelo entrenado con info extra en grid_search_info_.
    """
    trained_models = {}

    logger.info("%s", "\n" + "=" * 80)
    logger.info(" ENTRENAMIENTO DE MODELOS CON GRIDSEARCHCV")
    logger.info("="*80)

    for config in models_config:
        model_name = config.get('name', config['type'])
        model_type = config['type']

        # Validar que tenga param_grid
        if 'param_grid' not in config:
            raise ValueError(
                f"Configuración de '{model_name}' debe incluir 'param_grid'. "
                f"Esta función SIEMPRE usa GridSearchCV."
            )

        param_grid = config['param_grid']
        model_cv = config.get('cv', cv)
        model_scoring = config.get('scoring', scoring)

        # Obtener modelo base
        base_model = _get_base_model(model_type)

        logger.info("%s", "\n" + "=" * 80)
        logger.info("   Entrenando: %s", model_name)
        logger.info("   Tipo: %s", model_type)
        logger.info("   CV folds: %d", model_cv)
        logger.info("   Grilla de parámetros: %s", param_grid)

        # GridSearchCV
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=model_cv,
            scoring=model_scoring,
            n_jobs=-1,
            verbose=1,
            return_train_score=True
        )

        grid_search.fit(X_train_prepared, y_train)

        # Guardar modelo entrenado
        trained_models[model_name] = grid_search.best_estimator_

        # Agregar información de GridSearchCV al modelo
        trained_models[model_name].grid_search_info_ = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_,
            'cv_folds': model_cv
        }

        logger.info("   ✓ Completado")
        logger.info("   Mejores params: %s", grid_search.best_params_)
        logger.info("   Mejor score (CV): %.4f", grid_search.best_score_)

    return trained_models


def save_model(model, filepath):
    """
    Guarda un modelo (o pipeline) en disco usando joblib.
    """
    joblib.dump(model, filepath)
    print(f" Modelo guardado: {filepath}")


def load_model(filepath):
    """
    Carga un modelo o pipeline guardado en disco.
    """
    return joblib.load(filepath)
