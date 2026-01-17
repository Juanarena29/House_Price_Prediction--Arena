import os
from sklearn.model_selection import train_test_split
from src.data_preprocessing import load_and_clean_data
from src.evaluate import (
    evaluate_multiple_models,
    plot_predictions,
    plot_multiple_predictions,
    plot_residuals,
    compare_models_plots,
    plot_feature_importance,
)
from src.train_model import create_preprocessing_pipeline, save_model, train_multiple_models
from src.utils import ensure_directory


def _split_features_target(df, target_col="SalePrice"):
    """Separa el dataframe en features (X) y target (y)."""
    if target_col not in df.columns:
        raise ValueError(
            f"La columna objetivo '{target_col}' no está en el dataframe")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def _default_models_config():
    """Configuración de modelos con param_grid para GridSearchCV."""
    return [
        {
            "name": "RandomForest",
            "type": "random_forest",
            "param_grid": {
                "n_estimators": [100, 200, 300],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5]
            }
        },
        {
            "name": "GradientBoosting",
            "type": "gradient_boosting",
            "param_grid": {
                "n_estimators": [100, 200, 300],
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [3, 5, 7]
            }
        },
        {
            "name": "DecisionTree",
            "type": "decision_tree",
            "param_grid": {
                "max_depth": [5, 10, 15, None],
                "min_samples_split": [2, 5, 10]
            }
        }
    ]


def run_pipeline(
    data_path: str = os.path.join("data", "house_prices_records.csv"),
    models_dir: str = "models",
    show_plots: bool = True,
):
    """Ejecuta el flujo completo: cargar, limpiar, entrenar, evaluar y guardar modelos.

    Args:
        data_path: Ruta del archivo CSV con datos
        models_dir: Directorio donde guardar modelos
        show_plots: Si True, muestra gráficos de evaluación
    """

    ensure_directory(models_dir)

    # 1) Cargar y limpiar datos
    df = load_and_clean_data(data_path)
    X, y = _split_features_target(df)

    # 2) Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3) Pipeline de preprocesamiento (solo transformación)
    preprocessing_pipeline = create_preprocessing_pipeline(X_train)
    X_train_prepared = preprocessing_pipeline.fit_transform(X_train)
    X_test_prepared = preprocessing_pipeline.transform(X_test)

    # 4) Entrenar múltiples modelos con GridSearchCV
    models_config = _default_models_config()

    print("\n" + "="*60)
    print(" Optimizando hiperparámetros con GridSearchCV...")
    print("="*60)

    trained_models = train_multiple_models(
        X_train_prepared, y_train, models_config
    )

    # Mostrar mejores parámetros encontrados
    print("\n" + "="*60)
    print(" MEJORES HIPERPARÁMETROS ENCONTRADOS")
    print("="*60)
    for model_name, model in trained_models.items():
        if hasattr(model, 'grid_search_info_'):
            info = model.grid_search_info_
            print(f"\n{model_name}:")
            print(f"  Mejor score CV: {info['best_score']:.4f}")
            print(f"  Mejores params: {info['best_params']}")

    # 5) Evaluar y seleccionar mejor modelo
    results_df = evaluate_multiple_models(
        trained_models, X_train_prepared, y_train, X_test_prepared, y_test
    )
    best_model_name = results_df.iloc[0]["Modelo"]
    best_model = trained_models[best_model_name]

    # 5.1) Gráficos opcionales
    if show_plots:
        # Predicciones por modelo (para comparativa)
        predictions_dict = {
            name: model.predict(X_test_prepared)
            for name, model in trained_models.items()
        }

        # Gráficos del mejor modelo
        best_y_pred = best_model.predict(X_test_prepared)
        plot_predictions(y_test, best_y_pred,
                         model_name=best_model_name, dataset_type="Test")
        plot_residuals(y_test, best_y_pred,
                       model_name=best_model_name, dataset_type="Test")

        # Gráficos comparativos
        plot_multiple_predictions(
            y_test, predictions_dict, dataset_type="Test")
        compare_models_plots(results_df, dataset_type="Test")

        # Feature importance (solo si aplica)
        try:
            feature_names = (
                preprocessing_pipeline
                .named_steps["preprocessor"]
                .get_feature_names_out()
                .tolist()
            )
            plot_feature_importance(
                best_model, feature_names, model_name=best_model_name)
        except (AttributeError, KeyError) as e:
            print(f"⚠ No se pudo generar gráfico de feature importance: {e}")

    # 6) Guardar artefactos
    pipeline_path = os.path.join(models_dir, "pipeline.pkl")
    best_model_path = os.path.join(models_dir, "best_model.pkl")
    save_model(preprocessing_pipeline, pipeline_path)
    save_model(best_model, best_model_path)

    print(f"\n✓ Pipeline guardado en {pipeline_path}")
    print(f"✓ Mejor modelo ({best_model_name}) guardado en {best_model_path}")

    return {
        "results": results_df,
        "best_model_name": best_model_name,
        "pipeline_path": pipeline_path,
        "model_path": best_model_path,
    }


if __name__ == "__main__":
    run_pipeline()
