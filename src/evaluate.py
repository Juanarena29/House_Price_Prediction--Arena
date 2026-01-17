import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error


def evaluate_model(model, X_train_prepared,
                   y_train, X_test_prepared,
                   y_test, model_name="modelo"):
    """
    Evalúa un solo modelo y te devuelve métricas de train y test.
    compara cómo rinde con datos que vio vs datos que no vio,
    y avisa si hay overfitting.
    """
    # pred con datos que vió
    y_train_pred = model.predict(X_train_prepared)
    # pred con datos que no vió
    y_test_pred = model.predict(X_test_prepared)
    # Calcular metricas
    # mae
    mae_train = mean_absolute_error(y_train, y_train_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    # R2
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)
    # rmse
    rmse_train = root_mean_squared_error(y_train, y_train_pred)
    rmse_test = root_mean_squared_error(y_test, y_test_pred)
    print(f"\n{'='*60}")
    print(f"EVALUATION: {model_name}")
    print(f"{'='*60}")
    print(
        f"TRAIN → MAE: {mae_train:,.2f} | RMSE: {rmse_train:,.2f} | R²: {r2_train:.4f}")
    print(
        f"TEST  → MAE: {mae_test:,.2f} | RMSE: {rmse_test:,.2f} | R²: {r2_test:.4f}")
    r2_difference = r2_train - r2_test
    print(f"\nR² Difference: {r2_difference:.4f}")
    if r2_difference > 0.1:
        print("X OVERFITTING detected")
        overfitting = True
    else:
        print("The model generalizes well")
        overfitting = False
    data_evaluated = {
        "model_name": model_name,
        "train": {
            "MAE": mae_train,
            "R2": r2_train,
            "RMSE": rmse_train,
        },
        "test": {
            "MAE": mae_test,
            "R2": r2_test,
            "RMSE": rmse_test,
        },
        "overfitting": overfitting,
        "r2_difference": r2_difference
    }

    return data_evaluated


def evaluate_multiple_models(models, X_train_prepared, y_train, X_test_prepared, y_test):
    """
    Evalúa varios modelos, arma una tabla y te muestra cuál ganó.

    Devuelve un DataFrame ordenado por Test R^2.
    """
    results = []
    # itero sobre cada modelo y guardo en var result
    for model_name, model in models.items():
        result = evaluate_model(model, X_train_prepared, y_train,
                                X_test_prepared, y_test, model_name)
        # evaluate_model me manda diccionarios anidados, para guardarlos
        # en un dataframe necesito algo plano. Se aplanan los datos asi:
        flat_result = {
            'Modelo': result['model_name'],
            'Train_MAE': result['train']['MAE'],
            'Train_R2': result['train']['R2'],
            'Train_RMSE': result['train']['RMSE'],
            'Test_MAE': result['test']['MAE'],
            'Test_R2': result['test']['R2'],
            'Test_RMSE': result['test']['RMSE'],
            'Overfitting': result['overfitting'],
            'R2_Diff': result['r2_difference']
        }
        results.append(flat_result)
    # creo el dataframe
    results_df = pd.DataFrame(results)
    # ordeno por metrica r2 test
    results_df = results_df.sort_values("Test_R2", ascending=False)
    print(f"\n{'='*80}")
    print("MODEL COMPARISON")
    print(f"{'='*80}")
    print(results_df.to_string(index=False))
    # guardo el mejor modelo, posicion 0 del df ordenado
    best_model = results_df.iloc[0]  # Primera fila (ya ordenado por Test_R2)

    print(f"\n BEST MODEL: {best_model['Modelo']}")
    print(f"   Test R²: {best_model['Test_R2']:.4f}")
    print(f"   Test MAE: ${best_model['Test_MAE']:,.2f}")
    return results_df


def plot_predictions(y_true, y_pred, model_name="Modelo", dataset_type="Test"):
    """
    Grafica predicciones vs reales para UN modelo.

    Útil para ver si el modelo sigue la diagonal o se dispersa.
    """
    plt.figure(figsize=(10, 6))

    # Puntos: predicciones vs reales
    plt.scatter(y_true, y_pred, alpha=0.6, s=30,
                color='blue', label=model_name)

    # Línea diagonal perfecta (predicción = realidad)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--',
             linewidth=2, label='Perfect prediction')

    # Etiquetas y formato
    plt.xlabel('Actual Values', fontsize=12)
    plt.ylabel('Predictions', fontsize=12)
    plt.title(
        f'Predictions vs Actuals - {model_name} ({dataset_type})', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_multiple_predictions(y_true, predictions_dict, dataset_type="Test"):
    """
    Grafica varios modelos juntos en subplots (grid).

    Sirve para comparar visualmente en un solo vistazo.
    """
    n_models = len(predictions_dict)
    # Calcular filas y columnas para el grid
    n_cols = 3
    n_rows = 1

    _, axes = plt.subplots(n_rows, n_cols, figsize=(14, 6*n_rows))
    axes = axes.flatten()  # Convertir a array 1D para iterar fácil

    for idx, (model_name, y_pred) in enumerate(predictions_dict.items()):
        ax = axes[idx]
        ax.scatter(y_true, y_pred, alpha=0.6, s=30, color='blue')

        # Línea diagonal
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)

        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predictions')
        ax.set_title(f'{model_name} ({dataset_type})')
        ax.grid(True, alpha=0.3)

    # Ocultar subplots vacíos si hay número impar de modelos
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.show()


def plot_residuals(y_true, y_pred, model_name="Modelo", dataset_type="Test"):
    """
    Grafica residuos (error = real - predicción).

    Si ves patrones raros, hay sesgo o falta de features.
    """
    # Calcular residuos (error = real - predicción)
    residuos = y_true - y_pred

    plt.figure(figsize=(10, 6))

    # Puntos: predicciones vs residuos
    plt.scatter(y_pred, residuos, alpha=0.6, s=30,
                color='blue', label=model_name)

    # Línea horizontal en 0 (error cero = predicción perfecta)
    plt.axhline(y=0, color='red', linestyle='--',
                linewidth=2, label='Zero error')

    # Etiquetas y formato
    plt.xlabel('Predictions', fontsize=12)
    plt.ylabel('Residuals (Actual - Prediction)', fontsize=12)
    plt.title(
        f'Residual Analysis - {model_name} ({dataset_type})', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def compare_models_plots(results_df, dataset_type="Test"):
    """
    Barras comparativas de MAE, RMSE y R² entre modelos.
    """
    # Definir columnas según dataset_type
    prefix = dataset_type
    mae_col = f'{prefix}_MAE'
    rmse_col = f'{prefix}_RMSE'
    r2_col = f'{prefix}_R2'

    # Crear figura con 3 subplots (una por métrica)
    _, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Gráfico 1: MAE
    axes[0].bar(results_df['Modelo'], results_df[mae_col],
                color='skyblue', alpha=0.7)
    axes[0].set_xlabel('Model', fontsize=12)
    axes[0].set_ylabel('MAE', fontsize=12)
    axes[0].set_title(f'MAE - {dataset_type}', fontsize=14, fontweight='bold')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3, axis='y')

    # Gráfico 2: RMSE
    axes[1].bar(results_df['Modelo'], results_df[rmse_col],
                color='lightcoral', alpha=0.7)
    axes[1].set_xlabel('Model', fontsize=12)
    axes[1].set_ylabel('RMSE', fontsize=12)
    axes[1].set_title(f'RMSE - {dataset_type}', fontsize=14, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3, axis='y')

    # Gráfico 3: R²
    axes[2].bar(results_df['Modelo'], results_df[r2_col],
                color='lightgreen', alpha=0.7)
    axes[2].set_xlabel('Model', fontsize=12)
    axes[2].set_ylabel('R²', fontsize=12)
    axes[2].set_title(f'R² Score - {dataset_type}',
                      fontsize=14, fontweight='bold')
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()


def plot_feature_importance(model, feature_names, model_name="Modelo", top_n=10):
    """
    Muestra las features más importantes (solo modelos de árboles).
    """
    # Verificar que el modelo tenga feature_importances_
    if not hasattr(model, 'feature_importances_'):
        print(f"⚠️  Model {model_name} does not have feature_importances_")
        print("Only tree-based models (RandomForest, GradientBoosting, DecisionTree) have this attribute")
        return

    # Obtener importancias
    importances = model.feature_importances_

    # Crear DataFrame con features e importancias
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })

    # Ordenar por importancia descendente y tomar top_n
    feature_importance_df = feature_importance_df.sort_values(
        'Importance', ascending=False).head(top_n)

    # Crear gráfico de barras horizontales
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance_df['Feature'],
             feature_importance_df['Importance'], color='steelblue', alpha=0.7)

    # Etiquetas y formato
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.title(f'Top {top_n} Most Important Features - {model_name}',
              fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()  # Invertir eje Y para que la más importante esté arriba
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.show()
