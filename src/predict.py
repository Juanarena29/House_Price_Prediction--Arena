import os
from typing import Optional
import joblib
from src.data_preprocessing import prepare_new_data
from src.utils import ensure_directory


def predict_from_files(
    data_path: str = os.path.join("data", "inherited_houses.csv"),
    pipeline_path: str = os.path.join("models", "pipeline.pkl"),
    model_path: str = os.path.join("models", "best_model.pkl"),
    output_path: Optional[str] = os.path.join("data", "predictions.csv"),
):
    """Carga pipeline y modelo entrenado para predecir sobre un CSV nuevo."""

    if not os.path.exists(pipeline_path):
        raise FileNotFoundError(f"No se encontró el pipeline: {pipeline_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró el modelo: {model_path}")

    new_data = prepare_new_data(data_path)

    pipeline = joblib.load(pipeline_path)
    model = joblib.load(model_path)

    new_data_prepared = pipeline.transform(new_data)
    predictions = model.predict(new_data_prepared)

    results = new_data.copy()
    results["PredictedPrice"] = predictions

    if output_path:
        ensure_directory(os.path.dirname(output_path))
        results.to_csv(output_path, index=False)
        print(f"✓ Predicciones guardadas en {output_path}")

    return results


if __name__ == "__main__":
    predict_from_files()
