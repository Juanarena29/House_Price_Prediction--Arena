# House Price Prediction - Ames Housing Project

This project develops a supervised Machine Learning model to predict house sale prices using data from the Ames Housing Dataset (Kaggle).

The main objective is to demonstrate best practices in:
- data preparation (cleaning, encoding, and scaling),
- training regression models with hyperparameter optimization (GridSearchCV),
- evaluation using metrics (MAE, RMSE, R²),
- overfitting detection and error analysis,
- professional presentation through notebooks.

## 📊 Presentation Notebooks

This project includes **professional notebooks** that combine storytelling with technical analysis:

- **[notebooks/EDA.ipynb](notebooks/EDA.ipynb)**: Exploratory Data Analysis with narrative, visualizations guided by business questions, and explanation of preprocessing decisions.
- **[notebooks/Modeling_Report.ipynb](notebooks/02_Modeling_Report.ipynb)**: Comprehensive technical report with end-to-end pipeline, model comparison, residual analysis, and metrics evaluation.

Project Structure
```
ProyectoML/
│
├── data/
│   ├── house_prices_records.csv     # Main dataset
│   ├── house-metadata.txt           # Variable dictionary
│   └── inherited_houses.csv         # Data for prediction
│
├── models/
│   ├── best_model.pkl               # Best trained model
│   └── pipeline.pkl                 # Preprocessing pipeline
│
├── notebooks/
│   ├── EDA.ipynb                    # Professional exploratory analysis
│   └── Modeling_Report.ipynb        # Technical modeling report
│
├── src/
│   ├── data_preprocessing.py        # Cleaning and missing value handling
│   ├── train_model.py               # Training with GridSearchCV
│   ├── evaluate.py                  # Metrics and visualizations
│   ├── predict.py                   # Predictions with saved model
│   └── utils.py                     # Utility functions
│
├── tests/                           # Test suite with pytest
│   ├── test_data_preprocessing.py   # Tests for data cleaning and transformation
│   ├── test_evaluate.py             # Tests for metrics calculation and validation
│   └── test_train_model.py          # Tests for model training pipeline
│
├── main.py                          # Main script that coordinates everything
├── requirements.txt                 # Required libraries
└── README.md                        # This file
```

## Installation and Usage

### 1. Clone the repository
```bash
git clone <repository-url>
cd ProyectoML
```

### 2. Create virtual environment and install dependencies
```bash
python -m venv venv
venv\Scripts\activate  # On Windows
# source venv/bin/activate  # On Linux/Mac

pip install -r requirements.txt
```

### 3. Explore the notebooks (recommended)
Open Jupyter or VS Code and navigate to:
- `notebooks/EDA.ipynb` to view the exploratory analysis
- `notebooks/02_Modeling_Report.ipynb` for the complete technical report

### 4. Run the complete pipeline from terminal
```bash
python main.py
```

This executes:
1. Dataset loading and cleaning
2. Preprocessing with pipeline (StandardScaler + OneHotEncoder)
3. Training multiple models with GridSearchCV and cross-validation
4. Comparative evaluation with metrics (MAE, RMSE, R²)
5. Visualizations of predictions, residuals, and model comparison
6. Saving the best model and pipeline in `models/`

## Expected Results

The program prints comparative metrics for all trained models and generates visualizations that compare:
- Predictions vs actual values
- Residual analysis
- Metric comparison (MAE, RMSE, R²) across models
- Overfitting detection (difference between train and test)

Example console output:

```
====================================
  MODEL COMPARISON
====================================

Model               Train_MAE  Train_R2  Test_MAE  Test_R2  Overfitting
RandomForest        15,234     0.92      18,567    0.87     False
GradientBoosting    16,890     0.90      19,234    0.86     False
DecisionTree        12,045     0.95      24,789    0.78     True

   BEST MODEL: RandomForest
   Test R²: 0.8700
   Test MAE: $18,567
```

## What are model.pkl and pipeline.pkl

**model.pkl** contains the best Machine Learning model already trained (e.g., RandomForest or GradientBoosting), saved with the joblib library.

**pipeline.pkl** contains the preprocessing pipeline (StandardScaler + OneHotEncoder) fitted with the training data.

This allows:
- Loading the model without having to retrain
- Applying the same preprocessing to new data
- Ensuring consistency in production

Example usage for prediction:

```python
from joblib import load
import pandas as pd

# Load artifacts
pipeline = load('models/pipeline.pkl')
model = load('models/best_model.pkl')

# Prepare new data
new_data = pd.read_csv('data/inherited_houses.csv')
processed_data = pipeline.transform(new_data)

# Predict
predictions = model.predict(processed_data)
print(predictions)
```

## Metrics Used

- **MAE (Mean Absolute Error)**: Average absolute error in dollars. Easy to interpret: "on average we're off by $X".
- **RMSE (Root Mean Squared Error)**: Root mean squared error. Penalizes large errors more heavily.
- **R² (Coefficient of determination)**: Measures what percentage of price variability is explained by the model (0-1, higher is better).

**Overfitting Detection**: If the difference between train and test R² is > 0.10, the model is memorizing rather than learning generalizable patterns.

## Technologies Used

- **Python 3.x**
- **Data analysis**: pandas, numpy
- **Machine Learning**: scikit-learn (RandomForest, GradientBoosting, DecisionTree, GridSearchCV)
- **Visualization**: matplotlib, seaborn
- **Persistence**: joblib
- **Testing**: pytest

## Educational Objective and Best Practices

This project reinforces key concepts of professional Machine Learning:

✅ **Data splitting**: Train/test split with reproducibility (random_state)  
✅ **Preprocessing without data leakage**: Pipeline fitted only with train data  
✅ **Hyperparameter optimization**: GridSearchCV with cross-validation  
✅ **Robust evaluation**: Comparison of multiple models with standard metrics  
✅ **Overfitting detection**: Analysis of differences between train and test  
✅ **Visual interpretation**: Plots of predictions, residuals, and metrics  
✅ **Modularity**: Code organized in reusable functions with tests  
✅ **Reproducibility**: Fixed seeds, saved pipeline, clear documentation  
✅ **Professional presentation**: Notebooks with technical storytelling

## Dataset Source

Ames Housing Dataset - Kaggle
https://www.kaggle.com/datasets/codeinstitute/housing-prices-data
