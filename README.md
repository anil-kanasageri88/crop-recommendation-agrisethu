# 🌾 Crop Recommendation System

🔗 **Live Demo:** [Try the app here](https://crop-recommendation-agrisethu-2ffxltgx8blfmj4m3tu82a.streamlit.app/)

A machine learning system that recommends the most suitable crop to grow based on soil nutrients (N, P, K) and climate conditions (temperature, humidity, pH, rainfall).



---

## Overview

Farmers often struggle to decide which crop suits their land's soil and climate best. This project uses historical agricultural data to train a classification model that predicts the optimal crop given a set of soil and weather parameters, then exposes that model through a simple interactive interface.

## Dataset

- **Source:** `Crop_recommendation.csv` (public dataset, 2200 rows)
- **Features:** N (Nitrogen), P (Phosphorus), K (Potassium), temperature (°C), humidity (%), pH, rainfall (mm)
- **Target:** 22 crop labels (rice, maize, coffee, mango, watermelon, etc.)
- **Data quality:** No missing values, no duplicate rows. Outliers per crop group were detected using the IQR method and capped rather than removed, to preserve sample size.

## Approach

1. **EDA** — correlation heatmap, pairplots, and boxplots to understand feature relationships and distributions across crop types.
2. **Outlier handling** — IQR-based capping applied per crop group (since acceptable ranges of N/P/K/rainfall differ significantly by crop).
3. **Preprocessing** — numeric features scaled using `StandardScaler` inside a `ColumnTransformer`.
4. **Model** — `LogisticRegression` (max_iter=1000), wrapped in a single `sklearn` `Pipeline` alongside the preprocessor, so raw input goes straight in and a crop label comes straight out.
5. **Evaluation** — accuracy score and a confusion matrix across all 22 crop classes.
6. **Interface** — an interactive Jupyter widget (ipywidgets) lets a user enter the 7 parameters and get an instant prediction with confidence scores.

## Model Performance

| Metric | Score |
|---|---|
| Test Accuracy | **96.0%** |
| Training Accuracy | 97.8% |

Training and test accuracy are close, indicating the model generalizes well rather than overfitting.

## Project Structure

```
crop-recommendation/
├── crop_recommendation.ipynb   # Full notebook: EDA, preprocessing, training, evaluation, interface
├── model.pkl                    # Trained pipeline (scaler + logistic regression), saved with joblib
├── Crop_recommendation.csv      # Dataset
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Setup Instructions

1. **Clone this repository**
   ```bash
   git clone <your-repo-link>
   cd crop-recommendation
   ```

2. **Install dependencies**
   ```bash
   python -m pip install -r requirements.txt
   ```

3. **Run the notebook**
   ```bash
   jupyter notebook crop_recommendation.ipynb
   ```
   Run all cells in order. The last cell displays an interactive form — enter values for N, P, K, temperature, humidity, pH, and rainfall, then click **Recommend Crop** to see the prediction and top-5 confidence scores.

   > If `model.pkl` isn't found, re-run the earlier cells to retrain and regenerate it, or ensure `Crop_recommendation.csv` is in the same folder.

## Example Prediction

| N | P | K | Temp | Humidity | pH | Rainfall | Predicted Crop |
|---|---|---|---|---|---|---|---|
| 90 | 42 | 43 | 20.88 | 82.0 | 6.5 | 202.94 | Rice (70% confidence) |

## Tech Stack

- Python, pandas, NumPy
- scikit-learn (Pipeline, ColumnTransformer, StandardScaler, LogisticRegression)
- matplotlib, seaborn (EDA visualizations)
- ipywidgets (interactive interface)
- joblib (model persistence)

## Author

Anil Kanasageri
GitHub: [anil-kanasageri88](https://github.com/anil-kanasageri88)

## Challenges & Design Decisions

- **Outlier handling was done per-crop-group**, not globally — because what's a normal rainfall/nutrient level for rice is an outlier for a drought-tolerant crop like millet. A global IQR cutoff would have wrongly flagged valid data.
- **Logistic Regression was chosen** over more complex models for its interpretability and strong baseline performance (96% test accuracy) on this well-structured, low-noise dataset — a good fit for a first version. Tree-based models could be explored next for further gains.
- **Interface built with ipywidgets** instead of a standalone web app, chosen for reliability of local execution and to keep evaluation frictionless.
