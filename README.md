# Car Price Prediction Dashboard

This project is an interactive Streamlit web application designed to predict the selling price of used cars using a trained Random Forest regression model.
It provides both single-car predictions and batch predictions from CSV files, along with visual analytics such as scatter plots, histograms, and correlation heatmaps.

---

## Features

### 1. Login Authentication

A simple login screen is provided to restrict access.

### 2. Single Car Price Prediction

Users can manually input car attributes through the sidebar:

* Present Price
* Kilometers Driven
* Fuel Type
* Seller Type
* Transmission
* Owner Count
* Car Age

The app processes these features, applies feature engineering, encodes categorical variables, and predicts the estimated selling price.

### 3. Performance Metrics

The dashboard displays model performance using:

* Predicted Selling Price
* R² Score
* Mean Absolute Error (MAE)

### 4. Visualizations

The interface includes:

* Predicted vs Present Price scatter plot
* Selling Price distribution with user prediction highlighted
* Feature correlation heatmap
  These visualizations help users understand relationships within the dataset and model behavior.

### 5. Batch Prediction

Users can upload a CSV file containing car data.
The app:

* Performs feature engineering
* Aligns input features with the model
* Generates predictions for all entries
* Allows downloading a CSV report of predicted prices

### 6. Model & Data Handling

The dashboard loads:

* `random_forest_model.pkl` (trained Random Forest model)
* `car_dataset.csv` (original dataset used for visualizations)

Feature engineering applied:

* Price per KM
* Age category binning
* One-hot encoding for categorical variables

---

## Project Structure

```
CAR PREDICTION/
│
├── app.py                       # Main Streamlit application
├── train_model.py               # Script to train the Random Forest model
├── car_prediction.py            # Additional prediction utilities
├── random_forest_model.pkl      # Trained Random Forest model file
├── car_dataset.csv              # Dataset used for analysis and charts
├── your_batch.csv               # Sample batch input file
├── Heatmap/                     # Generated visualization assets (optional)
└── Predicted vs Present price/  # Generated visualization assets (optional)
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/car-price-prediction.git
cd car-price-prediction
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit Application

```bash
streamlit run app.py
```

---

## Input Format for Batch Prediction

The uploaded CSV must contain the following columns:

* Present_Price
* Kms_Driven
* Fuel_Type
* Seller_Type
* Transmission
* Owner
* Car_Age

The app automatically applies feature engineering and encoding.

---

## Model Information

* **Algorithm:** Random Forest Regressor
* **Evaluation Metrics:** MAE, R² Score
* **Training Script:** `train_model.py`

The model is trained on historical car pricing data and saved as a `.pkl` file for fast loading.

---

## Outputs

The application produces:

* Estimated selling price for single input
* Analytics charts
* Batch prediction table
* Downloadable CSV report containing predicted prices

---

## Notes

* This project is intended for educational and demonstration purposes.
* Predictions depend on dataset quality and may not reflect real market values.

---

## Author

Built by **Kiruthiga**.
