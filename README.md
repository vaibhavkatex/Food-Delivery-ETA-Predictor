# 🍔 Food Delivery ETA Predictor

A machine learning web application that predicts the estimated food delivery time in minutes from delivery-partner, order, traffic, weather, vehicle, location, distance, and time-related features.

The project uses a **Random Forest Regressor** inside a scikit-learn preprocessing pipeline and provides a simple **Streamlit** interface for making predictions.

## ✨ Features

- Predicts delivery ETA in minutes.
- Interactive Streamlit UI for entering delivery details.
- Handles both numerical and categorical features through a scikit-learn pipeline.
- Converts 12-hour order time to 24-hour format automatically.
- Derives weekday/weekend and time-period features.
- Compares multiple regression models during training.
- Saves the selected trained model with `joblib` for reuse by the Streamlit app.

## 🧠 Machine Learning Workflow

The training notebook performs the following workflow:

1. Load and inspect the cleaned Zomato food-delivery dataset.
2. Select prediction features and the target variable, `Time_taken (min)`.
3. Split the data into training and test sets using an 80/20 split with `random_state=42`.
4. Impute missing values.
5. Standardize numerical features.
6. One-hot encode categorical features with unknown-category handling.
7. Train and evaluate Linear Regression, Random Forest, and Gradient Boosting models.
8. Select Random Forest based on the lowest validation MAE.
9. Save the trained pipeline as `food_delivery_eta_model.pkl`.

### Model Performance

The notebook reports the following test-set results:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 4.69 min | 5.89 min | 0.5920 |
| Random Forest | **3.07 min** | **3.85 min** | **0.8254** |
| Gradient Boosting | 3.16 min | 3.95 min | 0.8171 |

Random Forest is selected as the final model because it achieved the lowest MAE among the evaluated models.

> **Note:** These metrics are the results recorded in the project's training notebook and may vary slightly if the data or training process changes.

## 🛠️ Tech Stack

- **Python**
- **Pandas** — data preparation and prediction input handling
- **Scikit-learn** — preprocessing, model training, and evaluation
- **Joblib** — model serialization
- **Streamlit** — interactive web application
- **Jupyter / Google Colab** — model development and experimentation

## 📁 Project Structure

```text
Food-Delivery-ETA-Predictor/
│
├── app.py                             # Streamlit prediction application
├── Food_Delivery_ETA_Predictor.ipynb  # Data preprocessing and model training
├── requirements.txt                   # Application dependencies
├── zomato_cleaned.csv                 # Cleaned training dataset
├── food_delivery_eta_model.pkl        # Trained Random Forest pipeline (generated during training)
└── README.md                          # Project documentation
```

> `food_delivery_eta_model.pkl` is required by `app.py`. The repository currently contains the training notebook that generates this file with `joblib.dump()`.

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/vaibhavkatex/Food-Delivery-ETA-Predictor.git
cd Food-Delivery-ETA-Predictor
```

### 2. Create a virtual environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 🧪 Train / Generate the Model

The model-training workflow is available in `Food_Delivery_ETA_Predictor.ipynb`.

Run the notebook from top to bottom. It preprocesses the dataset, compares the regression models, selects the Random Forest model, and generates:

```text
food_delivery_eta_model.pkl
```

The notebook uses additional packages for experimentation and visualization. If your environment does not already have them, install:

```bash
pip install numpy matplotlib seaborn jupyter
```

The notebook's final training configuration uses:

```python
RandomForestRegressor(
    n_estimators=150,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)
```

## ▶️ Run the Streamlit App

Once `food_delivery_eta_model.pkl` is available in the project root, run:

```bash
streamlit run app.py
```

Streamlit will start a local development server and provide a browser URL for the application.

## 🖥️ Using the Application

Enter the delivery information shown in the app, including:

- Delivery person age and rating
- Vehicle condition and vehicle type
- Number of multiple deliveries
- Delivery distance
- Weather conditions
- Road traffic density
- Order type
- Festival status
- City type
- Order time
- Order day

Click **Predict ETA** to get the estimated delivery time in minutes.

The app automatically derives:

- `order_hour`
- `order_weekday`
- `is_weekend`
- `time_period`

before sending the input to the trained model.

## 📊 Example Prediction

The training notebook includes an example order with values such as a 25-year-old delivery partner, 4.7 rating, 6.5 km distance, evening order time, and motorcycle vehicle type. The recorded prediction for that example is approximately **15.21 minutes**.

## 🔍 Input Features

The final model uses the following features:

**Numerical**

- `Delivery_person_Age`
- `Delivery_person_Ratings`
- `Vehicle_condition`
- `multiple_deliveries`
- `distance_km`
- `order_hour`
- `order_weekday`
- `is_weekend`

**Categorical**

- `Weather_conditions`
- `Road_traffic_density`
- `Type_of_order`
- `Type_of_vehicle`
- `Festival`
- `City`
- `time_period`

**Target**

```text
Time_taken (min)
```

## ⚠️ Important Notes

- The trained model file must be present at the path expected by `app.py`:

  ```text
  food_delivery_eta_model.pkl
  ```

- The model is serialized with `joblib`, so using a compatible scikit-learn version is important. The project pins **scikit-learn 1.6.1** in `requirements.txt`.
- The prediction application expects the feature names and categorical values used during training. Changing these values without retraining the model can cause prediction errors or reduce prediction quality.
- The dataset and notebook are included for project demonstration and experimentation; this should not be treated as a production-grade ETA service without additional validation, monitoring, and real-time operational data.

## 📚 Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit frontend and prediction logic |
| `Food_Delivery_ETA_Predictor.ipynb` | Data preparation, feature engineering, model comparison, and training |
| `zomato_cleaned.csv` | Cleaned dataset used for model development |
| `requirements.txt` | Python dependencies for the application |
| `food_delivery_eta_model.pkl` | Serialized trained Random Forest pipeline |

## 👤 Author

**Vaibhav Kate**

GitHub: https://github.com/vaibhavkatex

Project: https://github.com/vaibhavkatex/Food-Delivery-ETA-Predictor

## 📄 License

No license file is currently included in the repository. Unless a license is added, the project should be treated as **all rights reserved** by its copyright holder.
