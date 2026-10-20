import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

st.set_page_config(
    page_title="Food Delivery ETA Predictor",
    page_icon="🍔"
)

# IMPORTANT: Replace with your actual Hugging Face repository ID.
MODEL_REPO_ID = "YOUR_USERNAME/food-delivery-eta-model"
MODEL_FILENAME = "food_delivery_eta_model.pkl"

@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id=MODEL_REPO_ID,
        filename=MODEL_FILENAME,
        repo_type="model"
    )
    return joblib.load(model_path)

try:
    model = load_model()
except Exception as e:
    st.error("Model load nahi hua. Hugging Face repository ID, filename aur repository visibility check karo.")
    st.code(str(e))
    st.stop()

st.title("🍔 Food Delivery ETA Predictor")
st.write("Predict food delivery time using a Random Forest model.")

# Delivery details
st.header("Delivery Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Delivery Person Age",
        min_value=18,
        max_value=60,
        value=25
    )

    rating = st.number_input(
        "Delivery Person Rating",
        min_value=1.0,
        max_value=5.0,
        value=4.7,
        step=0.1
    )

    vehicle_condition = st.number_input(
        "Vehicle Condition",
        min_value=0,
        max_value=3,
        value=2
    )

    multiple_deliveries = st.number_input(
        "Multiple Deliveries",
        min_value=0,
        max_value=3,
        value=1
    )

    distance = st.number_input(
        "Delivery Distance (km)",
        min_value=0.1,
        max_value=50.0,
        value=6.5,
        step=0.1
    )

with col2:
    weather = st.selectbox(
        "Weather Conditions",
        ["Cloudy", "Fog", "Sandstorms", "Stormy", "Sunny", "Windy"]
    )

    traffic = st.selectbox(
        "Road Traffic Density",
        ["High", "Jam", "Low", "Medium"]
    )

    order_type = st.selectbox(
        "Type of Order",
        ["Buffet", "Drinks", "Meal", "Snack"]
    )

    vehicle = st.selectbox(
        "Type of Vehicle",
        ["electric_scooter", "motorcycle", "scooter"]
    )

    festival = st.selectbox(
        "Festival",
        ["No", "Yes"]
    )

    city = st.selectbox(
        "City",
        ["Metropolitian", "Semi-Urban", "Urban"]
    )









# ---------------- Time Details ----------------

st.header("🕒 Order Time")

col1, col2, col3 = st.columns(3)

with col1:
    hour = st.selectbox(
        "Hour",
        list(range(1, 13)),
        index=6   # Default: 7
    )

with col2:
    minute = st.selectbox(
        "Minute",
        [0, 15, 30, 45],
        format_func=lambda x: f"{x:02d}",
        index=0
    )

with col3:
    am_pm = st.selectbox(
        "AM / PM",
        ["AM", "PM"],
        index=1   # Default: PM
    )

# Convert 12-hour time to 24-hour time for the ML model
if am_pm == "AM":
    order_hour = 0 if hour == 12 else hour
else:
    order_hour = 12 if hour == 12 else hour + 12

# Display selected time
time_display = f"{hour}:{minute:02d} {am_pm}"

st.write(f"🕒 Order Time: **{time_display}**")

# ---------------- Weekday ----------------

order_weekday_name = st.selectbox(
    "📅 Order Day",
    [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ]
)

# Convert weekday name to number for the model
weekday_map = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6
}

order_weekday = weekday_map[order_weekday_name]

# Weekend feature
is_weekend = int(order_weekday >= 5)

# Time period
if order_hour < 12:
    time_period = "Morning"
elif order_hour < 17:
    time_period = "Afternoon"
elif order_hour < 21:
    time_period = "Evening"
else:
    time_period = "Night"

st.write(f"🌤️ Time Period: **{time_period}**")
st.write(f"📅 Selected Day: **{order_weekday_name}**")


# Prediction
if st.button("🚀 Predict ETA", type="primary", use_container_width=True):

    new_order = pd.DataFrame({
        "Delivery_person_Age": [age],
        "Delivery_person_Ratings": [rating],
        "Weather_conditions": [weather],
        "Road_traffic_density": [traffic],
        "Vehicle_condition": [vehicle_condition],
        "Type_of_order": [order_type],
        "Type_of_vehicle": [vehicle],
        "multiple_deliveries": [multiple_deliveries],
        "Festival": [festival],
        "City": [city],
        "distance_km": [distance],
        "order_hour": [order_hour],
        "order_weekday": [order_weekday],
        "is_weekend": [is_weekend],
        "time_period": [time_period]
    })

    try:
        prediction = model.predict(new_order)[0]
        prediction = max(0, prediction)

        st.success(
            f"🕒 Estimated Delivery Time: {prediction:.0f} minutes"
        )

    except Exception as e:
        st.error("Prediction failed.")
        st.write(e)

st.divider()
st.caption("Food Delivery ETA Predictor | Build by Vaibhav Kate")

