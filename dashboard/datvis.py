import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import numpy as np

st.set_page_config(page_title="Dashboard Smart Farming", layout="wide")
st.title("📊 Dashboard Data Sensor Pertanian - Smart Farming")

DATA_PATH = "/mnt/data/Smart_Farming_Crop_Yield_2024.csv"
df = pd.read_csv(DATA_PATH, parse_dates=['timestamp', 'sowing_date'])

st.subheader("🔍 Tinjauan Data")
st.dataframe(df.head())

st.markdown("""
**Fitur Penting:**
- temperature_C (Suhu dalam derajat Celcius)
- humidity_% (Kelembaban dalam persen)
- soil_moisture_% (Kelembaban tanah dalam persen)
- rainfall_mm (Curah hujan dalam mm)
- sunlight_hours (Jam penyinaran matahari)
- yield_kg_per_hectare (Hasil panen dalam kg per hektar)
""")

# Tren sensor default
default_sensors = ["temperature_C", "humidity_%", "rainfall_mm"]

st.subheader("🌡️ Tren Sensor")
df_plot = df[['timestamp'] + default_sensors].dropna()
fig_trend = px.line(df_plot, x='timestamp', y=default_sensors,
                    labels={'timestamp': 'Waktu', 'value': 'Nilai Sensor', 'variable': 'Sensor'},
                    title="Tren Sensor dari Waktu ke Waktu")
st.plotly_chart(fig_trend)

target_col = "yield_kg_per_hectare"

if target_col in df.columns:
    st.subheader("📊 Hubungan Dua Sensor dengan Hasil Panen")

    # Pilih dua sensor
    sensor_options = [
        "temperature_C",
        "humidity_%",
        "soil_moisture_%",
        "rainfall_mm",
        "sunlight_hours"
    ]
    sensor1 = st.selectbox("Pilih sensor 1:", sensor_options, index=sensor_options.index("rainfall_mm"))
    sensor2 = st.selectbox("Pilih sensor 2:", sensor_options, index=sensor_options.index("humidity_%"))

    data = df[[sensor1, sensor2, target_col]].dropna()

    X = data[[sensor1, sensor2]].values
    y = data[target_col].values

    # Fit multiple linear regression
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    # Tampilkan koefisien regresi
    st.markdown(f"**Model Regresi:**  \n`Hasil Panen = {model.intercept_:.2f} + {model.coef_[0]:.2f} * {sensor1} + {model.coef_[1]:.2f} * {sensor2}`")

    # Scatter 3D plot hasil panen vs dua sensor
    fig_3d = px.scatter_3d(data, x=sensor1, y=sensor2, z=target_col, color=target_col,
                           title=f"3D Scatter: {sensor1} & {sensor2} terhadap Hasil Panen",
                           labels={sensor1: sensor1, sensor2: sensor2, target_col: "Hasil Panen (kg/ha)"})
    st.plotly_chart(fig_3d)

    # Prediksi hasil panen dari input user (readonly output)
    st.subheader("🤖 Prediksi Hasil Panen dengan Dua Sensor")

    val1 = st.slider(f"Masukkan nilai {sensor1}:", float(data[sensor1].min()), float(data[sensor1].max()), float(data[sensor1].median()))
    val2 = st.slider(f"Masukkan nilai {sensor2}:", float(data[sensor2].min()), float(data[sensor2].max()), float(data[sensor2].median()))

    pred_yield = model.predict(np.array([[val1, val2]]))[0]
    st.markdown(f"**Prediksi hasil panen untuk {sensor1} = {val1:.2f} dan {sensor2} = {val2:.2f} adalah:**  \n**{pred_yield:.2f} kg/ha**")

else:
    st.warning(f"Kolom '{target_col}' tidak ditemukan dalam dataset.")

