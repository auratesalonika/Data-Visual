import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import numpy as np

# Konfigurasi halaman
st.set_page_config(page_title="Dashboard Smart Farming", layout="wide")
st.title("📊 Dashboard Data Sensor Pertanian - Smart Farming")

# Baca dataset dan parsing kolom tanggal
DATA_PATH = "data/Smart_Farming_Crop_Yield_2024.csv"
df = pd.read_csv(DATA_PATH, parse_dates=['timestamp', 'sowing_date'])

# Tampilkan data awal
st.subheader("🔍 Tinjauan Data")
st.dataframe(df.head())

# Info dataset
st.markdown("""
**Fitur Penting:**
- temperature_C (Suhu dalam derajat Celcius)
- humidity_% (Kelembaban dalam persen)
- soil_moisture_% (Kelembaban tanah dalam persen)
- rainfall_mm (Curah hujan dalam mm)
- sunlight_hours (Jam penyinaran matahari)
- yield_kg_per_hectare (Hasil panen dalam kg per hektar)
""")

# Jelaskan maksud tren sensor
st.markdown("""
### 🌡️ Tren Sensor
Visualisasi tren sensor menunjukkan perubahan nilai sensor (seperti suhu, kelembaban, curah hujan) dari waktu ke waktu.  
Ini membantu kita memahami pola dan fluktuasi kondisi lingkungan pertanian sepanjang periode pengamatan.
""")

# Sensor default untuk tren
default_sensors = ["temperature_C", "humidity_%", "rainfall_mm"]

# Visualisasi korelasi numerik (heatmap)
st.subheader("📈 Korelasi Antar Variabel")
numerical_df = df.select_dtypes(include='number')
fig_corr = px.imshow(numerical_df.corr(), 
                     text_auto=True, 
                     color_continuous_scale="YlGnBu", 
                     title="Matriks Korelasi Antar Fitur Numerik")
st.plotly_chart(fig_corr)

# Visualisasi tren sensor interaktif dengan Plotly
st.subheader("🌡️ Tren Sensor Interaktif")
df_plot = df[['timestamp'] + default_sensors].dropna()
fig_trend = px.line(df_plot, x='timestamp', y=default_sensors,
                    labels={'timestamp': 'Waktu', 'value': 'Nilai Sensor', 'variable': 'Sensor'},
                    title="Tren Sensor dari Waktu ke Waktu")
st.plotly_chart(fig_trend)

# Hubungan sensor dengan hasil panen dengan scatter + regresi
target_col = "yield_kg_per_hectare"

if target_col in df.columns:
    st.subheader("📊 Hubungan Sensor dengan Hasil Panen")

    sensor_options = [
        "temperature_C",
        "humidity_%",
        "soil_moisture_%",
        "rainfall_mm",
        "sunlight_hours"
    ]

    selected_sensor = st.selectbox("Pilih variabel sensor:", sensor_options)

    data = df[[selected_sensor, target_col]].dropna()

    # Fit regresi linier
    X = data[[selected_sensor]].values.reshape(-1, 1)
    y = data[target_col].values
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    # Buat scatter plot + garis regresi
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data[selected_sensor], y=data[target_col], mode='markers', name='Data'))
    fig.add_trace(go.Scatter(x=data[selected_sensor], y=y_pred, mode='lines', name='Regresi Linier'))
    fig.update_layout(
        title=f"Hubungan {selected_sensor} dengan Hasil Panen",
        xaxis_title=selected_sensor,
        yaxis_title="Hasil Panen (kg/ha)"
    )
    st.plotly_chart(fig)

    # Prediksi hasil panen (readonly)
    st.subheader("🤖 Prediksi Hasil Panen Sederhana dengan Regresi Linier")
    input_val = st.slider(f"Masukkan nilai {selected_sensor} untuk prediksi hasil panen:",
                          float(np.min(X)), float(np.max(X)), float(np.median(X)))

    pred_yield = model.predict(np.array([[input_val]]))[0]

    # Tampilkan hasil prediksi sebagai teks saja, tidak bisa diubah
    st.markdown(f"**Prediksi hasil panen untuk nilai {selected_sensor} = {input_val:.2f} adalah:**  \n**{pred_yield:.2f} kg/ha**")

else:
    st.warning(f"Kolom '{target_col}' tidak ditemukan dalam dataset.")
