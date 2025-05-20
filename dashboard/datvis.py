import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np

# Konfigurasi halaman
st.set_page_config(page_title="Dashboard Smart Farming", layout="wide")
st.title("📊 Dashboard Data Sensor Pertanian - Smart Farming")

# Baca dataset dan parsing kolom tanggal
DATA_PATH = "data/Smart_Farming_Crop_Yield_2024.csv"
df = pd.read_csv(DATA_PATH, parse_dates=['timestamp', 'sowing_date'])

# Tampilkan daftar kolom
st.write("Daftar kolom dataset:", df.columns.tolist())

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

# Sidebar: Filter berdasarkan tanggal (timestamp)
st.sidebar.subheader("Filter Data Berdasarkan Tanggal")
min_date = df['timestamp'].min()
max_date = df['timestamp'].max()

start_date, end_date = st.sidebar.date_input(
    "Pilih rentang tanggal (timestamp):",
    [min_date, max_date],
    min_value=min_date,
    max_value=max_date
)

if start_date > end_date:
    st.sidebar.error("Tanggal mulai harus sebelum tanggal akhir.")
else:
    df = df[(df['timestamp'] >= pd.to_datetime(start_date)) & (df['timestamp'] <= pd.to_datetime(end_date))]

# Sidebar: Pilih fitur sensor untuk tren
sensor_options = [
    "temperature_C",
    "humidity_%",
    "soil_moisture_%",
    "rainfall_mm",
    "sunlight_hours"
]

selected_sensors = st.sidebar.multiselect(
    "Pilih fitur sensor untuk analisis tren:",
    sensor_options,
    default=["temperature_C", "humidity_%"]
)

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
if selected_sensors:
    df_plot = df[['timestamp'] + selected_sensors].dropna()
    fig_trend = px.line(df_plot, x='timestamp', y=selected_sensors,
                        labels={'timestamp': 'Waktu', 'value': 'Nilai Sensor', 'variable': 'Sensor'},
                        title="Tren Sensor dari Waktu ke Waktu")
    st.plotly_chart(fig_trend)

# Analisis hubungan sensor dengan hasil panen
target_col = "yield_kg_per_hectare"

if target_col in df.columns:
    st.subheader("📊 Hubungan Sensor dengan Hasil Panen (Interaktif)")

    selected_sensor = st.selectbox("Pilih variabel sensor:", sensor_options)

    df_scatter = df[[selected_sensor, target_col]].dropna()

    fig_scatter = px.scatter(df_scatter, x=selected_sensor, y=target_col,
                             labels={selected_sensor: selected_sensor, target_col: "Hasil Panen (kg/ha)"},
                             title=f"Hubungan antara {selected_sensor} dan Hasil Panen")
    st.plotly_chart(fig_scatter)

    # Prediksi hasil panen dengan regresi linier sederhana
    st.subheader("🤖 Prediksi Hasil Panen Sederhana dengan Regresi Linier")

    data_reg = df[[selected_sensor, target_col]].dropna()

    X = data_reg[[selected_sensor]].values.reshape(-1, 1)
    y = data_reg[target_col].values

    model = LinearRegression()
    model.fit(X, y)

    input_val = st.number_input(f"Masukkan nilai {selected_sensor} untuk prediksi hasil panen:", 
                                min_value=float(np.min(X)), max_value=float(np.max(X)), value=float(np.median(X)))

    pred_yield = model.predict(np.array([[input_val]]))[0]

    st.markdown(f"Prediksi hasil panen untuk nilai **{selected_sensor} = {input_val:.2f}** adalah: **{pred_yield:.2f} kg/ha**")

else:
    st.warning(f"Kolom '{target_col}' tidak ditemukan dalam dataset.")
