import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# Konfigurasi halaman
st.set_page_config(page_title="Dashboard Smart Farming with Time Filter", layout="wide")
st.title("📊 Dashboard Data Sensor Pertanian - Smart Farming (Dengan Filter Waktu)")

# Load dataset
DATA_PATH = "data/Smart_Farming_Crop_Yield_2024.csv"
df = pd.read_csv(DATA_PATH)

# Pastikan kolom timestamp dalam bentuk datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Sidebar untuk filter waktu dan agregasi
st.sidebar.header("Filter Data")
min_date = df['timestamp'].min().date()
max_date = df['timestamp'].max().date()

date_range = st.sidebar.date_input("Pilih rentang tanggal:", [min_date, max_date], min_value=min_date, max_value=max_date)
aggregation = st.sidebar.selectbox("Pilih agregasi data:", ["Harian", "Mingguan"])

# Filter data berdasarkan tanggal
if len(date_range) == 2:
    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    filtered_df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
else:
    filtered_df = df.copy()

st.subheader(f"Data Sensor dari {start_date.date()} sampai {end_date.date()}")
st.dataframe(filtered_df.head())

# Agregasi data
if aggregation == "Harian":
    filtered_df['date'] = filtered_df['timestamp'].dt.date
    agg_df = filtered_df.groupby('date')[['temperature_C', 'humidity_%', 'soil_moisture_%', 'rainfall_mm']].mean().reset_index()
    x_col = 'date'
else:
    filtered_df['week'] = filtered_df['timestamp'].dt.to_period('W').apply(lambda r: r.start_time)
    agg_df = filtered_df.groupby('week')[['temperature_C', 'humidity_%', 'soil_moisture_%', 'rainfall_mm']].mean().reset_index()
    x_col = 'week'

# Visualisasi tren waktu interaktif dengan Plotly
st.subheader("📈 Tren Waktu Sensor Pertanian")
fig = px.line(
    agg_df,
    x=x_col,
    y=['temperature_C', 'humidity_%', 'soil_moisture_%', 'rainfall_mm'],
    labels={
        x_col: "Tanggal",
        "value": "Rata-rata Nilai Sensor",
        "variable": "Sensor"
    },
    title=f"Tren Sensor Pertanian ({aggregation})"
)
st.plotly_chart(fig, use_container_width=True)

# Analisis hubungan fitur utama dengan hasil panen
st.subheader("🔍 Eksplorasi Hubungan Fitur Utama dengan Hasil Panen")
# Kolom yang berpotensi berpengaruh
features = ['temperature_C', 'humidity_%', 'soil_moisture_%', 'rainfall_mm', 'sunlight_hours', 'NDVI_index', 'soil_pH', 'pesticide_usage_ml']

# Pastikan kolom target yield_kg_per_hectare ada
if 'yield_kg_per_hectare' in filtered_df.columns:
    # Scatter plot matriks dengan seaborn pairplot (tampilan statis)
    st.markdown("Scatterplot Matrix (Pairplot) fitur utama dan hasil panen:")
    pairplot_df = filtered_df[features + ['yield_kg_per_hectare']].dropna()
    fig2 = sns.pairplot(pairplot_df)
    st.pyplot(fig2)

    # Heatmap korelasi
    st.markdown("Heatmap korelasi fitur dan hasil panen:")
    corr = pairplot_df.corr()
    fig3, ax3 = plt.subplots(figsize=(8,6))
    sns.heatmap(corr, annot=True, cmap="YlGnBu", ax=ax3)
    st.pyplot(fig3)

    # Scatter plot 2 fitur sensor dengan warna hasil panen
    st.markdown("Visualisasi dua fitur sensor terhadap hasil panen:")
    fig4 = px.scatter(
        filtered_df,
        x='rainfall_mm',
        y='humidity_%',
        color='yield_kg_per_hectare',
        color_continuous_scale='Viridis',
        labels={
            'rainfall_mm': 'Curah Hujan (mm)',
            'humidity_%': 'Kelembaban (%)',
            'yield_kg_per_hectare': 'Hasil Panen (kg/ha)'
        },
        title="Pengaruh Curah Hujan dan Kelembaban terhadap Hasil Panen"
    )
    st.plotly_chart(fig4, use_container_width=True)

else:
    st.warning("Kolom 'yield_kg_per_hectare' tidak ditemukan dalam dataset untuk analisis hubungan dan prediksi.")

# Prediksi hasil panen sederhana dengan regresi linear
st.subheader("📊 Prediksi Hasil Panen Sederhana")
if 'yield_kg_per_hectare' in filtered_df.columns:
    pred_features = ['temperature_C', 'humidity_%', 'soil_moisture_%', 'rainfall_mm']
    pred_df = filtered_df.dropna(subset=pred_features + ['yield_kg_per_hectare'])

    X = pred_df[pred_features]
    y = pred_df['yield_kg_per_hectare']

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    pred_df = pred_df.assign(predicted_yield=y_pred)

    fig_pred = px.scatter(
        pred_df,
        x='yield_kg_per_hectare',
        y='predicted_yield',
        labels={
            'yield_kg_per_hectare': 'Hasil Panen Aktual (kg/ha)',
            'predicted_yield': 'Hasil Panen Prediksi (kg/ha)'
        },
        title='Perbandingan Hasil Panen Aktual dan Prediksi'
    )
    st.plotly_chart(fig_pred, use_container_width=True)

    # Tampilkan koefisien model
    coef_df = pd.DataFrame({
        'Fitur': pred_features,
        'Koefisien': model.coef_
    })
    st.markdown("Koefisien regresi linear:")
    st.dataframe(coef_df)

    st.markdown("""
    **Narasi Data Storytelling:**

    - Model regresi linear sederhana digunakan untuk memprediksi hasil panen berdasarkan 4 fitur sensor utama.
    - Koefisien positif/negatif menunjukkan pengaruh masing-masing fitur terhadap hasil panen.
    - Grafik perbandingan aktual vs prediksi memperlihatkan seberapa baik model sederhana ini dalam memperkirakan hasil panen.
    - Informasi ini dapat membantu petani dalam mengambil keputusan berbasis data untuk meningkatkan hasil panen.
    """)
else:
    st.warning("Data tidak cukup untuk melakukan prediksi hasil panen.")

