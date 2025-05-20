import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Load data
DATA_PATH = "/mnt/data/Smart_Farming_Crop_Yield_2024.csv"
df = pd.read_csv(DATA_PATH)

st.title("📊 Dashboard Data Sensor Pertanian - Tanpa Filter Waktu")

# Data preview
st.subheader("Data Preview")
st.dataframe(df.head())

# Pilih fitur numerik sensor yang relevan untuk analisis
sensor_features = ['soil_moisture_%', 'soil_pH', 'temperature_C', 'rainfall_mm', 'humidity_%', 'sunlight_hours', 'pesticide_usage_ml', 'NDVI_index']
target = 'yield_kg_per_hectare'

# Pastikan kolom target ada
if target not in df.columns:
    st.error(f"Kolom '{target}' tidak ditemukan di dataset.")
else:
    st.subheader("1. Heatmap Korelasi Fitur Sensor dengan Hasil Panen")
    corr = df[sensor_features + [target]].corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap="YlGnBu", ax=ax)
    st.pyplot(fig)

    st.subheader("2. Scatterplot Interaktif Fitur Sensor vs Hasil Panen")
    # Scatter plot interaktif pakai Plotly
    fig2 = px.scatter(df, x='rainfall_mm', y='humidity_%', color=target,
                      labels={'rainfall_mm':'Rainfall (mm)', 'humidity_%':'Humidity (%)', target:'Yield (kg/ha)'},
                      title='Hubungan Rainfall dan Humidity dengan Yield')
    st.plotly_chart(fig2)

    st.subheader("3. Boxplot Status Penyakit Tanaman vs Hasil Panen")
    fig3, ax3 = plt.subplots()
    sns.boxplot(x='crop_disease_status', y=target, data=df, ax=ax3)
    ax3.set_title("Pengaruh Status Penyakit Terhadap Hasil Panen")
    st.pyplot(fig3)

    st.subheader("4. Histogram Distribusi Hasil Panen")
    fig4 = px.histogram(df, x=target, nbins=30, title='Distribusi Yield (kg/ha)')
    st.plotly_chart(fig4)

    # Narasi storytelling sederhana
    st.subheader("📝 Narasi Data Storytelling")
    st.markdown("""
    - Korelasi menunjukkan bahwa fitur seperti `rainfall_mm` dan `humidity_%` memiliki hubungan positif dengan hasil panen.
    - Scatterplot interaktif menampilkan pola bagaimana kombinasi curah hujan dan kelembaban memengaruhi hasil panen.
    - Boxplot memperlihatkan dampak status penyakit tanaman terhadap penurunan hasil panen.
    - Histogram distribusi yield membantu kita memahami sebaran hasil panen pada dataset ini.
    
    Dengan visualisasi ini, para petani dan pengambil kebijakan dapat lebih memahami faktor utama yang memengaruhi produktivitas pertanian dan mengambil keputusan berbasis data.
    """)

