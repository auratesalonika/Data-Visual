import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np

st.set_page_config(page_title="Dashboard Smart Farming", layout="wide")

st.title("📊 Dashboard Data Sensor Pertanian - Smart Farming")

# Load dataset
DATA_PATH = "data/Smart_Farming_Crop_Yield_2024.csv"
df = pd.read_csv(DATA_PATH)

# Pilih kolom yang relevan
df = df[[
    "soil_moisture_%", "soil_pH", "temperature_C", "rainfall_mm", "humidity_%",
    "sunlight_hours", "NDVI_index", "pesticide_usage_ml", "yield_kg_per_hectare"
]].dropna()

# --- Narasi pembuka ---
st.markdown("""
### Selamat datang di Dashboard Data Sensor Pertanian  
Dashboard ini menampilkan analisis hubungan data sensor IoT dengan hasil panen untuk membantu pengambilan keputusan pertanian berbasis data.

Data meliputi sensor suhu, kelembaban, curah hujan, kelembaban tanah, sinar matahari, dan indeks vegetasi (NDVI).
""")

# --- Heatmap korelasi ---
st.subheader("📈 Korelasi Antar Variabel Sensor dan Hasil Panen")
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="YlGnBu", ax=ax)
st.pyplot(fig)
st.markdown("""
> Dari heatmap terlihat beberapa fitur sensor yang memiliki korelasi signifikan dengan hasil panen,  
seperti suhu, kelembaban tanah, dan indeks NDVI yang berkaitan erat dengan hasil panen.
""")

# --- Scatter interaktif 2D ---
st.subheader("🌧️ Visualisasi Interaktif: Rainfall dan Humidity terhadap Yield")
fig2 = px.scatter(df, x="rainfall_mm", y="humidity_%", color="yield_kg_per_hectare",
                  color_continuous_scale='Viridis',
                  labels={"rainfall_mm": "Curah Hujan (mm)", "humidity_%": "Kelembaban (%)", "yield_kg_per_hectare": "Hasil Panen (kg/ha)"},
                  title="Hubungan Curah Hujan & Kelembaban dengan Hasil Panen")
st.plotly_chart(fig2, use_container_width=True)
st.markdown("""
> Grafik ini menunjukkan bagaimana kombinasi curah hujan dan kelembaban berpengaruh pada hasil panen.  
Warna yang lebih terang menunjukkan hasil panen yang lebih tinggi.
""")

# --- Pairplot hubungan fitur utama ---
st.subheader("🔄 Eksplorasi Hubungan Fitur Utama dengan Hasil Panen")
pairplot_fig = sns.pairplot(
    df,
    vars=["temperature_C", "soil_moisture_%", "rainfall_mm", "yield_kg_per_hectare"],
    kind="scatter"
)
pairplot_fig.fig.suptitle("Eksplorasi Hubungan Fitur Utama dengan Hasil Panen", y=1.02)
st.pyplot(pairplot_fig.fig)

# --- 3D Scatter interaktif ---
st.subheader("📊 Visualisasi 3D Interaktif: Rainfall, Humidity, dan Hasil Panen")
fig4 = px.scatter_3d(df, x="rainfall_mm", y="humidity_%", z="yield_kg_per_hectare",
                     color="yield_kg_per_hectare", color_continuous_scale="Viridis",
                     labels={"rainfall_mm": "Curah Hujan (mm)", "humidity_%": "Kelembaban (%)", "yield_kg_per_hectare": "Hasil Panen (kg/ha)"},
                     title="3D Scatter Plot: Rainfall & Humidity vs Yield")
st.plotly_chart(fig4, use_container_width=True)
st.markdown("""
> Visualisasi 3D ini memudahkan pemahaman hubungan kompleks antara curah hujan, kelembaban, dan hasil panen secara interaktif.
""")

# --- Prediksi hasil panen sederhana ---
st.subheader("🤖 Prediksi Hasil Panen Sederhana dengan Regresi Linear")

# Fitur input X: rainfall dan humidity sebagai contoh
X = df[["rainfall_mm", "humidity_%"]]
y = df["yield_kg_per_hectare"]

model = LinearRegression()
model.fit(X, y)

# Prediksi hasil panen dari data aktual
df["predicted_yield"] = model.predict(X)

# Tampilkan scatter plot hasil aktual vs prediksi
fig5 = px.scatter(df, x="yield_kg_per_hectare", y="predicted_yield",
                  labels={"yield_kg_per_hectare": "Hasil Panen Aktual", "predicted_yield": "Prediksi Hasil Panen"},
                  title="Perbandingan Hasil Panen Aktual dan Prediksi")
fig5.add_shape(
    dict(type="line", x0=df["yield_kg_per_hectare"].min(), y0=df["yield_kg_per_hectare"].min(),
         x1=df["yield_kg_per_hectare"].max(), y1=df["yield_kg_per_hectare"].max(),
         line=dict(color="red", dash="dash"))
)
st.plotly_chart(fig5, use_container_width=True)
st.markdown("""
> Model regresi linear sederhana menggunakan curah hujan dan kelembaban untuk memprediksi hasil panen.  
> Garis merah putus-putus menunjukkan garis ideal jika prediksi sama dengan hasil aktual.
""")

# --- Penutup ---
st.markdown("""
---  
### Kesimpulan  
Analisis data sensor IoT ini membuka wawasan baru bagi pengelolaan pertanian berbasis data.  
Dengan memantau variabel lingkungan secara real-time dan menganalisis hubungannya dengan hasil panen, petani dapat membuat keputusan yang lebih tepat dan efisien.

Dashboard ini siap dikembangkan lebih lanjut untuk fitur prediksi yang lebih kompleks dan integrasi data waktu nyata.
""")
