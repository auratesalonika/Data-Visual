import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Konfigurasi halaman
st.set_page_config(page_title="Analisis Hasil Panen", layout="wide")

st.title("🌾 Analisis Hubungan Sensor dengan Hasil Panen")
st.markdown("Dataset: **Smart Farming Crop Yield 2024**")

# Load dataset dari file yang sudah diunggah
df = pd.read_csv("data/Smart_Farming_Crop_Yield_2024.csv")

# Pilih fitur yang relevan
features = [
    "soil_moisture_%", "soil_pH", "temperature_C", "rainfall_mm", "humidity_%",
    "sunlight_hours", "NDVI_index", "pesticide_usage_ml", "yield_kg_per_hectare"
]

df = df[features].dropna()

# Tampilkan data awal
st.subheader("🔍 Cuplikan Data")
st.dataframe(df.head())

# Heatmap Korelasi
st.subheader("📊 Heatmap Korelasi")
fig1, ax1 = plt.subplots(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="YlGnBu", ax=ax1)
st.pyplot(fig1)

# Scatterplot Rainfall vs Humidity
st.subheader("🌧️ Scatterplot: Rainfall vs Humidity terhadap Yield")
fig2, ax2 = plt.subplots(figsize=(8, 6))
scatter = ax2.scatter(df["rainfall_mm"], df["humidity_%"], c=df["yield_kg_per_hectare"], cmap="viridis")
ax2.set_xlabel("Rainfall (mm)")
ax2.set_ylabel("Humidity (%)")
ax2.set_title("Rainfall vs Humidity terhadap Yield")
cbar = fig2.colorbar(scatter, ax=ax2)
cbar.set_label("Yield (kg/ha)")
st.pyplot(fig2)

# Pairplot
st.subheader("🔗 Pairplot: Sensor terhadap Yield")
with st.spinner("Membuat visualisasi..."):
    fig3 = sns.pairplot(df, vars=["temperature_C", "soil_moisture_%", "rainfall_mm", "yield_kg_per_hectare"])
    st.pyplot(fig3)

# Plotly 3D Scatter
st.subheader("🧭 3D Scatter Plot Interaktif")
fig4 = px.scatter_3d(
    df, x="rainfall_mm", y="humidity_%", z="yield_kg_per_hectare",
    color="yield_kg_per_hectare",
    title="3D Scatter: Rainfall & Humidity vs Yield",
    labels={
        "rainfall_mm": "Rainfall (mm)",
        "humidity_%": "Humidity (%)",
        "yield_kg_per_hectare": "Yield (kg/ha)"
    }
)
st.plotly_chart(fig4)

st.success("✅ Visualisasi selesai! Gunakan sidebar atau upload ulang untuk analisis lain.")
