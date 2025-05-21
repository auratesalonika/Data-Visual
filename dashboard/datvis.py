import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Konfigurasi halaman
st.set_page_config(page_title="📈 Analisis Sensor vs Hasil Panen", layout="wide")

# Styling
st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    .block-container {padding-top: 2rem;}
    </style>
""", unsafe_allow_html=True)

# Judul & deskripsi
st.title("🌾 Analisis Sensor Terhadap Hasil Panen")
st.caption("Dataset: Smart Farming Crop Yield 2024")
st.markdown("""
Analisis eksploratif ini bertujuan memahami bagaimana variabel lingkungan dan sensor memengaruhi **hasil panen** 
(`yield_kg_per_hectare`).  
Gunakan sidebar untuk filter & kontrol visualisasi.
""")

# Load dataset
df = pd.read_csv("data/Smart_Farming_Crop_Yield_2024.csv")

# Fitur yang relevan
features = [
    "soil_moisture_%", "soil_pH", "temperature_C", "rainfall_mm", "humidity_%",
    "sunlight_hours", "NDVI_index", "pesticide_usage_ml", "yield_kg_per_hectare"
]

df = df[features].dropna()

# Sidebar: Filter
st.sidebar.header("🔧 Filter Data")
temp_range = st.sidebar.slider("Temperature (°C)", float(df["temperature_C"].min()), float(df["temperature_C"].max()),
                               (float(df["temperature_C"].min()), float(df["temperature_C"].max())))
rain_range = st.sidebar.slider("Rainfall (mm)", float(df["rainfall_mm"].min()), float(df["rainfall_mm"].max()),
                               (float(df["rainfall_mm"].min()), float(df["rainfall_mm"].max())))

filtered_df = df[(df["temperature_C"].between(*temp_range)) & (df["rainfall_mm"].between(*rain_range))]

# Tabs Layout
tab1, tab2, tab3, tab4 = st.tabs(["📋 Data", "📊 Korelasi", "🌧️ Scatter Plot", "🧭 3D Plot"])

with tab1:
    st.subheader("📋 Cuplikan Data")
    st.dataframe(filtered_df.head(10), use_container_width=True)

with tab2:
    st.subheader("📊 Heatmap Korelasi")
    selected_features = st.multiselect("Pilih fitur untuk korelasi", features, default=features)
    corr = filtered_df[selected_features].corr()
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap="YlGnBu", ax=ax1)
    st.pyplot(fig1)

with tab3:
    st.subheader("🌧️ Scatterplot: Rainfall vs Humidity terhadap Yield")
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    scatter = ax2.scatter(filtered_df["rainfall_mm"], filtered_df["humidity_%"], 
                          c=filtered_df["yield_kg_per_hectare"], cmap="viridis", alpha=0.7)
    ax2.set_xlabel("Rainfall (mm)")
    ax2.set_ylabel("Humidity (%)")
    ax2.set_title("Rainfall vs Humidity terhadap Yield")
    cbar = fig2.colorbar(scatter, ax=ax2)
    cbar.set_label("Yield (kg/ha)")
    st.pyplot(fig2)

with tab4:
    st.subheader("🧭 3D Scatter Plot Interaktif")
    fig3 = px.scatter_3d(
        filtered_df, x="rainfall_mm", y="humidity_%", z="yield_kg_per_hectare",
        color="yield_kg_per_hectare",
        title="3D Scatter: Rainfall & Humidity vs Yield",
        labels={
            "rainfall_mm": "Rainfall (mm)",
            "humidity_%": "Humidity (%)",
            "yield_kg_per_hectare": "Yield (kg/ha)"
        },
        opacity=0.8,
        template="plotly_white"
    )
    st.plotly_chart(fig3, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("📌 Dibuat oleh **Kelompok 2** – Smart Farming 2024 | Exploratory Data Analysis")

