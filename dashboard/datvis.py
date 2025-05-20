import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# ----- Set page config -----
st.set_page_config(page_title="Dashboard Smart Farming", layout="wide")
st.title("📊 Dashboard Data Sensor Pertanian - Smart Farming")

# ----- Load data -----
DATA_PATH = "/mnt/data/Smart_Farming_Crop_Yield_2024.csv"
df = pd.read_csv(DATA_PATH)

# Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['sowing_date'] = pd.to_datetime(df['sowing_date'])
df['harvest_date'] = pd.to_datetime(df['harvest_date'])

# ----- Sidebar filter -----
st.sidebar.header("Filter Waktu")
start_date = st.sidebar.date_input("Mulai tanggal", df['timestamp'].min().date())
end_date = st.sidebar.date_input("Sampai tanggal", df['timestamp'].max().date())

if start_date > end_date:
    st.sidebar.error("Error: Mulai tanggal harus sebelum sampai tanggal.")

# Filter data berdasarkan timestamp
mask = (df['timestamp'] >= pd.Timestamp(start_date)) & (df['timestamp'] <= pd.Timestamp(end_date))
filtered_df = df.loc[mask]

# Pilih agregasi
st.sidebar.header("Agregasi Data")
agg_option = st.sidebar.radio("Pilih agregasi waktu:", ('Harian', 'Mingguan'))

if agg_option == 'Harian':
    agg_df = filtered_df.groupby(filtered_df['timestamp'].dt.date).mean().reset_index()
    agg_df['timestamp'] = pd.to_datetime(agg_df['timestamp'])
else:
    agg_df = filtered_df.groupby(filtered_df['timestamp'].dt.to_period('W')).mean().reset_index()
    agg_df['timestamp'] = agg_df['timestamp'].dt.start_time

# ----- Visualisasi Tren Sensor Utama -----
st.subheader("🌡️ Tren Sensor Utama")
main_sensors = ['temperature_C', 'humidity_%', 'rainfall_mm', 'soil_moisture_%']
available_sensors = [col for col in main_sensors if col in agg_df.columns]

fig = px.line(
    agg_df,
    x='timestamp',
    y=available_sensors,
    labels={'value': 'Nilai Sensor', 'timestamp': 'Tanggal'},
    title="Tren Sensor Pertanian (Agregasi {})".format(agg_option)
)
st.plotly_chart(fig, use_container_width=True)

# ----- Analisis Hubungan Fitur dengan Hasil Panen -----
st.subheader("📊 Eksplorasi Hubungan Fitur Utama dengan Hasil Panen")

# Fitur sensor yang relevan
feature_cols = ['temperature_C', 'humidity_%', 'rainfall_mm', 'soil_moisture_%', 'soil_pH', 'sunlight_hours', 'NDVI_index', 'pesticide_usage_ml']
feature_cols = [col for col in feature_cols if col in df.columns]

if 'yield_kg_per_hectare' not in df.columns:
    st.warning("Kolom 'yield_kg_per_hectare' tidak ditemukan dalam dataset.")
else:
    st.markdown("### Heatmap Korelasi")
    corr_df = df[feature_cols + ['yield_kg_per_hectare']].corr()
    fig2, ax2 = plt.subplots(figsize=(10,6))
    sns.heatmap(corr_df, annot=True, cmap="YlGnBu", ax=ax2)
    st.pyplot(fig2)

    st.markdown("### Pairplot Fitur dan Hasil Panen")
    sns.set(style="ticks")
    pairplot_df = df[feature_cols + ['yield_kg_per_hectare']].dropna()
    pair_fig = sns.pairplot(pairplot_df)
    st.pyplot(pair_fig)

    st.markdown("### Scatterplot Interaktif: Hubungan Dua Fitur Sensor dengan Hasil Panen")
    x_opt = st.selectbox("Pilih fitur X:", feature_cols, index=0)
    y_opt = st.selectbox("Pilih fitur Y:", feature_cols, index=1 if len(feature_cols)>1 else 0)

    fig3 = px.scatter(
        df,
        x=x_opt,
        y=y_opt,
        color='yield_kg_per_hectare',
        color_continuous_scale='Viridis',
        labels={'color':'Yield (kg/ha)'},
        title=f"Hubungan antara {x_opt} dan {y_opt} dengan Hasil Panen"
    )
    st.plotly_chart(fig3, use_container_width=True)

# ----- Prediksi Hasil Panen Sederhana -----
st.subheader("🤖 Prediksi Hasil Panen Sederhana dengan Regresi Linear")

# Pilih fitur input prediksi
pred_features = ['temperature_C', 'humidity_%', 'rainfall_mm', 'soil_moisture_%', 'soil_pH', 'sunlight_hours', 'NDVI_index', 'pesticide_usage_ml']
pred_features = [col for col in pred_features if col in df.columns]

df_model = df.dropna(subset=pred_features + ['yield_kg_per_hectare'])
X = df_model[pred_features]
y = df_model['yield_kg_per_hectare']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Visualisasi hasil prediksi vs aktual
fig4 = px.scatter(x=y_test, y=y_pred, labels={'x':'Yield Aktual', 'y':'Yield Prediksi'}, title="Perbandingan Yield Aktual vs Prediksi")
fig4.add_shape(
    type='line', line=dict(dash='dash'),
    x0=y_test.min(), y0=y_test.min(),
    x1=y_test.max(), y1=y_test.max()
)
st.plotly_chart(fig4, use_container_width=True)

# Tampilkan skor model
score = model.score(X_test, y_test)
st.markdown(f"**Koefisien Determinasi (R²) model regresi:** {score:.3f}")

st.markdown("""
---
### Narasi Data Storytelling
- Filter waktu memungkinkan kita melihat tren dan pola data sensor secara spesifik dalam rentang tanggal tertentu.
- Agregasi data harian atau mingguan membantu mereduksi noise dan memudahkan visualisasi tren jangka panjang.
- Dari heatmap korelasi, kita bisa mengidentifikasi fitur sensor yang paling berpengaruh terhadap hasil panen.
- Visualisasi scatter interaktif menghubungkan dua fitur sensor dengan hasil panen, memudahkan eksplorasi pola hubungan.
- Model regresi linear sederhana memberikan prediksi hasil panen berdasarkan fitur sensor utama, yang dapat dipakai sebagai dasar pengambilan keputusan pertanian berbasis data.
""")
