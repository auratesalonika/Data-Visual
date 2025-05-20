import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np

# Konfigurasi halaman
st.set_page_config(page_title="Dashboard Smart Farming", layout="wide")
st.title("📊 Dashboard Data Sensor Pertanian - Smart Farming")

# Baca dataset
DATA_PATH = "data/Smart_Farming_Crop_Yield_2024.csv"
df = pd.read_csv(DATA_PATH)

# Tampilkan data awal
st.subheader("🔍 Tinjauan Data")
st.dataframe(df.head())

# Info dataset
st.markdown("""
**Fitur Penting:**
- temperature_C
- humidity_%
- soil_moisture_%
- rainfall_mm
- yield_kg_per_hectare
""")

# Visualisasi korelasi antar fitur numerik
st.subheader("📈 Korelasi Antar Variabel")
numerical_cols = ["temperature_C", "humidity_%", "soil_moisture_%", "rainfall_mm", "yield_kg_per_hectare"]
corr_df = df[numerical_cols].corr()
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(corr_df, annot=True, cmap="YlGnBu", ax=ax)
st.pyplot(fig)

# Eksplorasi hubungan fitur utama dengan hasil panen
st.subheader("🔄 Eksplorasi Hubungan Fitur Utama dengan Hasil Panen")
pairplot_fig = sns.pairplot(df, vars=numerical_cols, kind="scatter")
pairplot_fig.fig.suptitle("Eksplorasi Hubungan Fitur Utama dengan Hasil Panen", y=1.02)
st.pyplot(pairplot_fig.fig)

# Visualisasi hubungan Rainfall dan Humidity terhadap hasil panen
st.subheader("🌦️ Hubungan Rainfall dan Humidity dengan Hasil Panen")
fig2, ax2 = plt.subplots(figsize=(8,6))
scatter = ax2.scatter(df["rainfall_mm"], df["humidity_%"], c=df["yield_kg_per_hectare"], cmap="viridis", alpha=0.7)
ax2.set_xlabel("Rainfall (mm)")
ax2.set_ylabel("Humidity (%)")
cbar = plt.colorbar(scatter, ax=ax2)
cbar.set_label("Yield (kg/ha)")
st.pyplot(fig2)

# Prediksi hasil panen sederhana dengan regresi linear menggunakan fitur Rainfall dan Humidity
st.subheader("📊 Prediksi Hasil Panen Sederhana dengan Regresi Linear")

X = df[["rainfall_mm", "humidity_%"]].values
y = df["yield_kg_per_hectare"].values

model = LinearRegression()
model.fit(X, y)

# Prediksi hasil panen berdasarkan data aktual
y_pred = model.predict(X)

fig3, ax3 = plt.subplots(figsize=(8,6))
ax3.scatter(y, y_pred, alpha=0.7)
ax3.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
ax3.set_xlabel("Yield Aktual (kg/ha)")
ax3.set_ylabel("Yield Prediksi (kg/ha)")
ax3.set_title("Prediksi Hasil Panen dengan Regresi Linear")
st.pyplot(fig3)

# Narasi storytelling
st.subheader("📝 Narasi Data Storytelling")
st.markdown("""
- Visualisasi korelasi menunjukkan bagaimana fitur sensor utama berkaitan dengan hasil panen.
- Pairplot memberikan gambaran detail hubungan antar fitur, termasuk hubungan non-linear dan pola sebaran data.
- Plot scatter Rainfall dan Humidity yang diberi warna menurut Yield memperlihatkan pola kombinasi kedua sensor tersebut terhadap hasil panen.
- Model regresi linear sederhana dengan Rainfall dan Humidity sebagai fitur dapat memprediksi hasil panen dengan akurasi yang cukup baik, meski masih bisa ditingkatkan.
- Data ini bisa digunakan petani untuk pengambilan keputusan seperti optimasi irigasi dan pengelolaan kelembaban lahan guna meningkatkan hasil panen.
""")
