import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Konfigurasi halaman
st.set_page_config(page_title="Dashboard Smart Farming", layout="wide")
st.title("📊 Dashboard Data Sensor Pertanian - Smart Farming")

# Baca dataset langsung dari file lokal yang sudah diupload
DATA_PATH = "/mnt/data/Smart_Farming_Crop_Yield_2024.csv"
df = pd.read_csv(DATA_PATH)

# Tampilkan data awal
st.subheader("🔍 Tinjauan Data")
st.dataframe(df.head())

# Info dataset
st.markdown("""
**Fitur Penting:**
- Temperature
- Humidity
- Soil Moisture
- Light Intensity
- Rainfall
- Pressure
- Soil Temperature
- Target: Crop Yield
""")

# Visualisasi hubungan antar fitur
st.subheader("📈 Korelasi Antar Variabel")
numerical_df = df.select_dtypes(include='number')
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(numerical_df.corr(), annot=True, cmap="YlGnBu", ax=ax)
st.pyplot(fig)

# Visualisasi tren fitur utama
st.subheader("🌡️ Tren Suhu, Kelembaban, dan Curah Hujan")
available_columns = df.columns.tolist()
default_cols = [col for col in ["Temperature", "Humidity"] if col in available_columns]
cols = st.multiselect("Pilih fitur untuk dianalisis:", [col for col in ["Temperature", "Humidity", "Rainfall", "Soil Moisture", "Light Intensity"] if col in available_columns], default=default_cols)

if cols:
    fig2, ax2 = plt.subplots()
    df[cols].plot(ax=ax2)
    plt.xlabel("Index Waktu")
    plt.ylabel("Nilai Sensor")
    plt.title("Tren Sensor")
    st.pyplot(fig2)

# Analisis prediktif sederhana
st.subheader("📊 Hubungan Sensor dengan Hasil Panen")
sensor_options = [col for col in ["Temperature", "Humidity", "Soil Moisture", "Rainfall", "Light Intensity", "Pressure", "Soil Temperature"] if col in df.columns]
selected_x = st.selectbox("Pilih variabel sensor:", sensor_options)

if "CropYield" in df.columns:
    fig3, ax3 = plt.subplots()
    sns.scatterplot(x=df[selected_x], y=df["CropYield"], ax=ax3)
    ax3.set_xlabel(selected_x)
    ax3.set_ylabel("Crop Yield")
    ax3.set_title(f"Hubungan antara {selected_x} dan Crop Yield")
    st.pyplot(fig3)

    # Narasi Data Storytelling
    st.subheader("📝 Narasi Data Storytelling")
    st.markdown(f"""
    - Dari visualisasi heatmap, terlihat bahwa fitur **{selected_x}** memiliki hubungan tertentu terhadap hasil panen (**CropYield**).
    - Berdasarkan grafik scatter, kita bisa mengamati apakah semakin tinggi/lrendah nilai {selected_x}, maka hasil panen ikut berubah atau tidak.
    - Data sensor yang dikumpulkan melalui perangkat IoT ini bisa menjadi dasar untuk pengambilan keputusan seperti irigasi, pemupukan, atau prediksi hasil panen secara akurat.

    **Kesimpulan:**
    Dengan menganalisis data sensor secara berkala dan menghubungkannya dengan hasil panen, petani dapat meningkatkan efisiensi pertanian berbasis data.
    """)
else:
    st.warning("Kolom 'CropYield' tidak ditemukan dalam dataset. Pastikan dataset memiliki kolom tersebut untuk analisis lebih lanjut.")
