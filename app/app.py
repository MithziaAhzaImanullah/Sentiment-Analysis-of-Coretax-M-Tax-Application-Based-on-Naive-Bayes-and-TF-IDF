import streamlit as st
import pandas as pd
import joblib
import os
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from google_play_scraper import Sort  # Library Scraper Utama
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ==========================================
# 1. PENGATURAN HALAMAN & PATH
# ==========================================
st.set_page_config(page_title="SaaS NLP Sentiment Platform", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

# ==========================================
# 2. SIDEBAR CONFIGURATION (MENU KIRI)
# ==========================================
st.sidebar.title("📁 Upload Dataset Manual")
uploaded_file = st.sidebar.file_uploader("Atau upload file CSV lokal", type=["csv"])

st.sidebar.markdown("---")
st.sidebar.title("⚙️ Konfigurasi Model")
alpha_input = st.sidebar.slider("Alpha (Naive Bayes)", min_value=0.01, max_value=1.0, value=1.0, step=0.01)

st.sidebar.markdown("---")
st.sidebar.title("🗺️ Menu")
menu = st.sidebar.radio("Navigasi Halaman:", [
    "Overview", 
    "Live Scraping Data",
    "Auto-Label & Praproses", 
    "Visualisasi", 
    "Model & Evaluasi", 
    "Prediksi Teks"
])

# ==========================================
# 3. HELPER FUNCTIONS (PREPROCESSING)
# ==========================================
def proses_bersihkan_teks(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ==========================================
# 4. INITIALIZE SESSION STATE (RAM MANAGEMENT)
# ==========================================
if 'dataset' not in st.session_state:
    try:
        csv_path = os.path.join(PROJECT_DIR, 'dataset', 'reviews_mpajak_databersih.csv')
        st.session_state.dataset = pd.read_csv(csv_path)
    except FileNotFoundError:
        st.session_state.dataset = None

if uploaded_file is not None:
    if 'current_file' not in st.session_state or st.session_state.current_file != uploaded_file.name:
        st.session_state.dataset = pd.read_csv(uploaded_file)
        st.session_state.current_file = uploaded_file.name

df = st.session_state.dataset

# ==========================================
# 5. KONTEN MULTI-HALAMAN
# ==========================================

# --- HALAMAN 1: OVERVIEW ---
if menu == "Overview":
    st.title("📑 Overview Project")
    if df is None:
        st.warning("⚠️ Belum ada dataset yang dimuat. Gunakan menu 'Live Scraping Data' atau upload file CSV di sidebar.")
    else:
        col1, col2, col3 = st.columns(3)
        total_data = len(df)
        
        # SINKRONISASI 1 & 0: Menghitung berdasarkan angka biner
        pos_data = len(df[df['label'] == 1]) if 'label' in df.columns else 0
        neg_data = len(df[df['label'] == 0]) if 'label' in df.columns else 0
        
        col1.metric("Total Ulasan", f"{total_data} Data")
        col2.metric("Sentimen Positif (1)", f"{pos_data}", f"{(pos_data/total_data)*100:.1f}%" if total_data > 0 else "0%")
        col3.metric("Sentimen Negatif (0)", f"{neg_data}", f"{(neg_data/total_data)*100:.1f}%" if total_data > 0 else "0%", delta_color="inverse")
        
        st.markdown("### 📊 Preview Dataset Terkini")
        st.dataframe(df.head(10), use_container_width=True)


# --- HALAMAN 2: LIVE SCRAPING DATA ---
elif menu == "Live Scraping Data":
    st.title("🕷️ Live Google Play Store Scraper")
    st.write("Ambil data ulasan aplikasi apa pun secara langsung dari Google Play Store hanya menggunakan App ID.")
    
    with st.expander("💡 Bagaimana cara mencari App ID sebuah aplikasi?"):
        st.write("1. Buka Google Play Store di browser laptop Anda.")
        st.write("2. Cari aplikasi yang diinginkan (Misal: Gojek, Duolingo, atau M-Pajak).")
        st.write("3. Lihat URL/link di address bar atas, cari bagian `id=...`")
        st.info("Contoh URL: `.../details?id=com.gojek.app` -> Maka App ID-nya adalah: **`com.gojek.app`**")

    st.markdown("---")
    
    app_id = st.text_input("Masukkan App ID Aplikasi:", placeholder="Contoh: com.gojek.app")
    max_reviews = st.number_input("Jumlah ulasan maksimal yang ingin diambil:", min_value=10, max_value=5000, value=200, step=50)

    if st.button("🕸️ Mulai Ambil Data (Scraping)"):
        if app_id.strip():
            with st.spinner(f"Sedang mengambil {max_reviews} ulasan dari ID: {app_id}... Mohon tunggu sebentar..."):
                try:
                    # Menggunakan metode 'reviews' cepat dengan limit count ketat
                    from google_play_scraper import reviews 
                    
                    result, _ = reviews(
                        app_id,
                        lang='id',
                        country='id',
                        sort=Sort.NEWEST,
                        count=max_reviews
                    )
                    
                    if result:
                        raw_df = pd.DataFrame(result)
                        keep_columns = ['userName', 'score', 'at', 'content']
                        available_cols = [c for c in keep_columns if c in raw_df.columns]
                        filtered_df = raw_df[available_cols]
                        
                        st.session_state.dataset = filtered_df
                        st.success(f"🎉 Berhasil mengambil {len(filtered_df)} ulasan terbaru dari aplikasi '{app_id}'!")
                        st.write("⚠️ *Sekarang, silakan beralih ke menu **'Auto-Label & Praproses'** untuk mengolah data ini.*")
                        st.dataframe(filtered_df.head(10), use_container_width=True)
                        st.rerun()
                    else:
                        st.error("Ulasan gagal diambil. Pastikan ID Aplikasi benar.")
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat melakukan scraping: {str(e)}")
        else:
            st.warning("Silakan masukkan App ID terlebih dahulu!")


# --- HALAMAN 3: AUTO-LABEL & PRAPROSES ---
elif menu == "Auto-Label & Praproses":
    st.title("🛠️ Auto-Labeling & Text Preprocessing")
    
    if df is None:
        st.warning("Silakan ambil data lewat menu 'Live Scraping Data' atau upload file CSV terlebih dahulu!")
    else:
        punya_score = 'score' in df.columns
        punya_content = 'content' in df.columns
        punya_label = 'label' in df.columns
        punya_clean = 'clean_content' in df.columns
        
        st.markdown("### 🔍 Status Kelayakan Dokumen:")
        st.write(f"- Kolom Rating (`score`): {'✅ Tersedia' if punya_score else '❌ Tidak Ada'}")
        st.write(f"- Kolom Teks (`content`): {'✅ Tersedia' if punya_content else '❌ Tidak Ada'}")
        st.write(f"- Status Pelabelan (`label`): {'✅ Sudah Dilabeli' if punya_label else '⚠️ Belum Dilabeli'}")
        st.write(f"- Status Pembersihan (`clean_content`): {'✅ Sudah Bersih' if punya_clean else '⚠️ Belum Bersih'}")
        
        st.markdown("---")
        
        if not (punya_label and punya_clean):
            if hasattr(df, 'columns') and punya_score and punya_content:
                if st.button("🚀 Eksekusi Auto-Label & Pembersihan Sekarang"):
                    with st.spinner("Sedang memproses data... Mohon tunggu..."):
                        # SINKRONISASI: Menggunakan aturan >= 3 adalah 1 (Positif), sisanya 0 (Negatif)
                        df['label'] = df['score'].apply(lambda x: 1 if x >= 3 else 0)
                        
                        # Proses Pembersihan Kalimat
                        df['clean_content'] = df['content'].apply(proses_bersihkan_teks)
                        st.session_state.dataset = df
                        
                    st.success("🎉 Sukses! Dataset Anda kini telah otomatis ter-label biner (1/0) dan bersih!")
                    st.dataframe(df[['score', 'content', 'clean_content', 'label']].head(10), use_container_width=True)
                    st.rerun()
            else:
                st.error("Gagal memproses! Dataset wajib memiliki kolom bernama 'score' dan 'content'.")
        else:
            st.info("Dataset Anda sudah dalam kondisi prima. Silakan cek menu 'Visualisasi' atau 'Model & Evaluasi'!")


# --- HALAMAN 4: VISUALISASI ---
elif menu == "Visualisasi":
    st.title("📊 Visualisasi Data Analisis")
    if df is None or 'label' not in df.columns or 'clean_content' not in df.columns:
        st.warning("Dataset belum siap. Buka menu 'Auto-Label & Praproses' terlebih dahulu.")
    else:
        st.subheader("☁️ Word Cloud")
        w_col1, w_col2 = st.columns(2)
        
        with w_col1:
            st.success("🟢 Ulasan Berlabel Positif")
            # SINKRONISASI 1 & 0: Mencari index data yang bernilai 1
            teks_pos = " ".join(str(txt) for txt in df[df['label'] == 1]['clean_content'].dropna())
            if teks_pos.strip():
                wc_pos = WordCloud(width=800, height=400, background_color='black', colormap='Greens').generate(teks_pos)
                fig, ax = plt.subplots()
                ax.imshow(wc_pos, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
            else:
                st.write("Tidak ada kata kunci positif untuk dibentuk Word Cloud.")
                
        with w_col2:
            st.error("🔴 Ulasan Berlabel Negatif")
            # SINKRONISASI 1 & 0: Mencari index data yang bernilai 0
            teks_neg = " ".join(str(txt) for txt in df[df['label'] == 0]['clean_content'].dropna())
            if teks_neg.strip():
                wc_neg = WordCloud(width=800, height=400, background_color='black', colormap='Reds').generate(teks_neg)
                fig, ax = plt.subplots()
                ax.imshow(wc_neg, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
            else:
                st.write("Tidak ada kata kunci negatif untuk dibentuk Word Cloud.")


# --- HALAMAN 5: MODEL & EVALUASI ---
elif menu == "Model & Evaluasi":
    st.title("📉 Evaluasi Model & Performa")
    if df is None or 'label' not in df.columns or 'clean_content' not in df.columns:
        st.warning("Dataset belum siap. Silakan lakukan proses labeling & praproses terlebih dahulu.")
    else:
        st.write(f"Menampilkan hasil performa model dengan nilai **Alpha: {alpha_input}**")
        try:
            X = df['clean_content'].fillna('')
            y = df['label'].astype(int) # Memastikan dibaca sebagai integer biner
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            
            tfidf_sim = TfidfVectorizer(max_features=5000)
            X_train_tfidf = tfidf_sim.fit_transform(X_train)
            X_test_tfidf = tfidf_sim.transform(X_test)
            
            model_sim = MultinomialNB(alpha=alpha_input)
            model_sim.fit(X_train_tfidf, y_train)
            y_pred = model_sim.predict(X_test_tfidf)
            
            acc = accuracy_score(y_test, y_pred)
            st.metric(label="🎯 Akurasi Model Saat Ini", value=f"{acc * 100:.2f}%")
            
            st.markdown("### 📋 Classification Report")
            st.code(classification_report(y_test, y_pred))
        except Exception as e:
            st.error(f"Terjadi kesalahan saat melatih model: {str(e)}")


# --- HALAMAN 6: PREDIKSI TEKS ---
elif menu == "Prediksi Teks":
    st.title("🔮 Prediksi Sentimen Teks Baru")
    try:
        model_static = joblib.load(os.path.join(PROJECT_DIR, 'model', 'model_naive_bayes.pkl'))
        tfidf_static = joblib.load(os.path.join(PROJECT_DIR, 'model', 'tfidf_vectorizer.pkl'))
    except FileNotFoundError:
        st.error("File biner .pkl model bawaan belum ditemukan di folder model.")
        st.stop()
        
    user_input = st.text_area("Ketik kalimat ulasan baru di sini:")
    if st.button("Proses Sentimen"):
        if user_input.strip():
            v_input = tfidf_static.transform([user_input.lower()])
            pred = model_static.predict(v_input)[0]
            
            # SINKRONISASI 1 & 0: Mendukung pengecekan output biner maupun string lama
            if pred == 1 or str(pred) == "1" or str(pred).lower() == "positif":
                st.balloons()
                st.success("Hasil Analisis: **POSITIF** 👍")
            else:
                st.error("Hasil Analisis: **NEGATIF** 👎")