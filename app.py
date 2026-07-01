import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Grab Sentiment Analytics",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
#  GLOBAL STYLES
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Import font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Root & background ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #0d0f14;
    color: #e8eaf0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #13161e !important;
    border-right: 1px solid #1f2330;
}

[data-testid="stSidebar"] .stRadio label {
    color: #a0a8c0 !important;
    font-size: 0.88rem;
    padding: 6px 0;
    transition: color 0.2s;
}

[data-testid="stSidebar"] .stRadio label:hover {
    color: #00b14f !important;
}

/* ── Metric cards ── */
.metric-card {
    background: #13161e;
    border: 1px solid #1f2330;
    border-radius: 14px;
    padding: 20px 24px;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s, border-color 0.2s;
}

.metric-card:hover {
    transform: translateY(-3px);
    border-color: #00b14f55;
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: var(--accent, #00b14f);
    border-radius: 14px 14px 0 0;
}

.metric-icon {
    font-size: 1.8rem;
    margin-bottom: 8px;
}

.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: var(--accent, #00b14f);
    line-height: 1;
    margin-bottom: 4px;
}

.metric-label {
    font-size: 0.75rem;
    font-weight: 500;
    color: #6b7494;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ── Section headers ── */
.section-title {
    font-size: 1.35rem;
    font-weight: 600;
    color: #e8eaf0;
    margin: 32px 0 16px 0;
    display: flex;
    align-items: center;
    gap: 10px;
}

.section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #1f2330;
    margin-left: 12px;
}

/* ── Page title ── */
.page-header {
    padding: 28px 0 8px 0;
    border-bottom: 1px solid #1f2330;
    margin-bottom: 24px;
}

.page-header h1 {
    font-size: 1.9rem;
    font-weight: 700;
    color: #e8eaf0;
    margin: 0;
}

.page-header p {
    color: #6b7494;
    font-size: 0.9rem;
    margin: 6px 0 0 0;
}

/* ── Insight cards ── */
.insight-box {
    background: #13161e;
    border: 1px solid #1f2330;
    border-left: 4px solid #00b14f;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 12px;
    font-size: 0.88rem;
    color: #a0a8c0;
    line-height: 1.6;
}

.insight-box strong {
    color: #e8eaf0;
}

/* ── Prediction box ── */
.pred-positive {
    background: linear-gradient(135deg, #0d2818 0%, #13161e 100%);
    border: 1px solid #00b14f44;
    border-radius: 14px;
    padding: 28px;
    text-align: center;
}

.pred-negative {
    background: linear-gradient(135deg, #2b0d0d 0%, #13161e 100%);
    border: 1px solid #ff4d4d44;
    border-radius: 14px;
    padding: 28px;
    text-align: center;
}

.pred-emoji {
    font-size: 3.5rem;
    margin-bottom: 8px;
}

.pred-label-pos {
    font-size: 1.4rem;
    font-weight: 700;
    color: #00b14f;
}

.pred-label-neg {
    font-size: 1.4rem;
    font-weight: 700;
    color: #ff4d4d;
}

/* ── Tag badge ── */
.badge {
    display: inline-block;
    background: #00b14f22;
    color: #00b14f;
    border: 1px solid #00b14f44;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.75rem;
    font-weight: 600;
    margin: 2px;
}

.badge-red {
    background: #ff4d4d22;
    color: #ff4d4d;
    border-color: #ff4d4d44;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
}

/* ── Text area ── */
.stTextArea textarea {
    background: #13161e !important;
    border: 1px solid #1f2330 !important;
    color: #e8eaf0 !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
}

.stTextArea textarea:focus {
    border-color: #00b14f !important;
    box-shadow: 0 0 0 2px #00b14f22 !important;
}

/* ── Button ── */
.stButton > button {
    background: #00b14f !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 12px 32px !important;
    font-size: 0.9rem !important;
    transition: background 0.2s, transform 0.15s !important;
}

.stButton > button:hover {
    background: #009940 !important;
    transform: translateY(-2px) !important;
}

/* ── Selectbox ── */
.stSelectbox > div > div {
    background: #13161e !important;
    border-color: #1f2330 !important;
    color: #e8eaf0 !important;
    border-radius: 10px !important;
}

/* ── About cards ── */
.about-card {
    background: #13161e;
    border: 1px solid #1f2330;
    border-radius: 14px;
    padding: 24px;
    height: 100%;
}

.about-card h3 {
    color: #00b14f;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 0 0 12px 0;
}

.about-card p, .about-card li {
    color: #a0a8c0;
    font-size: 0.88rem;
    line-height: 1.7;
}

.about-card ul {
    padding-left: 16px;
    margin: 0;
}

/* ── Progress bar ── */
.stProgress > div > div > div {
    background: #00b14f !important;
    border-radius: 4px !important;
}

/* ── Matplotlib background ── */
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  MATPLOTLIB THEME
# ─────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#13161e",
    "axes.facecolor":    "#13161e",
    "axes.edgecolor":    "#1f2330",
    "axes.labelcolor":   "#a0a8c0",
    "xtick.color":       "#6b7494",
    "ytick.color":       "#6b7494",
    "text.color":        "#e8eaf0",
    "grid.color":        "#1f2330",
    "grid.alpha":        0.6,
})

# ─────────────────────────────────────────────
#  DATA & MODEL
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("../data/label_grab.csv")

@st.cache_resource
def load_model():
    model = joblib.load("../naive_bayes_model.pkl")
    tfidf = joblib.load("../tfidf.pkl")
    return model, tfidf

df = load_data()
model, tfidf = load_model()

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 20px 0 28px 0;">
        <div style="font-size:2.5rem">🚕</div>
        <div style="font-weight:700; font-size:1.05rem; color:#e8eaf0; margin-top:6px;">Grab Analytics</div>
        <div style="font-size:0.72rem; color:#6b7494; margin-top:2px;">Sentiment Intelligence Dashboard</div>
    </div>
    """, unsafe_allow_html=True)

    menu = st.radio(
        "Navigasi",
        ["🏠  Dashboard", "☁️  WordCloud", "🎯  Evaluasi Model", "🤖  Prediksi", "📄  Dataset", "📚  Tentang"],
        label_visibility="collapsed"
    )

    st.markdown("""
    <div style="margin-top:auto; padding: 20px 0 8px 0; border-top:1px solid #1f2330;">
        <div style="font-size:0.72rem; color:#6b7494; text-align:center;">
            Naive Bayes · TF-IDF · 30K ulasan
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  SHARED DATA
# ─────────────────────────────────────────────
total      = len(df)
positif    = int((df['sentiment'] == 1).sum())
negatif    = int((df['sentiment'] == 0).sum())
pct_pos    = positif / total * 100
pct_neg    = negatif / total * 100

CM = np.array([[1377, 208], [324, 4091]])

# ══════════════════════════════════════════════
#  PAGE: DASHBOARD
# ══════════════════════════════════════════════
if "Dashboard" in menu:

    st.markdown("""
    <div class="page-header">
        <h1>🏠 Dashboard Analisis Sentimen</h1>
        <p>Gambaran umum distribusi sentimen ulasan pengguna aplikasi Grab</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Metric cards ──
    c1, c2, c3, c4 = st.columns(4)

    cards = [
        (c1, "📊", f"{total:,}", "Total Ulasan", "#00b14f"),
        (c2, "😊", f"{positif:,}", "Sentimen Positif", "#00b14f"),
        (c3, "😞", f"{negatif:,}", "Sentimen Negatif", "#ff4d4d"),
        (c4, "🎯", "91.13%", "Akurasi Model", "#6c8eff"),
    ]

    for col, icon, value, label, color in cards:
        with col:
            st.markdown(f"""
            <div class="metric-card" style="--accent:{color}">
                <div class="metric-icon">{icon}</div>
                <div class="metric-value" style="color:{color}">{value}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Distribusi Sentimen</div>', unsafe_allow_html=True)

    col_chart, col_info = st.columns([3, 2], gap="large")

    with col_chart:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        # Bar chart
        ax = axes[0]
        bars = ax.bar(
            ["Positif", "Negatif"],
            [positif, negatif],
            color=["#00b14f", "#ff4d4d"],
            width=0.45,
            edgecolor="none",
            zorder=3
        )
        ax.set_ylabel("Jumlah Ulasan", fontsize=9)
        ax.set_title("Distribusi Kelas", fontsize=10, fontweight='600', pad=12)
        ax.yaxis.grid(True, zorder=0)
        ax.set_axisbelow(True)
        ax.spines[['top','right','left']].set_visible(False)
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 40,
                f"{int(bar.get_height()):,}",
                ha='center', va='bottom', fontsize=9, color='#e8eaf0', fontweight='600'
            )

        # Pie chart
        ax2 = axes[1]
        wedges, texts, autotexts = ax2.pie(
            [positif, negatif],
            labels=["Positif", "Negatif"],
            colors=["#00b14f", "#ff4d4d"],
            autopct="%1.1f%%",
            startangle=90,
            wedgeprops=dict(edgecolor="#0d0f14", linewidth=2),
            pctdistance=0.75
        )
        for at in autotexts:
            at.set_color("#e8eaf0")
            at.set_fontsize(9)
            at.set_fontweight('600')
        for t in texts:
            t.set_color("#a0a8c0")
            t.set_fontsize(9)
        ax2.set_title("Proporsi Kelas", fontsize=10, fontweight='600', pad=12)

        fig.tight_layout(pad=2)
        st.pyplot(fig)

    with col_info:
        st.markdown("""
        <div class="insight-box">
            <strong>📌 Komposisi Dataset</strong><br>
            Dataset berisi ulasan pengguna Grab dari Google Play Store yang sudah dilabeli secara manual ke dalam dua kelas sentimen.
        </div>
        """, unsafe_allow_html=True)

        pos_pct = f"{pct_pos:.1f}%"
        neg_pct = f"{pct_neg:.1f}%"

        st.markdown(f"""
        <div class="insight-box">
            <strong>😊 Positif — {pos_pct}</strong><br>
            Sebagian besar pengguna memberikan ulasan positif, menunjukkan tingkat kepuasan yang tinggi.
        </div>
        <div class="insight-box">
            <strong>😞 Negatif — {neg_pct}</strong><br>
            Ulasan negatif umumnya menyoroti masalah layanan driver, promo, dan antarmuka aplikasi.
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  PAGE: WORDCLOUD
# ══════════════════════════════════════════════
elif "WordCloud" in menu:

    st.markdown("""
    <div class="page-header">
        <h1>☁️ Word Cloud Visualisasi</h1>
        <p>Kata-kata yang paling sering muncul dalam ulasan positif dan negatif</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["😊  Ulasan Positif", "😞  Ulasan Negatif"])

    def make_wordcloud(text, colormap, bg="#13161e"):
        wc = WordCloud(
            width=1000,
            height=420,
            background_color=bg,
            colormap=colormap,
            max_words=120,
            prefer_horizontal=0.85,
            collocations=False,
            margin=8
        ).generate(text)
        return wc

    with tab1:
        positive_text = " ".join(df[df['sentiment'] == 1]['content'].dropna())
        wc = make_wordcloud(positive_text, "Greens")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        fig.patch.set_facecolor("#13161e")
        fig.tight_layout(pad=0)
        st.pyplot(fig)

        # Top words
        words = positive_text.lower().split()
        top_words = Counter(words).most_common(10)
        st.markdown('<div class="section-title">Top 10 Kata — Positif</div>', unsafe_allow_html=True)
        tw_df = pd.DataFrame(top_words, columns=["Kata", "Frekuensi"])

        fig2, ax2 = plt.subplots(figsize=(8, 3))
        ax2.barh(tw_df["Kata"][::-1], tw_df["Frekuensi"][::-1], color="#00b14f", edgecolor="none")
        ax2.set_xlabel("Frekuensi", fontsize=9)
        ax2.xaxis.grid(True)
        ax2.set_axisbelow(True)
        ax2.spines[['top','right','bottom']].set_visible(False)
        fig2.tight_layout()
        st.pyplot(fig2)

    with tab2:
        negative_text = " ".join(df[df['sentiment'] == 0]['content'].dropna())
        wc = make_wordcloud(negative_text, "Reds")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        fig.patch.set_facecolor("#13161e")
        fig.tight_layout(pad=0)
        st.pyplot(fig)

        words = negative_text.lower().split()
        top_words = Counter(words).most_common(10)
        st.markdown('<div class="section-title">Top 10 Kata — Negatif</div>', unsafe_allow_html=True)
        tw_df = pd.DataFrame(top_words, columns=["Kata", "Frekuensi"])

        fig2, ax2 = plt.subplots(figsize=(8, 3))
        ax2.barh(tw_df["Kata"][::-1], tw_df["Frekuensi"][::-1], color="#ff4d4d", edgecolor="none")
        ax2.set_xlabel("Frekuensi", fontsize=9)
        ax2.xaxis.grid(True)
        ax2.set_axisbelow(True)
        ax2.spines[['top','right','bottom']].set_visible(False)
        fig2.tight_layout()
        st.pyplot(fig2)

# ══════════════════════════════════════════════
#  PAGE: EVALUASI MODEL
# ══════════════════════════════════════════════
elif "Evaluasi" in menu:

    st.markdown("""
    <div class="page-header">
        <h1>🎯 Evaluasi Model</h1>
        <p>Performa model Naive Bayes pada data uji</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Metric row ──
    TP, FN = CM[1][1], CM[1][0]
    FP, TN = CM[0][1], CM[0][0]

    accuracy  = (TP + TN) / CM.sum()
    precision = TP / (TP + FP)
    recall    = TP / (TP + FN)
    f1        = 2 * precision * recall / (precision + recall)

    m1, m2, m3, m4 = st.columns(4)
    for col, label, val, color in [
        (m1, "Accuracy",  f"{accuracy*100:.2f}%",  "#6c8eff"),
        (m2, "Precision", f"{precision*100:.2f}%", "#00b14f"),
        (m3, "Recall",    f"{recall*100:.2f}%",    "#f5a623"),
        (m4, "F1-Score",  f"{f1*100:.2f}%",        "#c47aff"),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-card" style="--accent:{color}">
                <div class="metric-value" style="color:{color}">{val}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Confusion Matrix</div>', unsafe_allow_html=True)

    col_cm, col_report = st.columns([1, 1], gap="large")

    with col_cm:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            CM,
            annot=True,
            fmt='d',
            cmap='Greens',
            linewidths=1.5,
            linecolor="#0d0f14",
            cbar=False,
            ax=ax,
            annot_kws={"size": 14, "weight": "bold", "color": "#0d0f14"}
        )
        ax.set_xlabel("Predicted Label", fontsize=10, labelpad=8)
        ax.set_ylabel("Actual Label", fontsize=10, labelpad=8)
        ax.set_xticklabels(["Negatif", "Positif"], fontsize=9)
        ax.set_yticklabels(["Negatif", "Positif"], fontsize=9, rotation=0)
        fig.tight_layout()
        st.pyplot(fig)

    with col_report:
        st.markdown("""
        <div class="insight-box">
            <strong>True Positive (TP)</strong> — 4,091<br>
            Ulasan positif yang diprediksi benar sebagai positif.
        </div>
        <div class="insight-box">
            <strong>True Negative (TN)</strong> — 1,377<br>
            Ulasan negatif yang diprediksi benar sebagai negatif.
        </div>
        <div class="insight-box" style="border-left-color:#ff4d4d">
            <strong>False Positive (FP)</strong> — 208<br>
            Ulasan negatif yang keliru diprediksi sebagai positif.
        </div>
        <div class="insight-box" style="border-left-color:#ff4d4d">
            <strong>False Negative (FN)</strong> — 324<br>
            Ulasan positif yang keliru diprediksi sebagai negatif.
        </div>
        """, unsafe_allow_html=True)

    # Classification report per class
    st.markdown('<div class="section-title">Laporan Per Kelas</div>', unsafe_allow_html=True)

    report_data = {
        "Kelas":     ["Negatif (0)", "Positif (1)"],
        "Precision": [f"{TN/(TN+FN)*100:.1f}%", f"{precision*100:.1f}%"],
        "Recall":    [f"{TN/(TN+FP)*100:.1f}%",  f"{recall*100:.1f}%"],
        "F1-Score":  ["—", f"{f1*100:.1f}%"],
        "Support":   [TN+FP, TP+FN],
    }
    st.dataframe(pd.DataFrame(report_data), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════
#  PAGE: PREDIKSI
# ══════════════════════════════════════════════
elif "Prediksi" in menu:

    st.markdown("""
    <div class="page-header">
        <h1>🤖 Prediksi Sentimen</h1>
        <p>Masukkan ulasan untuk diklasifikasikan oleh model</p>
    </div>
    """, unsafe_allow_html=True)

    col_input, col_result = st.columns([1, 1], gap="large")

    with col_input:
        review = st.text_area(
            "Tulis ulasan Grab di sini…",
            placeholder="Contoh: Driver tepat waktu, aplikasinya juga mudah digunakan!",
            height=180
        )

        examples = [
            "Driver sangat ramah dan cepat sampai tujuan",
            "Aplikasi sering error dan promo sering tidak berlaku",
            "Harga terjangkau dan pelayanan memuaskan",
        ]

        st.markdown("<div style='font-size:0.8rem; color:#6b7494; margin:12px 0 6px 0;'>✨ Coba contoh ulasan:</div>", unsafe_allow_html=True)
        for ex in examples:
            if st.button(ex[:45] + "…" if len(ex) > 45 else ex, key=ex):
                review = ex

        predict_btn = st.button("🔍  Analisis Sentimen", use_container_width=True)

    with col_result:
        if predict_btn and review.strip():
            vector     = tfidf.transform([review])
            pred       = model.predict(vector)[0]
            prob       = model.predict_proba(vector)[0]
            confidence = max(prob) * 100

            if pred == 1:
                st.markdown(f"""
                <div class="pred-positive">
                    <div class="pred-emoji">😊</div>
                    <div class="pred-label-pos">Sentimen Positif</div>
                    <div style="color:#6b7494; font-size:0.82rem; margin-top:6px;">
                        Model yakin ulasan ini mengandung sentimen positif
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="pred-negative">
                    <div class="pred-emoji">😞</div>
                    <div class="pred-label-neg">Sentimen Negatif</div>
                    <div style="color:#6b7494; font-size:0.82rem; margin-top:6px;">
                        Model yakin ulasan ini mengandung sentimen negatif
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown(f"""
            <div style="margin-top:20px;">
                <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                    <span style="font-size:0.82rem; color:#a0a8c0; font-weight:500;">Confidence Score</span>
                    <span style="font-size:0.9rem; color:#e8eaf0; font-weight:700;">{confidence:.1f}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(confidence / 100)

            # Prob bars
            st.markdown("<div style='margin-top:16px;'>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5, 1.5))
            classes = ["Negatif", "Positif"]
            colors  = ["#ff4d4d", "#00b14f"]
            bars = ax.barh(classes, [prob[0]*100, prob[1]*100], color=colors, edgecolor="none", height=0.45)
            ax.set_xlim(0, 100)
            ax.set_xlabel("Probabilitas (%)", fontsize=8)
            ax.spines[['top','right','bottom']].set_visible(False)
            for bar, p in zip(bars, [prob[0]*100, prob[1]*100]):
                ax.text(p + 1, bar.get_y() + bar.get_height()/2, f"{p:.1f}%", va='center', fontsize=8, color='#e8eaf0')
            fig.tight_layout()
            st.pyplot(fig)

        elif predict_btn and not review.strip():
            st.warning("⚠️ Masukkan ulasan terlebih dahulu.")
        else:
            st.markdown("""
            <div style="text-align:center; padding:60px 20px; color:#6b7494;">
                <div style="font-size:3rem; margin-bottom:12px;">🤖</div>
                <div style="font-size:0.88rem;">Hasil prediksi akan muncul di sini</div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  PAGE: DATASET
# ══════════════════════════════════════════════
elif "Dataset" in menu:

    st.markdown("""
    <div class="page-header">
        <h1>📄 Dataset Explorer</h1>
        <p>Jelajahi dan filter dataset ulasan Grab</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        search_term = st.text_input("🔍 Cari kata dalam ulasan", placeholder="Ketik kata kunci…")
    with c2:
        pilihan = st.selectbox("Filter Sentimen", ["Semua", "Positif 😊", "Negatif 😞"])
    with c3:
        n_rows = st.selectbox("Tampilkan", [50, 100, 200, 500, "Semua"], index=1)

    data = df.copy()
    if "Positif" in pilihan:
        data = data[data['sentiment'] == 1]
    elif "Negatif" in pilihan:
        data = data[data['sentiment'] == 0]

    if search_term:
        data = data[data['content'].str.contains(search_term, case=False, na=False)]

    if n_rows != "Semua":
        data = data.head(int(n_rows))

    st.markdown(f"""
    <div style="font-size:0.82rem; color:#6b7494; margin-bottom:12px;">
        Menampilkan <strong style="color:#e8eaf0">{len(data):,}</strong> baris
    </div>
    """, unsafe_allow_html=True)

    st.dataframe(
        data.reset_index(drop=True),
        use_container_width=True,
        height=480
    )

    csv = data.to_csv(index=False).encode('utf-8')
    st.download_button(
        "⬇️  Download CSV",
        data=csv,
        file_name="grab_sentiment_filtered.csv",
        mime="text/csv"
    )

# ══════════════════════════════════════════════
#  PAGE: TENTANG
# ══════════════════════════════════════════════
elif "Tentang" in menu:

    st.markdown("""
    <div class="page-header">
        <h1>📚 Tentang Penelitian</h1>
        <p>Informasi metodologi dan detail teknis proyek ini</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown("""
        <div class="about-card">
            <h3>🎯 Tujuan Penelitian</h3>
            <p>
                Mengklasifikasikan sentimen ulasan pengguna aplikasi Grab dari Google Play Store
                ke dalam dua kelas: <strong style="color:#00b14f">positif</strong> dan
                <strong style="color:#ff4d4d">negatif</strong>, untuk membantu memahami
                persepsi pengguna secara otomatis menggunakan pendekatan machine learning.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="about-card">
            <h3>📊 Dataset</h3>
            <ul>
                <li>Sumber: Google Play Store</li>
                <li>Total: <strong style="color:#e8eaf0">30.000 ulasan</strong></li>
                <li>Label: Positif & Negatif</li>
                <li>Metode labeling: Semi-otomatis + verifikasi manual</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c3, c4 = st.columns(2, gap="large")

    with c3:
        st.markdown("""
        <div class="about-card">
            <h3>⚙️ Pipeline NLP</h3>
            <ul>
                <li>Preprocessing: case folding, stopword removal, stemming</li>
                <li>Ekstraksi fitur: <strong style="color:#e8eaf0">TF-IDF Vectorizer</strong></li>
                <li>Classifier: <strong style="color:#e8eaf0">Multinomial Naive Bayes</strong></li>
                <li>Validasi: train-test split 80:20</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown("""
        <div class="about-card">
            <h3>📈 Hasil Evaluasi</h3>
            <ul>
                <li>Accuracy: <strong style="color:#6c8eff">91.13%</strong></li>
                <li>Precision: <strong style="color:#00b14f">95.2%</strong></li>
                <li>Recall: <strong style="color:#f5a623">92.7%</strong></li>
                <li>F1-Score: <strong style="color:#c47aff">93.9%</strong></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background:#13161e; border:1px solid #1f2330; border-radius:14px; padding:20px 24px; text-align:center;">
        <span style="font-size:0.8rem; color:#6b7494;">
            Dibuat dengan ❤️ menggunakan Streamlit · Naive Bayes · TF-IDF
        </span>
    </div>
    """, unsafe_allow_html=True)