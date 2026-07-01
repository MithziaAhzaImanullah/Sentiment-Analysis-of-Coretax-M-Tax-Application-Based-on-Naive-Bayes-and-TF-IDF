import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from wordcloud import WordCloud
import re
import nltk
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import joblib
import io
from collections import Counter
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split

# -------------------------------------------------------------------
# 1. PAGE CONFIG & CUSTOM CSS
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Sentimen Coretax",
    layout="wide",
    page_icon="🧠",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Poppins:wght@600;700&display=swap');

    /* Global */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        min-height: 100vh;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 1px solid rgba(99, 102, 241, 0.3);
    }
    [data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }

    /* Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #06b6d4 100%);
        border-radius: 20px;
        padding: 40px 50px;
        margin-bottom: 30px;
        position: relative;
        overflow: hidden;
    }
    .hero-banner::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -10%;
        width: 400px;
        height: 400px;
        background: rgba(255,255,255,0.05);
        border-radius: 50%;
    }
    .hero-banner::after {
        content: '';
        position: absolute;
        bottom: -60%;
        right: 10%;
        width: 300px;
        height: 300px;
        background: rgba(255,255,255,0.05);
        border-radius: 50%;
    }
    .hero-title {
        font-family: 'Poppins', sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        color: white;
        margin: 0 0 10px 0;
        line-height: 1.2;
    }
    .hero-subtitle {
        font-size: 1rem;
        color: rgba(255,255,255,0.85);
        margin: 0;
        max-width: 600px;
        line-height: 1.6;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(255,255,255,0.2);
        border: 1px solid rgba(255,255,255,0.3);
        color: white;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 500;
        margin-bottom: 16px;
        letter-spacing: 0.5px;
    }

    /* Metric Cards */
    .metric-card {
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 16px;
        padding: 24px 28px;
        text-align: center;
        backdrop-filter: blur(10px);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        position: relative;
        overflow: hidden;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.3);
    }
    .metric-card .icon {
        font-size: 2rem;
        margin-bottom: 10px;
        display: block;
    }
    .metric-card .value {
        font-family: 'Poppins', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        color: white;
        line-height: 1;
        margin-bottom: 6px;
    }
    .metric-card .label {
        font-size: 0.85rem;
        color: rgba(255,255,255,0.65);
        font-weight: 500;
        letter-spacing: 0.3px;
    }
    .metric-card.total { border-top: 3px solid #6366f1; }
    .metric-card.negative { border-top: 3px solid #f43f5e; }
    .metric-card.positive { border-top: 3px solid #10b981; }
    .metric-card .percent-badge {
        display: inline-block;
        font-size: 0.75rem;
        padding: 2px 8px;
        border-radius: 10px;
        margin-top: 6px;
        font-weight: 600;
    }
    .badge-neg { background: rgba(244,63,94,0.2); color: #f43f5e; }
    .badge-pos { background: rgba(16,185,129,0.2); color: #10b981; }

    /* Section Headers */
    .section-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin: 32px 0 20px 0;
    }
    .section-header .line {
        flex: 1;
        height: 1px;
        background: rgba(255,255,255,0.1);
    }
    .section-title {
        font-family: 'Poppins', sans-serif;
        font-size: 1.15rem;
        font-weight: 600;
        color: #e2e8f0;
        white-space: nowrap;
    }

    /* Chart containers */
    .chart-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
    }
    .chart-title {
        font-size: 1rem;
        font-weight: 600;
        color: #e2e8f0;
        margin-bottom: 16px;
    }

    /* Pipeline Step Cards */
    .pipeline-step {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 14px;
        padding: 18px 22px;
        margin-bottom: 14px;
        position: relative;
        border-left: 4px solid #6366f1;
    }
    .pipeline-step .step-num {
        font-size: 0.72rem;
        font-weight: 700;
        color: #6366f1;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .pipeline-step .step-label {
        font-size: 0.95rem;
        font-weight: 600;
        color: #e2e8f0;
        margin-bottom: 8px;
    }
    .pipeline-step .step-result {
        font-size: 0.9rem;
        color: rgba(255,255,255,0.7);
        background: rgba(0,0,0,0.2);
        padding: 10px 14px;
        border-radius: 8px;
        font-family: monospace;
        word-break: break-word;
        line-height: 1.5;
    }
    .pipeline-step.final {
        border-left-color: #10b981;
    }
    .pipeline-step.final .step-num { color: #10b981; }
    .pipeline-step.final .step-result { color: #6ee7b7; }

    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 6px;
        gap: 4px;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: rgba(255,255,255,0.6) !important;
        font-weight: 500;
        padding: 8px 20px;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        color: white !important;
    }

    /* Text area & button */
    .stTextArea textarea {
        background: rgba(255,255,255,0.07) !important;
        border: 1px solid rgba(99, 102, 241, 0.4) !important;
        border-radius: 12px !important;
        color: white !important;
        font-family: 'Inter', sans-serif !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 10px 28px !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        transition: opacity 0.2s !important;
        width: 100%;
    }
    .stButton > button:hover {
        opacity: 0.88 !important;
    }

    /* Dataframe */
    .stDataFrame { border-radius: 12px; overflow: hidden; }

    /* Sidebar nav items */
    .nav-item {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 10px 16px;
        border-radius: 10px;
        margin-bottom: 6px;
        cursor: pointer;
        transition: background 0.2s;
        color: rgba(255,255,255,0.7);
        font-size: 0.9rem;
        font-weight: 500;
    }
    .nav-item:hover, .nav-item.active {
        background: rgba(99, 102, 241, 0.25);
        color: white;
    }

    /* Info boxes */
    .info-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(99,102,241,0.15);
        border: 1px solid rgba(99,102,241,0.3);
        color: #a5b4fc;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
        margin-bottom: 16px;
    }

    /* Prediksi result box */
    .result-box {
        border-radius: 16px;
        padding: 28px 32px;
        text-align: center;
        margin-top: 16px;
    }
    .result-box.neg {
        background: rgba(244,63,94,0.1);
        border: 2px solid rgba(244,63,94,0.4);
    }
    .result-box.pos {
        background: rgba(16,185,129,0.1);
        border: 2px solid rgba(16,185,129,0.4);
    }
    .result-emoji { font-size: 3rem; margin-bottom: 8px; display: block; }
    .result-label { font-family: 'Poppins', sans-serif; font-size: 1.6rem; font-weight: 700; }
    .result-label.neg { color: #f43f5e; }
    .result-label.pos { color: #10b981; }
    .result-prob { font-size: 0.9rem; color: rgba(255,255,255,0.55); margin-top: 6px; }

    /* About page card */
    .about-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 28px 32px;
        margin-bottom: 20px;
    }
    .about-card h3 {
        font-family: 'Poppins', sans-serif;
        font-size: 1.05rem;
        font-weight: 600;
        color: #a5b4fc;
        margin-bottom: 14px;
    }
    .about-card p, .about-card li {
        color: rgba(255,255,255,0.75);
        font-size: 0.92rem;
        line-height: 1.75;
    }
    .about-card ul { padding-left: 20px; }
    .tag-row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }
    .tag {
        background: rgba(99,102,241,0.2);
        border: 1px solid rgba(99,102,241,0.35);
        color: #a5b4fc;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------------------
# 2. SETUP & CACHING
# -------------------------------------------------------------------
@st.cache_resource
def download_nltk_data():
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

download_nltk_data()

@st.cache_resource
def load_stemmer():
    factory = StemmerFactory()
    return factory.create_stemmer()

stemmer = load_stemmer()

@st.cache_data
def load_data():
    df = pd.read_csv(r'D:\Kuliah\Semester 6\Kecerdasan Buatan\Project ML\dataset\Label_pajak.csv')
    return df

@st.cache_resource
def load_model_and_tfidf():
    model = joblib.load(r'D:\Kuliah\Semester 6\Kecerdasan Buatan\Project ML\model\naive_bayes_model.pkl')
    tfidf = joblib.load(r'D:\Kuliah\Semester 6\Kecerdasan Buatan\Project ML\model\tfidf.pkl')
    return model, tfidf

# -------------------------------------------------------------------
# SLANGWORDS
# -------------------------------------------------------------------
slangwords = {"@": "di", "abis": "habis", "wtb": "beli", "masi": "masih", "wts": "jual", "wtt": "tukar", "bgt": "banget", "maks": "maksimal", "plisss": "tolong", "bgttt": "banget", "indo": "indonesia", "bgtt": "banget", "ad": "ada", "rv": "redvelvet", "plis": "tolong", "pls": "tolong", "cr": "sumber", "cod": "bayar ditempat", "adlh": "adalah", "afaik": "as far as i know", "ahaha": "haha", "aj": "saja", "ajep-ajep": "dunia gemerlap", "ak": "saya", "akika": "aku", "akkoh": "aku", "akuwh": "aku", "alay": "norak", "alow": "halo", "ambilin": "ambilkan", "ancur": "hancur", "anjrit": "anjing", "anter": "antar", "ap2": "apa-apa", "apasih": "apa sih", "apes": "sial", "aps": "apa", "aq": "saya", "aquwh": "aku", "asbun": "asal bunyi", "aseekk": "asyik", "asekk": "asyik", "asem": "asam", "aspal": "asli tetapi palsu", "astul": "asal tulis", "ato": "atau", "au ah": "tidak mau tahu", "awak": "saya", "ay": "sayang", "ayank": "sayang", "b4": "sebelum", "bakalan": "akan", "bandes": "bantuan desa", "bangedh": "banget", "banpol": "bantuan polisi", "banpur": "bantuan tempur", "basbang": "basi", "bcanda": "bercanda", "bdg": "bandung", "begajulan": "nakal", "beliin": "belikan", "bencong": "banci", "bentar": "sebentar", "ber3": "bertiga", "beresin": "membereskan", "bete": "bosan", "beud": "banget", "bg": "abang", "bgmn": "bagaimana", "bijimane": "bagaimana", "bintal": "bimbingan mental", "bkl": "akan", "bknnya": "bukannya", "blegug": "bodoh", "blh": "boleh", "bln": "bulan", "blum": "belum", "bnci": "benci", "bnran": "yang benar", "bodor": "lucu", "bokap": "ayah", "boker": "buang air besar", "bokis": "bohong", "boljug": "boleh juga", "bonek": "bocah nekat", "boyeh": "boleh", "br": "baru", "brg": "bareng", "bro": "saudara laki-laki", "bru": "baru", "bs": "bisa", "bsen": "bosan", "bt": "buat", "btw": "ngomong-ngomong", "buaya": "tidak setia", "bubbu": "tidur", "bubu": "tidur", "bumil": "ibu hamil", "bw": "bawa", "bwt": "buat", "byk": "banyak", "byrin": "bayarkan", "cabal": "sabar", "cadas": "keren", "calo": "makelar", "can": "belum", "capcus": "pergi", "caper": "cari perhatian", "ce": "cewek", "cekal": "cegah tangkal", "cemen": "penakut", "cengengesan": "tertawa", "cepet": "cepat", "cew": "cewek", "chuyunk": "sayang", "cimeng": "ganja", "cipika cipiki": "cium pipi kanan cium pipi kiri", "ciyh": "sih", "ckepp": "cakep", "ckp": "cakep", "cmiiw": "correct me if i'm wrong", "cmpur": "campur", "cong": "banci", "conlok": "cinta lokasi", "cowwyy": "maaf", "cp": "siapa", "cpe": "capek", "cppe": "capek", "cucok": "cocok", "cuex": "cuek", "cumi": "Cuma miscall", "cups": "culun", "curanmor": "pencurian kendaraan bermotor", "curcol": "curahan hati colongan", "cwek": "cewek", "cyin": "cinta", "d": "di", "dah": "deh", "dapet": "dapat", "de": "adik", "dek": "adik", "demen": "suka", "deyh": "deh", "dgn": "dengan", "diancurin": "dihancurkan", "dimaafin": "dimaafkan", "dimintak": "diminta", "disono": "di sana", "dket": "dekat", "dkk": "dan kawan-kawan", "dll": "dan lain-lain", "dlu": "dulu", "dngn": "dengan", "dodol": "bodoh", "doku": "uang", "dongs": "dong", "dpt": "dapat", "dri": "dari", "drmn": "darimana", "drtd": "dari tadi", "dst": "dan seterusnya", "dtg": "datang", "duh": "aduh", "duren": "durian", "ed": "edisi", "egp": "emang gue pikirin", "eke": "aku", "elu": "kamu", "emangnya": "memangnya", "emng": "memang", "endak": "tidak", "enggak": "tidak", "envy": "iri", "ex": "mantan", "fax": "facsimile", "fifo": "first in first out", "folbek": "follow back", "fyi": "sebagai informasi", "gaada": "tidak ada uang", "gag": "tidak", "gaje": "tidak jelas", "gak papa": "tidak apa-apa", "gan": "juragan", "gaptek": "gagap teknologi", "gatek": "gagap teknologi", "gawe": "kerja", "gbs": "tidak bisa", "gebetan": "orang yang disuka", "geje": "tidak jelas", "gepeng": "gelandangan dan pengemis", "ghiy": "lagi", "gile": "gila", "gimana": "bagaimana", "gino": "gigi nongol", "githu": "gitu", "gj": "tidak jelas", "gmana": "bagaimana", "gn": "begini", "goblok": "bodoh", "golput": "golongan putih", "gowes": "mengayuh sepeda", "gpny": "tidak punya", "gr": "gede rasa", "gretongan": "gratisan", "gtau": "tidak tahu", "gua": "saya", "guoblok": "goblok", "gw": "saya", "ha": "tertawa", "haha": "tertawa", "hallow": "halo", "hankam": "pertahanan dan keamanan", "hehe": "he", "helo": "halo", "hey": "hai", "hlm": "halaman", "hny": "hanya", "hoax": "isu bohong", "hr": "hari", "hrus": "harus", "hubdar": "perhubungan darat", "huff": "mengeluh", "hum": "rumah", "humz": "rumah", "ilang": "hilang", "ilfil": "tidak suka", "imho": "in my humble opinion", "imoetz": "imut", "item": "hitam", "itungan": "hitungan", "iye": "iya", "ja": "saja", "jadiin": "jadi", "jaim": "jaga image", "jayus": "tidak lucu", "jdi": "jadi", "jem": "jam", "jga": "juga", "jgnkan": "jangankan", "jir": "anjing", "jln": "jalan", "jomblo": "tidak punya pacar", "jubir": "juru bicara", "jutek": "galak", "k": "ke", "kab": "kabupaten", "kabor": "kabur", "kacrut": "kacau", "kadiv": "kepala divisi", "kagak": "tidak", "kalo": "kalau", "kampret": "sialan", "kamtibmas": "keamanan dan ketertiban masyarakat", "kamuwh": "kamu", "kanwil": "kantor wilayah", "karna": "karena", "kasubbag": "kepala subbagian", "katrok": "kampungan", "kayanya": "kayaknya", "kbr": "kabar", "kdu": "harus", "kec": "kecamatan", "kejurnas": "kejuaraan nasional", "kekeuh": "keras kepala", "kel": "kelurahan", "kemaren": "kemarin", "kepengen": "mau", "kepingin": "mau", "kepsek": "kepala sekolah", "kesbang": "kesatuan bangsa", "kesra": "kesejahteraan rakyat", "ketrima": "diterima", "kgiatan": "kegiatan", "kibul": "bohong", "kimpoi": "kawin", "kl": "kalau", "klianz": "kalian", "kloter": "kelompok terbang", "klw": "kalau", "km": "kamu", "kmps": "kampus", "kmrn": "kemarin", "knal": "kenal", "knp": "kenapa", "kodya": "kota madya", "komdis": "komisi disiplin", "komsov": "komunis sovyet", "kongkow": "kumpul bareng teman-teman", "kopdar": "kopi darat", "korup": "korupsi", "kpn": "kapan", "krenz": "keren", "krm": "kirim", "kt": "kita", "ktmu": "ketemu", "ktr": "kantor", "kuper": "kurang pergaulan", "kw": "imitasi", "kyk": "seperti", "la": "lah", "lam": "salam", "lamp": "lampiran", "lanud": "landasan udara", "latgab": "latihan gabungan", "lebay": "berlebihan", "leh": "boleh", "lelet": "lambat", "lemot": "lambat", "lgi": "lagi", "lgsg": "langlangsung", "liat": "lihat", "litbang": "penelitian dan pengembangan", "lmyn": "lumayan", "lo": "kamu", "loe": "kamu", "lola": "lambat berfikir", "louph": "cinta", "low": "kalau", "lp": "lupa", "luber": "langsung, umum, bebas, dan rahasia", "luchuw": "lucu", "lum": "belum", "luthu": "lucu", "lwn": "lawan", "maacih": "terima kasih", "mabal": "bolos", "macem": "macam", "macih": "masih", "maem": "makan", "magabut": "makan gaji buta", "maho": "homo", "mak jang": "kaget", "maksain": "memaksa", "malem": "malam", "mam": "makan", "maneh": "kamu", "maniez": "manis", "mao": "mau", "masukin": "masukkan", "melu": "ikut", "mepet": "dekat sekali", "mgu": "minggu", "migas": "minyak dan gas bumi", "mikol": "minuman beralkohol", "miras": "minuman keras", "mlah": "malah", "mngkn": "mungkin", "mo": "mau", "mokad": "mati", "moso": "masa", "mpe": "sampai", "msk": "masuk", "mslh": "masalah", "mt": "makan teman", "mubes": "musyawarah besar", "mulu": "melulu", "mumpung": "selagi", "munas": "musyawarah nasional", "muntaber": "muntah dan berak", "musti": "mesti", "muupz": "maaf", "mw": "now watching", "n": "dan", "nanam": "menanam", "nanya": "bertanya", "napa": "kenapa", "napi": "narapidana", "napza": "narkotika, alkohol, psikotropika, dan zat adiktif", "narkoba": "narkotika, psikotropika, dan obat terlarang", "nasgor": "nasi goreng", "nda": "tidak", "ndiri": "sendiri", "ne": "ini", "nekolin": "neokolonialisme", "nembak": "menyatakan cinta", "ngabuburit": "menunggu berbuka puasa", "ngaku": "mengaku", "ngambil": "mengambil", "nganggur": "tidak punya pekerjaan", "ngapah": "kenapa", "ngaret": "terlambat", "ngasih": "memberikan", "ngebandel": "berbuat bandel", "ngegosip": "bergosip", "ngeklaim": "mengklaim", "ngeksis": "menjadi eksis", "ngeles": "berkilah", "ngelidur": "menggigau", "ngerampok": "merampok", "ngga": "tidak", "ngibul": "berbohong", "ngiler": "mau", "ngiri": "iri", "ngisiin": "mengisikan", "ngmng": "bicara", "ngomong": "bicara", "ngubek2": "mencari-cari", "ngurus": "mengurus", "nie": "ini", "nih": "ini", "niyh": "nih", "nmr": "nomor", "nntn": "nonton", "nobar": "nonton bareng", "np": "now playing", "ntar": "nanti", "ntn": "nonton", "numpuk": "bertumpuk", "nutupin": "menutupi", "nyari": "mencari", "nyekar": "menyekar", "nyicil": "mencicil", "nyoblos": "mencoblos", "nyokap": "ibu", "ogah": "tidak mau", "ol": "online", "ongkir": "ongkos kirim", "oot": "out of topic", "org2": "orang-orang", "ortu": "orang tua", "otda": "otonomi daerah", "otw": "on the way, sedang di jalan", "pacal": "pacar", "pake": "pakai", "pala": "kepala", "pansus": "panitia khusus", "parpol": "partai politik", "pasutri": "pasangan suami istri", "pd": "pada", "pede": "percaya diri", "pelatnas": "pemusatan latihan nasional", "pemda": "pemerintah daerah", "pemkot": "pemerintah kota", "pemred": "pemimpin redaksi", "penjas": "pendidikan jasmani", "perda": "peraturan daerah", "perhatiin": "perhatikan", "pesenan": "pesanan", "pgang": "pegang", "pi": "tapi", "pilkada": "pemilihan kepala daerah", "pisan": "sangat", "pk": "penjahat kelamin", "plg": "paling", "pmrnth": "pemerintah", "polantas": "polisi lalu lintas", "ponpes": "pondok pesantren", "pp": "pulang pergi", "prg": "pergi", "prnh": "pernah", "psen": "pesan", "pst": "pasti", "pswt": "pesawat", "pw": "posisi nyaman", "qmu": "kamu", "rakor": "rapat koordinasi", "ranmor": "kendaraan bermotor", "re": "reply", "ref": "referensi", "rehab": "rehabilitasi", "rempong": "sulit", "repp": "balas", "restik": "reserse narkotika", "rhs": "rahasia", "rmh": "rumah", "ru": "baru", "ruko": "rumah toko", "rusunawa": "rumah susun sewa", "ruz": "terus", "saia": "saya", "salting": "salah tingkah", "sampe": "sampai", "samsek": "sama sekali", "sapose": "siapa", "satpam": "satuan pengamanan", "sbb": "sebagai berikut", "sbh": "sebuah", "sbnrny": "sebenarnya", "scr": "secara", "sdgkn": "sedangkan", "sdkt": "sedikit", "se7": "setuju", "sebelas dua belas": "mirip", "sembako": "sembilan bahan pokok", "sempet": "sempat", "sendratari": "seni drama tari", "sgt": "sangat", "shg": "sehingga", "siech": "sih", "sikon": "situasi dan kondisi", "sinetron": "sinema elektronik", "siramin": "siramkan", "sj": "saja", "skalian": "sekalian", "sklh": "sekolah", "skt": "sakit", "slesai": "selesai", "sll": "selalu", "slma": "selama", "slsai": "selesai", "smpt": "sempat", "smw": "semua", "sndiri": "sendiri", "soljum": "sholat jumat", "songong": "sombong", "sory": "maaf", "sosek": "sosial-ekonomi", "sotoy": "sok tahu", "spa": "siapa", "sppa": "siapa", "spt": "seperti", "srtfkt": "sertifikat", "stiap": "setiap", "stlh": "setelah", "suk": "masuk", "sumpek": "sempit", "syg": "sayang", "t4": "tempat", "tajir": "kaya", "tau": "tahu", "taw": "tahu", "td": "tadi", "tdk": "tidak", "teh": "kakak perempuan", "telat": "terlambat", "telmi": "telat berpikir", "temen": "teman", "tengil": "menyebalkan", "tepar": "terkapar", "tggu": "tunggu", "tgu": "tunggu", "thankz": "terima kasih", "thn": "tahun", "tilang": "bukti pelanggaran", "tipiwan": "TvOne", "tks": "terima kasih", "tlp": "telepon", "tls": "tulis", "tmbah": "tambah", "tmen2": "teman-teman", "tmpah": "tumpah", "tmpt": "tempat", "tngu": "tunggu", "tnyta": "ternyata", "tokai": "tai", "toserba": "toko serba ada", "tpi": "tapi", "trdhulu": "terdahulu", "trima": "terima kasih", "trm": "terima", "trs": "terus", "trutama": "terutama", "ts": "penulis", "tst": "tahu sama tahu", "ttg": "tentang", "tuch": "tuh", "tuir": "tua", "tw": "tahu", "u": "kamu", "ud": "sudah", "udah": "sudah", "ujg": "ujung", "ul": "ulangan", "unyu": "lucu", "uplot": "unggah", "urang": "saya", "usah": "perlu", "utk": "untuk", "valas": "valuta asing", "w/": "dengan", "wadir": "wakil direktur", "wamil": "wajib militer", "warkop": "warung kopi", "warteg": "warung tegal", "wat": "buat", "wkt": "waktu", "wtf": "what the fuck", "xixixi": "tertawa", "ya": "iya", "yap": "iya", "yaudah": "ya sudah", "yawdah": "ya sudah", "yg": "yang", "yl": "yang lain", "yo": "iya", "yowes": "ya sudah", "yup": "iya", "7an": "tujuan", "ababil": "abg labil", "acc": "accord", "adlah": "adalah", "adoh": "aduh", "aha": "tertawa", "aing": "saya", "aja": "saja", "ajj": "saja", "aka": "dikenal juga sebagai", "akko": "aku", "akku": "aku", "akyu": "aku", "aljasa": "asal jadi saja", "ama": "sama", "ambl": "ambil", "anjir": "anjing", "ank": "anak", "ap": "apa", "apaan": "apa", "ape": "apa", "aplot": "unggah", "apva": "apa", "aqu": "aku", "asap": "sesegera mungkin", "aseek": "asyik", "asek": "asyik", "aseknya": "asyiknya", "asoy": "asyik", "astrojim": "astagfirullahaladzim", "ath": "kalau begitu", "atuh": "kalau begitu", "ava": "avatar", "aws": "awas", "ayang": "sayang", "ayok": "ayo", "bacot": "banyak bicara", "bales": "balas", "bangdes": "pembangunan desa", "bangkotan": "tua", "banpres": "bantuan presiden", "bansarkas": "bantuan sarana kesehatan", "bazis": "badan amal, zakat, infak, dan sedekah", "bcoz": "karena", "beb": "sayang", "bejibun": "banyak", "belom": "belum", "bener": "benar", "ber2": "berdua", "berdikari": "berdiri di atas kaki sendiri", "bet": "banget", "beti": "beda tipis", "beut": "banget", "bgd": "banget", "bgs": "bagus", "bhubu": "tidur", "bimbuluh": "bimbingan dan penyuluhan", "bisi": "kalau-kalau", "bkn": "bukan", "bl": "beli", "blg": "bilang", "blm": "belum", "bls": "balas", "bnchi": "benci", "bngung": "bingung", "bnyk": "banyak", "bohay": "badan aduhai", "bokep": "porno", "bokin": "pacar", "bole": "boleh", "bolot": "bodoh", "bonyok": "ayah ibu", "bpk": "bapak", "brb": "segera kembali", "brngkt": "berangkat", "brp": "berapa", "brur": "saudara laki-laki", "bsa": "bisa", "bsk": "besok", "bu_bu": "tidur", "bubarin": "bubarkan", "buber": "buka bersama", "bujubune": "luar biasa", "buser": "buru sergap", "bwhn": "bawahan", "byar": "bayar", "byr": "bayar", "c8": "chat", "cabut": "pergi", "caem": "cakep", "cama-cama": "sama-sama", "cangcut": "celana dalam", "cape": "capek", "caur": "jelek", "cekak": "tidak ada uang", "cekidot": "coba lihat", "cemplungin": "cemplungkan", "ceper": "pendek", "ceu": "kakak perempuan", "cewe": "cewek", "cibuk": "sibuk", "cin": "cinta", "ciye": "cie", "ckck": "ck", "clbk": "cinta lama bersemi kembali", "cmpr": "campur", "cnenk": "senang", "congor": "mulut", "cow": "cowok", "coz": "karena", "cpa": "siapa", "gokil": "gila", "gombal": "suka merayu", "gpl": "tidak pakai lama", "gpp": "tidak apa-apa", "gretong": "gratis", "gt": "begitu", "gtw": "tidak tahu", "gue": "saya", "guys": "teman-teman", "gws": "cepat sembuh", "haghaghag": "tertawa", "hakhak": "tertawa", "handak": "bahan peledak", "hansip": "pertahanan sipil", "hellow": "halo", "helow": "halo", "hi": "hai", "hlng": "hilang", "hnya": "hanya", "houm": "rumah", "hrs": "harus", "hubad": "hubungan angkatan darat", "hubla": "perhubungan laut", "huft": "mengeluh", "humas": "hubungan masyarakat", "idk": "saya tidak tahu", "ilfeel": "tidak suka", "imba": "jago sekali", "imoet": "imut", "info": "informasi", "itung": "hitung", "isengin": "bercanda", "iyala": "iya lah", "iyo": "iya", "jablay": "jarang dibelai", "jadul": "jaman dulu", "jancuk": "anjing", "jd": "jadi", "jdikan": "jadikan", "jg": "juga", "jgn": "jangan", "jijay": "jijik", "jkt": "jakarta", "jnj": "janji", "jth": "jatuh", "jurdil": "jujur adil", "jwb": "jawab", "ka": "kakak", "kabag": "kepala bagian", "kacian": "kasihan", "kadit": "kepala direktorat", "kaga": "tidak", "kaka": "kakak", "kamtib": "keamanan dan ketertiban", "kamuh": "kamu", "kamyu": "kamu", "kapt": "kapten", "kasat": "kepala satuan", "kasubbid": "kepala subbidang", "kau": "kamu", "kbar": "kabar", "kcian": "kasihan", "keburu": "terlanjur", "kedubes": "kedutaan besar", "kek": "seperti", "keknya": "kayaknya", "keliatan": "kelihatan", "keneh": "masih", "kepikiran": "terpikirkan", "kepo": "mau tahu urusan orang", "kere": "tidak punya uang", "kesian": "kasihan", "ketauan": "ketahuan", "keukeuh": "keras kepala", "khan": "kan", "kibus": "kaki busuk", "kk": "kakak", "klian": "kalian", "klo": "kalau", "kluarga": "keluarga", "klwrga": "keluarga", "kmari": "kemari", "kmpus": "kampus", "kn": "kan", "knl": "kenal", "knpa": "kenapa", "kog": "kok", "kompi": "komputer", "komtiong": "komunis Tiongkok", "konjen": "konsulat jenderal", "koq": "kok", "kpd": "kepada", "kptsan": "keputusan", "krik": "garing", "krn": "karena", "ktauan": "ketahuan", "ktny": "katanya", "kudu": "harus", "kuq": "kok", "ky": "seperti", "kykny": "kayanya", "laka": "kecelakaan", "lambreta": "lambat", "lansia": "lanjut usia", "lapas": "lembaga pemasyarakatan", "lbur": "libur", "lekong": "laki-laki", "lg": "lagi", "lgkp": "lengkap", "lht": "lihat", "linmas": "perlindungan masyarakat", "lmyan": "lumayan", "lngkp": "lengkap", "loch": "loh", "lol": "tertawa", "lom": "belum", "loupz": "cinta", "lowh": "kamu", "lu": "kamu", "luchu": "lucu", "luff": "cinta", "luph": "cinta", "lw": "kamu", "lwt": "lewat", "maaciw": "terima kasih", "mabes": "markas besar", "macem-macem": "macam-macam", "madesu": "masa depan suram", "maen": "main", "mahatma": "maju sehat bersama", "mak": "ibu", "makasih": "terima kasih", "malah": "bahkan", "malu2in": "memalukan", "mamz": "makan", "manies": "manis", "mantep": "mantap", "markus": "makelar kasus", "mba": "mbak", "mending": "lebih baik", "mgkn": "mungkin", "mhn": "mohon", "miker": "minuman keras", "milis": "mailing list", "mksd": "maksud", "mls": "malas", "mnt": "minta", "moge": "motor gede", "mokat": "mati", "mosok": "masa", "msh": "masih", "mskpn": "meskipun", "msng2": "masing-masing", "muahal": "mahal", "muker": "musyawarah kerja", "mumet": "pusing", "muna": "munafik", "munaslub": "musyawarah nasional luar biasa", "musda": "musyawarah daerah", "muup": "maaf", "muuv": "maaf", "nal": "kenal", "nangis": "menangis", "naon": "apa", "napol": "narapidana politik", "naq": "anak", "narsis": "bangga pada diri sendiri", "nax": "anak", "ndak": "tidak", "ndut": "gendut", "nekolim": "neokolonialisme", "nelfon": "menelepon", "ngabis2in": "menghabiskan", "ngakak": "tertawa", "ngambek": "marah", "ngampus": "pergi ke kampus", "ngantri": "mengantri", "ngapain": "sedang apa", "ngaruh": "berpengaruh", "ngawur": "berbicara sembarangan", "ngeceng": "kumpul bareng-bareng", "ngeh": "sadar", "ngekos": "tinggal di kos", "ngelamar": "melamar", "ngeliat": "melihat", "ngemeng": "bicara terus-terusan", "ngerti": "mengerti", "nggak": "tidak", "ngikut": "ikut", "nginep": "menginap", "ngisi": "mengisi", "ngmg": "bicara", "ngocol": "lucu", "ngomongin": "membicarakan", "ngumpul": "berkumpul", "ni": "ini", "nyasar": "tersesat", "nyariin": "mencari", "nyiapin": "mempersiapkan", "nyiram": "menyiram", "nyok": "ayo", "o/": "oleh", "ok": "ok", "priksa": "periksa", "pro": "profesional", "psn": "pesan", "psti": "pasti", "puanas": "panas", "qmo": "kamu", "qt": "kita", "rame": "ramai", "raskin": "rakyat miskin", "red": "redaksi", "reg": "register", "rejeki": "rezeki", "renstra": "rencana strategis", "reskrim": "reserse kriminal", "sni": "sini", "somse": "sombong sekali", "sorry": "maaf", "sosbud": "sosial-budaya", "sospol": "sosial-politik", "sowry": "maaf", "spd": "sepeda", "sprti": "seperti", "spy": "supaya", "stelah": "setelah", "subbag": "subbagian", "sumbangin": "sumbangkan", "sy": "saya", "syp": "siapa", "tabanas": "tabungan pembangunan nasional", "tar": "nanti", "taun": "tahun", "tawh": "tahu", "tdi": "tadi", "te2p": "tetap", "tekor": "rugi", "telkom": "telekomunikasi", "telp": "telepon", "temen2": "teman-teman", "tengok": "menjenguk", "terbitin": "terbitkan", "tgl": "tanggal", "thanks": "terima kasih", "thd": "terhadap", "thx": "terima kasih", "tipi": "TV", "tkg": "tukang", "tll": "terlalu", "tlpn": "telepon", "tman": "teman", "tmbh": "tambah", "tmn2": "teman-teman", "tmph": "tumpah", "tnda": "tanda", "tnh": "tanah", "togel": "toto gelap", "tp": "tapi", "tq": "terima kasih", "trgntg": "tergantung", "trims": "terima kasih", "cb": "coba", "y": "ya", "munfik": "munafik", "reklamuk": "reklamasi", "sma": "sama", "tren": "trend", "ngehe": "kesal", "mz": "mas", "analisise": "analisis", "sadaar": "sadar", "sept": "september", "nmenarik": "menarik", "zonk": "bodoh", "rights": "benar", "simiskin": "miskin", "ngumpet": "sembunyi", "hardcore": "keras", "akhirx": "akhirnya", "solve": "solusi", "watuk": "batuk", "ngebully": "intimidasi", "masy": "masyarakat", "still": "masih", "tauk": "tahu", "mbual": "bual", "tioghoa": "tionghoa", "ngentotin": "senggama", "kentot": "senggama", "faktakta": "fakta", "sohib": "teman", "rubahnn": "rubah", "trlalu": "terlalu", "nyela": "cela", "heters": "pembenci", "nyembah": "sembah", "most": "paling", "ikon": "lambang", "light": "terang", "pndukung": "pendukung", "setting": "atur", "seting": "akting", "next": "lanjut", "waspadalah": "waspada", "gantengsaya": "ganteng", "parte": "partai", "nyerang": "serang", "nipu": "tipu", "ktipu": "tipu", "jentelmen": "berani", "buangbuang": "buang", "tsangka": "tersangka", "kurng": "kurang", "ista": "nista", "less": "kurang", "koar": "teriak", "paranoid": "takut", "problem": "masalah", "tahi": "kotoran", "tirani": "tiran", "tilep": "tilap", "happy": "bahagia", "tak": "tidak", "penertiban": "tertib", "uasai": "kuasa", "mnolak": "tolak", "trending": "trend", "taik": "tahi", "wkwkkw": "tertawa", "ahokncc": "ahok", "istaa": "nista", "benarjujur": "jujur", "mgkin": "mungkin"}

# -------------------------------------------------------------------
# 3. PREPROCESSING FUNCTIONS
# -------------------------------------------------------------------
def cleaningText(text):
    text = str(text)
    text = re.sub(r'@[A-Za-z0-9_]+', ' ', text)
    text = re.sub(r'#[A-Za-z0-9_]+', ' ', text)
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'RT[\s]+', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def casefoldingText(text):
    return text.lower()

def fix_slangwords(text):
    words = text.split()
    fixed_words = [slangwords.get(word.lower(), word) for word in words]
    return " ".join(fixed_words)

def tokenizingText(text):
    return text.split()

def get_stopwords():
    listStopwords = set(stopwords.words('indonesian'))
    listStopwords.update(stopwords.words('english'))
    listStopwords.update(['iya','yaa','gak','nya','na','sih','ku','di','ga','ya','gaa','loh','kah','woi','woii','woy'])
    negation_words = {'tidak', 'tak', 'bukan', 'belum', 'jangan'}
    for word in negation_words:
        if word in listStopwords:
            listStopwords.remove(word)
    return listStopwords

stop_words = get_stopwords()

def filteringText(text_list):
    return [txt for txt in text_list if txt not in stop_words]

def toSentence(tokens):
    return " ".join(tokens)

def stemmingText(text):
    return stemmer.stem(text)

def preprocess_pipeline(text):
    text = cleaningText(text)
    text = casefoldingText(text)
    text = fix_slangwords(text)
    tokens = tokenizingText(text)
    filtered = filteringText(tokens)
    sentence = toSentence(filtered)
    return stemmingText(sentence)

# -------------------------------------------------------------------
# 4. MATPLOTLIB THEME
# -------------------------------------------------------------------
plt.rcParams.update({
    'figure.facecolor': 'none',
    'axes.facecolor': 'none',
    'axes.edgecolor': (1, 1, 1, 0.15),
    'axes.labelcolor': '#94a3b8',
    'xtick.color': '#94a3b8',
    'ytick.color': '#94a3b8',
    'text.color': '#e2e8f0',
    'grid.color': (1, 1, 1, 0.08),
    'grid.linestyle': '--',
})

# -------------------------------------------------------------------
# 5. SIDEBAR
# -------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 20px 0 10px 0;">
        <div style="font-size: 2.5rem;">🧠</div>
        <div style="font-family: 'Poppins', sans-serif; font-size: 1.1rem; font-weight: 700; color: white; margin-top: 8px;">Sentimen</div>
        <div style="font-size: 0.8rem; color: rgba(255,255,255,0.5);">Coretax NLP Dashboard</div>
    </div>
    <hr style="border-color: rgba(255,255,255,0.1); margin: 10px 0 20px 0;">
    """, unsafe_allow_html=True)

    st.markdown("**Navigasi**")
    page = st.radio(
        "",
        [
            "Overview & EDA",
            "WordCloud",
            "Dataset",
            "Uji Preprocessing",
            "Model & Evaluasi",
            "Prediksi Real-time",
            "Tentang Penelitian"
        ],
        label_visibility="collapsed"
    )

    st.markdown("""
    <hr style="border-color: rgba(255,255,255,0.1); margin: 20px 0;">
    <div style="font-size: 0.75rem; color: rgba(255,255,255,0.4); line-height: 1.7;">
        <b style="color:rgba(255,255,255,0.6);">Stack</b><br>
        🐍 Python · Streamlit<br>
        🔤 NLTK · Sastrawi<br>
        📊 Seaborn · Matplotlib<br>
        ☁️ WordCloud · Scikit-learn
    </div>
    """, unsafe_allow_html=True)

# -------------------------------------------------------------------
# 6. LOAD DATA
# -------------------------------------------------------------------
try:
    df = load_data()
    data_loaded = True
except Exception as e:
    data_loaded = False

try:
    nb_model, tfidf_vectorizer = load_model_and_tfidf()
    model_loaded = True
except Exception as e:
    model_loaded = False

# -------------------------------------------------------------------
# 7. HERO BANNER
# -------------------------------------------------------------------
st.markdown("""
<div class="hero-banner">
    <div class="hero-badge">🧪 NLP · Analisis Sentimen · EDA</div>
    <div class="hero-title">Dashboard Analisis Sentimen<br>Aplikasi Coretax</div>
    <p class="hero-subtitle">
        Eksplorasi data ulasan pengguna Coretax menggunakan pendekatan Natural Language Processing —
        mulai dari distribusi sentimen, visualisasi teks, hingga simulasi pipeline preprocessing.
    </p>
</div>
""", unsafe_allow_html=True)

if not data_loaded:
    st.error("⚠️ Dataset `Label_pajak.csv` tidak ditemukan. Periksa path file di fungsi `load_data()`.")
    st.stop()

# -------------------------------------------------------------------
# 8. METRIC CARDS
# -------------------------------------------------------------------
total_data = len(df)
neg_count  = len(df[df['sentiment'] == 0])
pos_count  = len(df[df['sentiment'] == 1])
neg_pct    = neg_count / total_data * 100
pos_pct    = pos_count / total_data * 100

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"""
    <div class="metric-card total">
        <span class="icon">📝</span>
        <div class="value">{total_data:,}</div>
        <div class="label">Total Ulasan</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="metric-card negative">
        <span class="icon">😠</span>
        <div class="value">{neg_count:,}</div>
        <div class="label">Sentimen Negatif</div>
        <span class="percent-badge badge-neg">{neg_pct:.1f}%</span>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="metric-card positive">
        <span class="icon">😊</span>
        <div class="value">{pos_count:,}</div>
        <div class="label">Sentimen Positif</div>
        <span class="percent-badge badge-pos">{pos_pct:.1f}%</span>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# -------------------------------------------------------------------
# 9. PAGES
# -------------------------------------------------------------------

# ── PAGE 1: Overview & EDA ─────────────────────────────────────────
if page == "Overview & EDA":

    st.markdown('<div class="section-header"><span class="section-title">📊 Distribusi & Statistik</span><div class="line"></div></div>', unsafe_allow_html=True)

    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.markdown('<div class="chart-card"><div class="chart-title">Distribusi Label Sentimen</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ['#f43f5e', '#10b981']
        bars = ax.bar(['Negatif (0)', 'Positif (1)'], [neg_count, pos_count], color=colors, width=0.5, edgecolor='none')
        for bar, val in zip(bars, [neg_count, pos_count]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    f'{val:,}', ha='center', va='bottom', fontsize=11, fontweight='bold', color='white')
        ax.set_ylabel('Jumlah Ulasan', fontsize=10)
        ax.set_ylim(0, max(neg_count, pos_count) * 1.18)
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.patch.set_alpha(0)
        st.pyplot(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="chart-card"><div class="chart-title">Proporsi Sentimen</div>', unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        wedge_colors = ['#f43f5e', '#10b981']
        wedges, texts, autotexts = ax2.pie(
            [neg_count, pos_count],
            labels=['Negatif', 'Positif'],
            colors=wedge_colors,
            autopct='%1.1f%%',
            startangle=140,
            wedgeprops={'edgecolor': (1, 1, 1, 0.1), 'linewidth': 2},
            pctdistance=0.75
        )
        for t in texts:    t.set_color('#94a3b8'); t.set_fontsize(11)
        for at in autotexts: at.set_color('white'); at.set_fontweight('bold'); at.set_fontsize(11)
        centre_circle = plt.Circle((0,0), 0.55, fc='none')
        ax2.add_artist(centre_circle)
        ax2.text(0, 0, f'{total_data:,}\nUlasan', ha='center', va='center',
                 fontsize=10, fontweight='bold', color='white')
        fig2.patch.set_alpha(0)
        st.pyplot(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Panjang Teks
    st.markdown('<div class="section-header"><span class="section-title">📏 Analisis Panjang Teks</span><div class="line"></div></div>', unsafe_allow_html=True)

    df['text_length'] = df['content'].astype(str).apply(len)
    df['word_count']  = df['content'].astype(str).apply(lambda x: len(x.split()))

    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown('<div class="chart-card"><div class="chart-title">Distribusi Panjang Karakter per Sentimen</div>', unsafe_allow_html=True)
        fig3, ax3 = plt.subplots(figsize=(6, 3.5))
        for label, color in [(0, '#f43f5e'), (1, '#10b981')]:
            data = df[df['sentiment'] == label]['text_length']
            ax3.hist(data, bins=40, alpha=0.6, color=color,
                     label=f"{'Negatif' if label==0 else 'Positif'}", edgecolor='none')
        ax3.legend(fontsize=9)
        ax3.set_xlabel('Jumlah Karakter')
        ax3.set_ylabel('Frekuensi')
        ax3.yaxis.grid(True); ax3.set_axisbelow(True)
        ax3.spines['top'].set_visible(False); ax3.spines['right'].set_visible(False)
        fig3.patch.set_alpha(0)
        st.pyplot(fig3, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_d:
        st.markdown('<div class="chart-card"><div class="chart-title">Rata-rata Statistik Teks</div>', unsafe_allow_html=True)
        stats = df.groupby('sentiment').agg(
            Rata_Karakter=('text_length', 'mean'),
            Rata_Kata=('word_count', 'mean'),
            Max_Karakter=('text_length', 'max')
        ).round(1).reset_index()
        stats['Sentimen'] = stats['sentiment'].map({0: '😠 Negatif', 1: '😊 Positif'})
        stats = stats[['Sentimen', 'Rata_Karakter', 'Rata_Kata', 'Max_Karakter']]
        stats.columns = ['Sentimen', 'Rata-rata Karakter', 'Rata-rata Kata', 'Maks Karakter']
        st.dataframe(stats, hide_index=True, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── TOP 10 KATA ────────────────────────────────────────────────
    st.markdown('<div class="section-header"><span class="section-title">🔤 Top 10 Kata Terbanyak</span><div class="line"></div></div>', unsafe_allow_html=True)
    st.markdown('<div class="info-pill">💡 Dihitung dari sampel 200 data setelah preprocessing</div>', unsafe_allow_html=True)

    @st.cache_data
    def compute_top_words(sentiment_label, n_sample=200):
        sample = df[df['sentiment'] == sentiment_label].head(n_sample).copy()
        all_words = []
        for text in sample['content']:
            processed = preprocess_pipeline(str(text))
            all_words.extend(processed.split())
        counter = Counter(all_words)
        top = counter.most_common(10)
        return pd.DataFrame(top, columns=['Kata', 'Frekuensi'])

    col_w1, col_w2 = st.columns(2)
    with col_w1:
        st.markdown('<div class="chart-card"><div class="chart-title">Top 10 Kata — Sentimen Negatif</div>', unsafe_allow_html=True)
        with st.spinner("Menghitung..."):
            top_neg = compute_top_words(0)
        fig_tw1, ax_tw1 = plt.subplots(figsize=(6, 4))
        bars_neg = ax_tw1.barh(top_neg['Kata'][::-1], top_neg['Frekuensi'][::-1],
                                color='#f43f5e', edgecolor='none', height=0.6)
        for bar, val in zip(bars_neg, top_neg['Frekuensi'][::-1]):
            ax_tw1.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                        str(val), va='center', fontsize=9, color='white')
        ax_tw1.set_xlabel('Frekuensi')
        ax_tw1.xaxis.grid(True); ax_tw1.set_axisbelow(True)
        ax_tw1.spines['top'].set_visible(False); ax_tw1.spines['right'].set_visible(False)
        fig_tw1.patch.set_alpha(0)
        st.pyplot(fig_tw1, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_w2:
        st.markdown('<div class="chart-card"><div class="chart-title">Top 10 Kata — Sentimen Positif</div>', unsafe_allow_html=True)
        with st.spinner("Menghitung..."):
            top_pos = compute_top_words(1)
        fig_tw2, ax_tw2 = plt.subplots(figsize=(6, 4))
        bars_pos = ax_tw2.barh(top_pos['Kata'][::-1], top_pos['Frekuensi'][::-1],
                                color='#10b981', edgecolor='none', height=0.6)
        for bar, val in zip(bars_pos, top_pos['Frekuensi'][::-1]):
            ax_tw2.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                        str(val), va='center', fontsize=9, color='white')
        ax_tw2.set_xlabel('Frekuensi')
        ax_tw2.xaxis.grid(True); ax_tw2.set_axisbelow(True)
        ax_tw2.spines['top'].set_visible(False); ax_tw2.spines['right'].set_visible(False)
        fig_tw2.patch.set_alpha(0)
        st.pyplot(fig_tw2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ── PAGE 2: WordCloud ──────────────────────────────────────────────
elif page == "WordCloud":

    st.markdown('<div class="section-header"><span class="section-title">☁️ Visualisasi Kata</span><div class="line"></div></div>', unsafe_allow_html=True)
    st.markdown('<div class="info-pill">💡 Diproses dari 50 sampel untuk kecepatan rendering</div>', unsafe_allow_html=True)

    tab_pos, tab_neg = st.tabs(["😊 Sentimen Positif", "😠 Sentimen Negatif"])

    def make_wordcloud(sentiment_label, colormap):
        sample = df[df['sentiment'] == sentiment_label].head(50).copy()
        sample['clean_text'] = sample['content'].apply(preprocess_pipeline) if 'clean_text' not in df.columns else sample['clean_text']
        text = ' '.join(sample['clean_text'] if 'clean_text' in sample.columns else sample['content'])
        return WordCloud(
            width=1400, height=600,
            background_color=None, mode='RGBA',
            colormap=colormap,
            max_words=120,
            prefer_horizontal=0.8
        ).generate(text)

    with tab_pos:
        with st.spinner("Membuat WordCloud positif..."):
            wc_pos = make_wordcloud(1, 'YlGn')
        fig_p, ax_p = plt.subplots(figsize=(14, 6))
        ax_p.imshow(wc_pos, interpolation='bilinear')
        ax_p.axis('off')
        fig_p.patch.set_alpha(0)
        st.pyplot(fig_p, use_container_width=True)

    with tab_neg:
        with st.spinner("Membuat WordCloud negatif..."):
            wc_neg = make_wordcloud(0, 'RdPu')
        fig_n, ax_n = plt.subplots(figsize=(14, 6))
        ax_n.imshow(wc_neg, interpolation='bilinear')
        ax_n.axis('off')
        fig_n.patch.set_alpha(0)
        st.pyplot(fig_n, use_container_width=True)

        # ── Download WordCloud Negatif sebagai PNG ─────────────────
        buf_wc = io.BytesIO()
        wc_neg_img = wc_neg.to_image()
        wc_neg_img.save(buf_wc, format='PNG')
        buf_wc.seek(0)
        st.download_button(
            label="⬇️ Download WordCloud Negatif (PNG)",
            data=buf_wc,
            file_name="wordcloud_negatif.png",
            mime="image/png"
        )


# ── PAGE 3: Dataset ────────────────────────────────────────────────
elif page == "Dataset":

    st.markdown('<div class="section-header"><span class="section-title">📋 Dataset Mentah</span><div class="line"></div></div>', unsafe_allow_html=True)

    col_f1, col_f2 = st.columns([2, 1])
    with col_f1:
        search_q = st.text_input("🔍 Cari ulasan...", placeholder="Ketik kata kunci...", label_visibility="collapsed")
    with col_f2:
        filter_sent = st.selectbox("Filter", ["Semua", "Positif (1)", "Negatif (0)"], label_visibility="collapsed")

    filtered_df = df.copy()
    if search_q:
        filtered_df = filtered_df[filtered_df['content'].astype(str).str.contains(search_q, case=False, na=False)]
    if filter_sent == "Positif (1)":
        filtered_df = filtered_df[filtered_df['sentiment'] == 1]
    elif filter_sent == "Negatif (0)":
        filtered_df = filtered_df[filtered_df['sentiment'] == 0]

    st.caption(f"Menampilkan {len(filtered_df):,} dari {total_data:,} ulasan")
    st.dataframe(
        filtered_df.head(200).style.map(
            lambda v: 'color: #10b981; font-weight: 600' if v == 1 else ('color: #f43f5e; font-weight: 600' if v == 0 else ''),
            subset=['sentiment']
        ),
        use_container_width=True,
        height=420
    )

    # ── Download CSV Hasil Filter ──────────────────────────────────
    st.markdown('<div class="section-header"><span class="section-title">⬇️ Download Data</span><div class="line"></div></div>', unsafe_allow_html=True)
    csv_bytes = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label=f"⬇️ Download {len(filtered_df):,} Ulasan Hasil Filter (.csv)",
        data=csv_bytes,
        file_name="data_sentimen_filtered.csv",
        mime="text/csv"
    )


# ── PAGE 4: Uji Preprocessing ──────────────────────────────────────
elif page == "Uji Preprocessing":

    st.markdown('<div class="section-header"><span class="section-title">🔬 Simulasi Pipeline NLP</span><div class="line"></div></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-pill">⚙️ Cleaning → Casefolding → Slangword Fix → Tokenizing → Stopword Removal → Stemming</div>
    """, unsafe_allow_html=True)

    user_input = st.text_area(
        "Masukkan teks ulasan:",
        "aplikasi ribet rencana biar simpel tapi priteksinya seperti aplikasi menyimpan uang milyaran mau masuk lupa email nggak bisa diperbaiki",
        height=110,
        label_visibility="collapsed",
        placeholder="Tulis ulasan Coretax di sini..."
    )

    if st.button("⚡ Proses Teks Sekarang"):
        if user_input.strip():
            with st.spinner("Memproses..."):
                cleaned     = cleaningText(user_input)
                cased       = casefoldingText(cleaned)
                slang_fixed = fix_slangwords(cased)
                tokenized   = tokenizingText(slang_fixed)
                filtered    = filteringText(tokenized)
                joined      = toSentence(filtered)
                stemmed     = stemmingText(joined)

            steps = [
                ("LANGKAH 1", "🧹 Cleaning Teks", "Hapus mention, hashtag, URL, angka & karakter khusus", cleaned, False),
                ("LANGKAH 2", "🔡 Case Folding", "Ubah semua teks menjadi huruf kecil", cased, False),
                ("LANGKAH 3", "🔄 Normalisasi Slangword", "Ganti kata tidak baku dengan padanan formalnya", slang_fixed, False),
                ("LANGKAH 4", "✂️ Tokenizing & Stopword Removal", "Pecah teks & hapus kata tidak bermakna", joined, False),
                ("LANGKAH 5", "🌱 Stemming (Sastrawi)", "Kembalikan kata ke bentuk dasar", stemmed, True),
            ]

            for step_num, step_label, step_desc, result, is_final in steps:
                css_class = "pipeline-step final" if is_final else "pipeline-step"
                st.markdown(f"""
                <div class="{css_class}">
                    <div class="step-num">{step_num} · {step_desc}</div>
                    <div class="step-label">{step_label}</div>
                    <div class="step-result">{result}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("Silakan masukkan teks terlebih dahulu.")


# ── PAGE 5: Model & Evaluasi ───────────────────────────────────────
elif page == "Model & Evaluasi":

    st.markdown('<div class="section-header"><span class="section-title">📉 Evaluasi Model Naive Bayes</span><div class="line"></div></div>', unsafe_allow_html=True)

    if not model_loaded:
        st.warning("⚠️ File model (`naive_bayes_model.pkl` / `tfidf.pkl`) tidak ditemukan.")
    else:
        st.markdown('<div class="info-pill">🔍 Evaluasi menggunakan data uji (20% split stratified dari dataset lengkap)</div>', unsafe_allow_html=True)

        @st.cache_data
        def run_evaluation():
            X = df['content'].astype(str).apply(preprocess_pipeline)
            y = df['sentiment'].astype(int)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            X_test_tfidf = tfidf_vectorizer.transform(X_test)
            y_pred = nb_model.predict(X_test_tfidf)
            acc = accuracy_score(y_test, y_pred)
            f1_neg = f1_score(y_test, y_pred, pos_label=0)
            f1_pos = f1_score(y_test, y_pred, pos_label=1)
            f1_mac = f1_score(y_test, y_pred, average='macro')
            cm = confusion_matrix(y_test, y_pred)
            # Gunakan output_dict=True agar parsing tidak bergantung pada format string
            report_dict = classification_report(
                y_test, y_pred,
                target_names=['Negatif (0)', 'Positif (1)'],
                output_dict=True
            )
            return acc, f1_neg, f1_pos, f1_mac, cm, report_dict, len(y_test)

        with st.spinner("Mengevaluasi model..."):
            acc, f1_neg, f1_pos, f1_mac, cm, report_dict, n_test = run_evaluation()

        # Metric cards
        m1, m2, m3, m4 = st.columns(4)
        for col, icon, val, lbl, color in [
            (m1, "🎯", f"{acc*100:.2f}%", "Akurasi", "#6366f1"),
            (m2, "📊", f"{f1_neg*100:.2f}%", "F1 Negatif", "#f43f5e"),
            (m3, "📊", f"{f1_pos*100:.2f}%", "F1 Positif", "#10b981"),
            (m4, "⚖️", f"{f1_mac*100:.2f}%", "F1 Macro Avg", "#06b6d4"),
        ]:
            with col:
                st.markdown(f"""
                <div class="metric-card" style="border-top: 3px solid {color};">
                    <span class="icon">{icon}</span>
                    <div class="value" style="font-size:1.8rem;">{val}</div>
                    <div class="label">{lbl}</div>
                    <div class="label" style="font-size:0.72rem; margin-top:4px;">n_test = {n_test:,}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Confusion Matrix + Classification Report
        col_cm, col_cr = st.columns([1, 1])

        with col_cm:
            st.markdown('<div class="chart-card"><div class="chart-title">Confusion Matrix</div>', unsafe_allow_html=True)
            fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
            sns.heatmap(
                cm, annot=True, fmt='d', cmap='RdPu',
                xticklabels=['Negatif', 'Positif'],
                yticklabels=['Negatif', 'Positif'],
                linewidths=0.5, linecolor=(1, 1, 1, 0.1),
                ax=ax_cm, cbar=True,
                annot_kws={"size": 14, "weight": "bold"}
            )
            ax_cm.set_xlabel('Prediksi', fontsize=10)
            ax_cm.set_ylabel('Aktual', fontsize=10)
            ax_cm.tick_params(colors='#94a3b8')
            fig_cm.patch.set_alpha(0)
            ax_cm.set_facecolor('none')
            st.pyplot(fig_cm, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_cr:
            st.markdown('<div class="chart-card"><div class="chart-title">Classification Report</div>', unsafe_allow_html=True)
            # Bangun DataFrame langsung dari dict — tidak ada parsing string yang bisa salah
            label_map = {
                'Negatif (0)': '😠 Negatif (0)',
                'Positif (1)': '😊 Positif (1)',
                'macro avg':   '⚖️ Macro Avg',
                'weighted avg': '📊 Weighted Avg',
            }
            rows = []
            for key in ['Negatif (0)', 'Positif (1)', 'macro avg', 'weighted avg']:
                if key in report_dict:
                    d = report_dict[key]
                    rows.append({
                        'Kelas':     label_map.get(key, key),
                        'Precision': round(d['precision'], 3),
                        'Recall':    round(d['recall'], 3),
                        'F1-Score':  round(d['f1-score'], 3),
                        'Support':   int(d['support']),
                    })
            df_report = pd.DataFrame(rows)
            st.dataframe(df_report, hide_index=True, use_container_width=True, height=220)
            st.markdown('</div>', unsafe_allow_html=True)


# ── PAGE 6: Prediksi Real-time ─────────────────────────────────────
elif page == "Prediksi Real-time":

    st.markdown('<div class="section-header"><span class="section-title">⚡ Prediksi Sentimen Real-time</span><div class="line"></div></div>', unsafe_allow_html=True)

    if not model_loaded:
        st.warning("⚠️ Model tidak dapat dimuat. Pastikan file `naive_bayes_model.pkl` dan `tfidf.pkl` tersedia.")
    else:
        st.markdown('<div class="info-pill">🤖 Powered by Naive Bayes + TF-IDF · Pipeline NLP Lengkap</div>', unsafe_allow_html=True)

        rt_input = st.text_area(
            "Masukkan ulasan untuk diprediksi:",
            placeholder="Contoh: aplikasi ini sangat lambat dan sering error...",
            height=130,
            label_visibility="collapsed"
        )

        if st.button("🔮 Prediksi Sentimen"):
            if rt_input.strip():
                with st.spinner("Menganalisis sentimen..."):
                    preprocessed = preprocess_pipeline(rt_input)
                    X_vec = tfidf_vectorizer.transform([preprocessed])
                    pred = nb_model.predict(X_vec)[0]
                    prob = nb_model.predict_proba(X_vec)[0]
                    conf_neg = prob[0] * 100
                    conf_pos = prob[1] * 100

                if pred == 0:
                    st.markdown(f"""
                    <div class="result-box neg">
                        <span class="result-emoji">😠</span>
                        <div class="result-label neg">Sentimen NEGATIF</div>
                        <div class="result-prob">Keyakinan Negatif: <b>{conf_neg:.1f}%</b> &nbsp;|&nbsp; Keyakinan Positif: <b>{conf_pos:.1f}%</b></div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-box pos">
                        <span class="result-emoji">😊</span>
                        <div class="result-label pos">Sentimen POSITIF</div>
                        <div class="result-prob">Keyakinan Positif: <b>{conf_pos:.1f}%</b> &nbsp;|&nbsp; Keyakinan Negatif: <b>{conf_neg:.1f}%</b></div>
                    </div>
                    """, unsafe_allow_html=True)

                # Tampilkan hasil preprocessing juga
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="chart-card"><div class="chart-title">🔬 Hasil Preprocessing</div>', unsafe_allow_html=True)
                st.markdown(f"""
                <div class="pipeline-step final">
                    <div class="step-num">OUTPUT PREPROCESSING</div>
                    <div class="step-label">Teks setelah pipeline NLP</div>
                    <div class="step-result">{preprocessed if preprocessed.strip() else '(teks kosong setelah preprocessing)'}</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning("Silakan masukkan teks terlebih dahulu.")

        # Contoh teks cepat
        st.markdown('<div class="section-header"><span class="section-title">💡 Coba Contoh Teks</span><div class="line"></div></div>', unsafe_allow_html=True)
        st.markdown('<div class="info-pill">Klik salah satu contoh di bawah lalu tekan Prediksi</div>', unsafe_allow_html=True)

        examples = [
            ("😠 Contoh Negatif 1", "Aplikasi ini sangat lambat dan sering error, tidak bisa login sama sekali sangat mengecewakan"),
            ("😠 Contoh Negatif 2", "Ribet banget prosesnya, buang-buang waktu, sistem pajak online malah bikin bingung"),
            ("😊 Contoh Positif 1", "Memudahkan pelaporan pajak, tampilan bersih dan mudah dipahami, sangat membantu"),
            ("😊 Contoh Positif 2", "Alhamdulillah sekarang bayar pajak bisa dari rumah, aplikasi ini keren dan responsif"),
        ]
        cols_ex = st.columns(2)
        for i, (lbl, txt) in enumerate(examples):
            with cols_ex[i % 2]:
                st.markdown(f"""
                <div class="pipeline-step" style="cursor:default;">
                    <div class="step-num">{lbl}</div>
                    <div class="step-result">{txt}</div>
                </div>
                """, unsafe_allow_html=True)


# ── PAGE 7: Tentang Penelitian ─────────────────────────────────────
elif page == "Tentang Penelitian":

    st.markdown('<div class="section-header"><span class="section-title">📖 Tentang Penelitian</span><div class="line"></div></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="about-card">
        <h3>🎯 Latar Belakang & Tujuan</h3>
        <p>
            Penelitian ini bertujuan untuk menganalisis sentimen publik terhadap aplikasi <b>Coretax</b> —
            sistem inti administrasi perpajakan yang diluncurkan oleh Direktorat Jenderal Pajak (DJP) Indonesia.
            Data ulasan dikumpulkan dari platform publik dan dianalisis menggunakan pendekatan
            <i>Natural Language Processing (NLP)</i> untuk memahami persepsi pengguna.
        </p>
    </div>

    <div class="about-card">
        <h3>📊 Dataset</h3>
        <p>
            Dataset terdiri dari ulasan pengguna Coretax yang telah diberi label sentimen secara manual:
        </p>
        <ul>
            <li><b>Total data:</b> 7.317 ulasan</li>
            <li><b>Label 0 (Negatif):</b> 5.788 ulasan (79.1%)</li>
            <li><b>Label 1 (Positif):</b> 1.529 ulasan (20.9%)</li>
            <li><b>Kolom:</b> <code>content</code> (teks ulasan), <code>sentiment</code> (label 0/1)</li>
        </ul>
    </div>

    <div class="about-card">
        <h3>⚙️ Pipeline NLP</h3>
        <p>Setiap teks ulasan diproses melalui pipeline berikut sebelum dimasukkan ke model:</p>
        <ul>
            <li><b>Cleaning:</b> Hapus mention, hashtag, URL, angka, dan karakter khusus</li>
            <li><b>Case Folding:</b> Ubah semua teks ke huruf kecil</li>
            <li><b>Normalisasi Slangword:</b> Kamus 600+ kata tidak baku → bentuk formal</li>
            <li><b>Tokenizing:</b> Pisahkan kalimat menjadi token kata</li>
            <li><b>Stopword Removal:</b> Hapus kata umum bahasa Indonesia & Inggris</li>
            <li><b>Stemming:</b> Bentuk dasar kata menggunakan library <b>Sastrawi</b></li>
        </ul>
    </div>

    <div class="about-card">
        <h3>🤖 Model Machine Learning</h3>
        <p>
            Model yang digunakan adalah <b>Multinomial Naive Bayes</b> yang dikombinasikan dengan
            <b>TF-IDF Vectorizer</b> (max_features=5.000) untuk representasi fitur teks.
            Model dilatih menggunakan 80% data dan dievaluasi pada 20% data uji.
        </p>
        <div class="tag-row">
            <span class="tag">Multinomial Naive Bayes</span>
            <span class="tag">TF-IDF Vectorizer</span>
            <span class="tag">Scikit-learn</span>
            <span class="tag">Train/Test Split 80:20</span>
            <span class="tag">Stratified Sampling</span>
        </div>
    </div>

    <div class="about-card">
        <h3>🛠️ Teknologi yang Digunakan</h3>
        <div class="tag-row">
            <span class="tag">🐍 Python 3.x</span>
            <span class="tag">📊 Streamlit</span>
            <span class="tag">🔤 NLTK</span>
            <span class="tag">🌱 PySastrawi</span>
            <span class="tag">📉 Scikit-learn</span>
            <span class="tag">🐼 Pandas</span>
            <span class="tag">📈 Matplotlib · Seaborn</span>
            <span class="tag">☁️ WordCloud</span>
            <span class="tag">💾 Joblib</span>
        </div>
    </div>
    """, unsafe_allow_html=True)