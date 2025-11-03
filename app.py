import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from rapidfuzz.distance import DamerauLevenshtein
from rapidfuzz import process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import os, unicodedata, re

# ---------------------------------------------
# Inisialisasi NLTK
# ---------------------------------------------
nltk.download('stopwords')
nltk.download('punkt')

# ---------------------------------------------
# Fungsi bantu preprocessing
# ---------------------------------------------
def case_folding(text):
    return str(text).lower()

def cleaning(text):
    text = str(text)
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def normalization(tokens, slang_dict):
    return [slang_dict.get(w, w) for w in tokens]

def filtering(tokens, stop_words):
    return [w for w in tokens if w not in stop_words]

def stemming(tokens, stemmer):
    return [stemmer.stem(w) for w in tokens]

def word_correction(tokens, vocab, threshold=0.85):
    corrected = []
    for word in tokens:
        if word in vocab:
            corrected.append(word)
        else:
            match = process.extractOne(
                word, vocab, scorer=DamerauLevenshtein.normalized_distance,
                score_cutoff=threshold
            )
            corrected.append(match[0] if match else word)
    return corrected

# ---------------------------------------------
# Load kamus dan slangword
# ---------------------------------------------
def load_kamus(file_path):
    kamus = set()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            kamus.add(line.strip().lower())
    return kamus

def load_slang(file_path):
    slang_dict = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if "=" in line:
                k, v = line.strip().split("=")
                slang_dict[k.strip()] = v.strip()
    return slang_dict

kamus_baku = load_kamus("kamus.txt")
slang_dict = load_slang("slangwords.txt")
factory = StemmerFactory()
stemmer = factory.create_stemmer()
stop_words = set(stopwords.words("indonesian"))

# ---------------------------------------------
# Fungsi navbar
# ---------------------------------------------
st.set_page_config(page_title="Analisis Sentimen SVM", layout="wide")
menu = st.sidebar.radio("📑 Menu", [
    "1️⃣ Upload Data",
    "2️⃣ Statistik Dataset",
    "3️⃣ Preprocessing",
    "4️⃣ Koreksi Kata (DLD)",
    "5️⃣ N-Gram TF-IDF + SVM"
])

st.title("📊 Analisis Sentimen Multi-Aspek Wisata dengan SVM")
st.markdown("Proyek Skripsi oleh **Cindy Permata Sari**")

# =============================================
# 1️⃣ Upload Data
# =============================================
if menu == "1️⃣ Upload Data":
    st.header("📂 Upload Dataset CSV")
    file = st.file_uploader("Unggah file CSV ulasan", type=["csv"])

    if file:
        df = pd.read_csv(file)
        st.session_state["df"] = df
        st.success(f"File berhasil diunggah! Jumlah data: {len(df)}")
        st.dataframe(df.head())
    else:
        st.info("Silakan unggah file CSV terlebih dahulu.")

# =============================================
# 2️⃣ Statistik Dataset
# =============================================
elif menu == "2️⃣ Statistik Dataset":
    st.header("📈 Statistik dan Distribusi Sentimen")
    if "df" not in st.session_state:
        st.warning("⚠️ Silakan upload file pada menu pertama.")
    else:
        df = st.session_state["df"]
        st.subheader("🔹 Ringkasan Kolom")
        st.write(df.describe(include='all'))
        st.subheader("🔹 Distribusi Aspek")
        aspects = [c for c in df.columns if c.lower() not in ["ulasan", "review", "text"]]
        fig, ax = plt.subplots()
        df[aspects].apply(pd.Series.value_counts).T.plot(kind="bar", ax=ax)
        plt.title("Distribusi Sentimen per Aspek")
        st.pyplot(fig)

# =============================================
# 3️⃣ Preprocessing
# =============================================
elif menu == "3️⃣ Preprocessing":
    st.header("🧹 Tahapan Preprocessing Teks")
    if "df" not in st.session_state:
        st.warning("⚠️ Silakan upload file terlebih dahulu.")
    else:
        df = st.session_state["df"].copy()
        ulasan_col = [c for c in df.columns if "ulas" in c.lower()][0]
        df["text"] = df[ulasan_col].astype(str)
        df["casefold"] = df["text"].apply(case_folding)
        df["clean"] = df["casefold"].apply(cleaning)
        df["token"] = df["clean"].apply(word_tokenize)
        df["normal"] = df["token"].apply(lambda x: normalization(x, slang_dict))
        df["filter"] = df["normal"].apply(lambda x: filtering(x, stop_words))
        df["stem"] = df["filter"].apply(lambda x: stemming(x, stemmer))
        st.session_state["preprocessed"] = df
        st.dataframe(df.head(5)[["text", "stem"]])
        st.success("✅ Preprocessing selesai!")

# =============================================
# 4️⃣ Koreksi Kata (DLD)
# =============================================
elif menu == "4️⃣ Koreksi Kata (DLD)":
    st.header("🔤 Koreksi Kata dengan Damerau-Levenshtein Distance")
    if "preprocessed" not in st.session_state:
        st.warning("⚠️ Jalankan preprocessing dulu.")
    else:
        df = st.session_state["preprocessed"].copy()
        df["corrected"] = df["stem"].apply(lambda x: word_correction(x, kamus_baku))
        st.session_state["corrected"] = df
        st.dataframe(df[["stem", "corrected"]].head(5))
        st.success("✅ Koreksi kata selesai!")

# =============================================
# 5️⃣ N-Gram TF-IDF + SVM
# =============================================
elif menu == "5️⃣ N-Gram TF-IDF + SVM":
    st.header("📘 Pelatihan Model SVM dengan N-Gram (1,2)")
    if "corrected" not in st.session_state:
        st.warning("⚠️ Jalankan koreksi kata dulu.")
    else:
        df = st.session_state["corrected"].copy()
        aspek = st.selectbox("Pilih Aspek Sentimen", [c for c in df.columns if c.lower() not in ["ulasan","text","casefold","clean","token","normal","filter","stem","corrected"]])
        df_aspek = df[df[aspek] != 0].copy()
        df_aspek["sentimen"] = df_aspek[aspek].map({1:"Positif",-1:"Negatif"})
        df_aspek["final_text"] = df_aspek["corrected"].apply(lambda x: " ".join(x))
        
        X = df_aspek["final_text"]
        y = df_aspek["sentimen"]
        vectorizer = TfidfVectorizer(ngram_range=(1,2))
        X_tfidf = vectorizer.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

        model = SVC(kernel='linear')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.subheader("📊 Hasil Evaluasi Model SVM")
        st.text(classification_report(y_test, y_pred))

        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, display_labels=model.classes_, cmap="Blues", ax=ax)
        st.pyplot(fig)
        st.success("✅ Model selesai dilatih!")

