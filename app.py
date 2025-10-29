import streamlit as st
import fitz  # PyMuPDF
import re, string, nltk, pickle, os
import pandas as pd
from nltk.corpus import stopwords

nltk.download("stopwords")

# ---------- Helper Functions ----------

def extract_text_from_pdf(path):
    """Extract text from a PDF file."""
    text = ""
    try:
        with fitz.open(path) as pdf:
            for page in pdf:
                text += page.get_text()
    except Exception as e:
        st.error(f"Error reading {path}: {e}")
    return text


def clean_text(text):
    """Lowercase, remove punctuation, digits, and stopwords."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\\d+", "", text)
    words = text.split()
    stop_words = set(stopwords.words("english"))
    return set([w for w in words if w not in stop_words])


def jaccard_similarity(s1, s2):
    """Compute Jaccard similarity."""
    if not s1 or not s2:
        return 0
    return len(s1.intersection(s2)) / len(s1.union(s2))


# ---------- Reference PDFs ----------

REFERENCE_PDFS = ["ref1.pdf", "ref2.pdf", "ref3.pdf", "ref4.pdf"]
PICKLE_FILE = "refs.pkl"

st.title("📄 PDF Plagiarism Checker (Jaccard Similarity)")

missing_files = [f for f in REFERENCE_PDFS if not os.path.exists(f)]

if missing_files:
    st.error("⚠️ The following reference PDF files are missing:")
    for f in missing_files:
        st.markdown(f"- ❌ **{f}**")
    st.stop()  # stop execution here
else:
    st.success("✅ All 4 reference PDFs found!")

    # ---------- Load or Create Pickle ----------
    if not os.path.exists(PICKLE_FILE):
        st.info("Processing reference PDFs...")
        ref_texts = {}
        for ref in REFERENCE_PDFS:
            txt = extract_text_from_pdf(ref)
            ref_texts[ref] = clean_text(txt)
        with open(PICKLE_FILE, "wb") as f:
            pickle.dump(ref_texts, f)
    else:
        with open(PICKLE_FILE, "rb") as f:
            ref_texts = pickle.load(f)
        st.success("✅ Loaded reference data from pickle!")


    # ---------- Streamlit UI ----------
    uploaded = st.file_uploader("📤 Upload a PDF file to compare", type=["pdf"])

    if uploaded:
        import tempfile
        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp.write(uploaded.read())
        user_text = extract_text_from_pdf(tmp.name)
        user_clean = clean_text(user_text)

        results = {r: round(jaccard_similarity(user_clean, t) * 100, 2)
                   for r, t in ref_texts.items()}

        df = pd.DataFrame(results.items(), columns=["Reference File", "Similarity (%)"])
        st.dataframe(df, use_container_width=True)

        if results:
            top = max(results, key=results.get)
            st.success(f"🔥 Highest similarity: {top} ({results[top]} %)")

        st.bar_chart(df.set_index("Reference File"))
