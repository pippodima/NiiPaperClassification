import os
import re
import time
from html import unescape
import pandas as pd
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from tqdm import tqdm
from sklearn.preprocessing import normalize
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
import torch
from sentence_transformers import SentenceTransformer


tqdm.pandas()

# ============================================================================
# TEXT CLEANING
# ============================================================================


def clean_abstract(text: str) -> str:
    """Clean and normalize abstract text by removing HTML, LaTeX, and formatting."""
    if not isinstance(text, str):
        return ""

    text = re.sub(r"<[^>]+>", " ", text)
    text = unescape(text)

    # Simplify LaTeX escapes
    text = re.sub(r'\\"([a-zA-Z])', lambda m: m.group(1), text)
    text = re.sub(r"\\'", "", text)
    text = re.sub(r"\$(.*?)\$", r"\1", text)
    text = re.sub(r"\\[a-zA-Z]+\{(.*?)\}", r"\1", text)
    text = re.sub(r"\\(begin|end)\{.*?\}", " ", text)
    text = re.sub(r"\\[a-zA-Z]+", "", text)
    text = text.replace("``", '"').replace("''", '"')
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================================
# LANGUAGE DETECTION
# ============================================================================

def detect_language(text: str) -> str:
    try:
        return detect(text)
    except LangDetectException:
        return None


# ============================================================================
# DOCUMENT CLASSIFICATION
# ============================================================================

DIAGNOSTIC_PATTERNS = re.compile(
    r'\blhd\s*(bolometer|flxloop|fpellet|interferometer|cxrs|ece|diagnostic|probe)\b|'
    r'\blhd\s*#?\d+',
    flags=re.IGNORECASE
)


def classify_document_type(title: str) -> str:
    if pd.isna(title):
        return "unknown"
    title_clean = str(title).strip().lower()
    return "diagnostic_report" if DIAGNOSTIC_PATTERNS.search(title_clean) else "scientific_paper"


# ============================================================================
# EMBEDDINGS (OpenAI or Local)
# ============================================================================

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=4, max=30))
def _get_openai_embeddings_batch(client, batch_texts, model):
    """Internal helper with retry on rate limits."""
    response = client.embeddings.create(model=model, input=batch_texts)
    return [d.embedding for d in response.data]


def _generate_openai_embeddings(texts: list, model: str = "text-embedding-3-small") -> list:
    """Generate multilingual embeddings using OpenAI API."""
    api_key = os.getenv("OPENAI_KEY")
    if not api_key:
        raise ValueError("❌ OPENAI_KEY environment variable not set.")

    client = OpenAI(api_key=api_key)
    embeddings = []

    for i in tqdm(range(0, len(texts), 100), desc="Generating OpenAI embeddings"):
        batch_texts = texts[i:i + 100]
        batch_embeddings = _get_openai_embeddings_batch(client, batch_texts, model)
        embeddings.extend(batch_embeddings)
        time.sleep(0.5)  # courtesy delay

    embeddings = normalize(embeddings)
    return [emb.tolist() for emb in embeddings]


def _generate_local_embeddings(texts: list,
                               model_name: str = "intfloat/multilingual-e5-small",
                               batch_size: int = 64) -> list:
    """Generate multilingual embeddings locally using Hugging Face models."""
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"🔍 Using local embedding model: {model_name} on {device}")

    model = SentenceTransformer(model_name, device=device)

    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating local embeddings"):
        batch_texts = texts[i:i + batch_size]
        batch_embeddings = model.encode(batch_texts, normalize_embeddings=True, convert_to_numpy=True)
        embeddings.extend(batch_embeddings)

    embeddings = normalize(embeddings)
    return [emb.tolist() for emb in embeddings]


def generate_embeddings(texts: list,
                        use_openai: bool = True,
                        model_name: str = "text-embedding-3-small") -> list:
    """Unified embedding generator that supports OpenAI or local model."""
    if use_openai:
        print("🔑 Using OpenAI API for embeddings")
        return _generate_openai_embeddings(texts, model=model_name)
    else:
        print("⚙️ Using local Hugging Face model for embeddings")
        return _generate_local_embeddings(texts)


# ============================================================================
# DATAFRAME OPERATIONS
# ============================================================================

def drop_empty_rows(df: pd.DataFrame) -> pd.DataFrame:
    print("Dropping empty rows")
    df_clean = df.dropna(subset=['title', 'abstract'])
    df_clean = df_clean[df_clean['title'].str.strip() != '']
    df_clean = df_clean[df_clean['abstract'].str.strip() != '']
    return df_clean


def apply_clean_text_to_df(df: pd.DataFrame) -> pd.DataFrame:
    print("Cleaning text")
    df["clean_abstract"] = df["abstract"].progress_apply(clean_abstract)
    return df


def add_abstract_language(df: pd.DataFrame) -> pd.DataFrame:
    print("Detecting languages")
    df['title_lang'] = df['title'].astype(str).progress_apply(detect_language)
    df['abstract_lang'] = df['clean_abstract'].astype(str).progress_apply(detect_language)
    return df


def remove_empty_lang(df: pd.DataFrame, title_col: str = 'title_lang',
                      abstract_col: str = 'abstract_lang') -> pd.DataFrame:
    return df.dropna(subset=[title_col, abstract_col])


def classify_documents(df: pd.DataFrame) -> pd.DataFrame:
    print("Classifying document types")
    df["type"] = df["title"].progress_apply(classify_document_type)
    df_diag = df[df["type"] == "diagnostic_report"].copy()
    df_sci = df[df["type"] == "scientific_paper"].copy()

    print(f"Diagnostic reports: {len(df_diag)} | Scientific papers: {len(df_sci)}")
    return df_diag, df_sci


def add_embeddings_to_df(df: pd.DataFrame,
                         text_columns: list = ['title', 'clean_abstract'],
                         use_openai: bool = True,
                         model_name: str = "text-embedding-3-small") -> pd.DataFrame:
    print("Adding multilingual embeddings")
    texts = (df[text_columns[0]].fillna('') + ' ' + df[text_columns[1]].fillna('')).tolist()
    df['embedding'] = generate_embeddings(texts, use_openai=use_openai, model_name=model_name)
    return df


# ============================================================================
# I/O OPERATIONS
# ============================================================================

def load_df(path: str = "processed/rdf_results_final.csv.gz") -> pd.DataFrame:
    print("Loading df")
    return pd.read_csv(path, compression="gzip")


def save(df: pd.DataFrame, path: str = "final/data.parquet") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"✅ Saved to {path} (Parquet format)")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main(use_openai: bool = False):
    """Execute the full multilingual data processing pipeline."""
    df = load_df("processed/rdf_results_final.csv.gz")
    print(df.shape)
    df = drop_empty_rows(df)
    df = apply_clean_text_to_df(df)
    df_diag, df_sci = classify_documents(df)
    df_sci = add_embeddings_to_df(df_sci, use_openai=use_openai)
    save(df_sci, "final/data_sci.parquet")
    save(df_diag, "final/data_diag.parquet")


if __name__ == "__main__":
    main(use_openai=False)
