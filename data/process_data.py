import os
import re
from html import unescape
import pandas as pd
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from sklearn.preprocessing import normalize

tqdm.pandas()


# ============================================================================
# TEXT CLEANING
# ============================================================================

def clean_abstract(text: str) -> str:
    """Clean and normalize abstract text by removing HTML, LaTeX, and formatting."""
    if not isinstance(text, str):
        return ""

    # Remove HTML tags and decode entities
    text = re.sub(r"<[^>]+>", " ", text)
    text = unescape(text)

    # Handle LaTeX accent escapes (e.g., Bl\"ochl -> Blöchl)
    text = re.sub(r'\\"([a-zA-Z])', lambda m: m.group(1).replace('o', 'ö').replace('a', 'ä').replace('u', 'ü'), text)
    text = re.sub(r"\\'", "", text)

    # Replace LaTeX math ($...$) with inner text only
    text = re.sub(r"\$(.*?)\$", r"\1", text)

    # Simplify LaTeX commands but keep inner content
    text = re.sub(r"\\[a-zA-Z]+\{(.*?)\}", r"\1", text)
    text = re.sub(r"\\(begin|end)\{.*?\}", " ", text)
    text = re.sub(r"\\[a-zA-Z]+", "", text)

    # Normalize quotes and whitespace
    text = text.replace("``", '"').replace("''", '"')
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================================
# LANGUAGE DETECTION
# ============================================================================

def detect_language(text: str) -> str:
    """Detect language of text, returning None on failure."""
    try:
        return detect(text)
    except LangDetectException:
        return None


# ============================================================================
# DOCUMENT CLASSIFICATION
# ============================================================================

# Compile regex once at module level for efficiency
DIAGNOSTIC_PATTERNS = re.compile(
    r'\blhd\s*(bolometer|flxloop|fpellet|interferometer|cxrs|ece|diagnostic|probe)\b|'
    r'\blhd\s*#?\d+',
    flags=re.IGNORECASE
)


def classify_document_type(title: str) -> str:
    """Classify document as diagnostic report or scientific paper."""
    if pd.isna(title):
        return "unknown"

    title_clean = str(title).strip().lower()
    if DIAGNOSTIC_PATTERNS.search(title_clean):
        return "diagnostic_report"
    return "scientific_paper"


# ============================================================================
# EMBEDDING GENERATION
# ============================================================================

def generate_embeddings(texts: list, model_name: str, device: str, batch_size: int) -> list:
    """Generate normalized embeddings for a list of texts."""
    model = SentenceTransformer(model_name, device=device)
    embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Computing embeddings"):
        batch_texts = texts[i:i + batch_size]
        batch_embeddings = model.encode(batch_texts, show_progress_bar=False)
        embeddings.extend(batch_embeddings)

    embeddings = normalize(embeddings)
    return [emb.tolist() for emb in embeddings]


# ============================================================================
# DATAFRAME OPERATIONS
# ============================================================================

def drop_empty_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with missing or empty title/abstract."""
    print("Dropping empty rows")
    df_clean = df.dropna(subset=['title', 'abstract'])
    df_clean = df_clean[df_clean['title'].str.strip() != '']
    df_clean = df_clean[df_clean['abstract'].str.strip() != '']
    return df_clean


def apply_clean_text_to_df(df: pd.DataFrame) -> pd.DataFrame:
    """Apply text cleaning to abstract column."""
    print("Cleaning text")
    df["clean_abstract"] = df["abstract"].progress_apply(clean_abstract)
    return df


def add_abstract_language(df: pd.DataFrame) -> pd.DataFrame:
    """Detect and add language columns for title and abstract."""
    print("Detecting and applying languages")
    df['title_lang'] = df['title'].astype(str).progress_apply(detect_language)
    df['abstract_lang'] = df['clean_abstract'].astype(str).progress_apply(detect_language)
    return df


def remove_empty_lang(df: pd.DataFrame, title_col: str = 'title_lang',
                      abstract_col: str = 'abstract_lang') -> pd.DataFrame:
    """Remove rows with missing language detection."""
    return df.dropna(subset=[title_col, abstract_col])


def classify_documents(df: pd.DataFrame) -> pd.DataFrame:
    """Add document type classification column."""
    print("Classifying types")
    df["type"] = df["title"].progress_apply(classify_document_type)
    return df


def add_embeddings_to_df(df: pd.DataFrame, device: str = "mps",
                         text_columns: list = ['title', 'clean_abstract'],
                         model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
                         batch_size: int = 64) -> pd.DataFrame:
    """Generate and add embeddings to dataframe."""
    print("Adding embeddings")

    # Combine text columns
    texts = (df[text_columns[0]].fillna('') + ' ' + df[text_columns[1]].fillna('')).tolist()

    # Generate embeddings
    df['embedding'] = generate_embeddings(texts, model_name, device, batch_size)

    return df


def split_by_language(df: pd.DataFrame, title_col: str = 'title_lang',
                      abstract_col: str = 'abstract_lang') -> tuple:
    """Split dataframe into language-based subsets."""
    print("Splitting df based on languages")

    df_full_en = df[(df[title_col] == 'en') & (df[abstract_col] == 'en')].reset_index(drop=True)
    df_full_jp = df[(df[title_col] == 'ja') & (df[abstract_col] == 'ja')].reset_index(drop=True)
    df_title_jp_abstract_en = df[(df[title_col] == 'ja') & (df[abstract_col] == 'en')].reset_index(drop=True)
    df_title_en_abstract_jp = df[(df[title_col] == 'en') & (df[abstract_col] == 'ja')].reset_index(drop=True)

    return df_full_en, df_full_jp, df_title_jp_abstract_en, df_title_en_abstract_jp


# ============================================================================
# I/O OPERATIONS
# ============================================================================

def load_df(path: str = "processed/rdf_results_final.csv.gz") -> pd.DataFrame:
    """Load dataframe from compressed CSV."""
    print("Loading df")
    return pd.read_csv(path, compression="gzip")


def load_sample_random_rows(path: str, n: int, random_state: int = None) -> pd.DataFrame:
    """Load random sample of rows from compressed CSV."""
    return pd.read_csv(path, compression="gzip").sample(n=n, random_state=random_state).reset_index(drop=True)


def save(df: pd.DataFrame, path: str = "final/data.csv.gz") -> None:
    """Save dataframe to compressed CSV, creating folders if missing."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, compression="gzip")
    print(f"✅ Saved to {path}")


def show_len(df: pd.DataFrame) -> None:
    """Display number of files in dataframe."""
    print(len(df["file"]))


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Execute the full data processing pipeline."""
    df = load_df("processed/rdf_results_final.csv.gz")

    # Data cleaning and preparation
    df = drop_empty_rows(df)
    df = apply_clean_text_to_df(df)

    # Language detection and filtering
    df = add_abstract_language(df)
    df = remove_empty_lang(df)

    # Feature engineering
    df = add_embeddings_to_df(df)
    df = classify_documents(df)

    # Save results
    save(df, "final/data.csv.gz")


if __name__ == "__main__":
    main()
