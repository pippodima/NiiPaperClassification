import re
from html import unescape
import pandas as pd
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from sklearn.preprocessing import normalize
tqdm.pandas()


def clean_abstract(text: str) -> str:
    if not isinstance(text, str):
        return ""

    # 1️⃣ Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # 2️⃣ Decode HTML entities
    text = unescape(text)

    # 3️⃣ Handle LaTeX accent escapes (e.g., Bl\"ochl -> Blöchl)
    text = re.sub(r'\\"([a-zA-Z])', lambda m: m.group(1).replace('o', 'ö').replace('a', 'ä').replace('u', 'ü'), text)
    text = re.sub(r"\\'", "", text)  # remove single-quote accents if any

    # 4️⃣ Replace LaTeX math ($...$) with inner text only (preserve chemical notation)
    text = re.sub(r"\$(.*?)\$", r"\1", text)

    # 5️⃣ Simplify LaTeX commands but keep inner content
    text = re.sub(r"\\[a-zA-Z]+\{(.*?)\}", r"\1", text)
    text = re.sub(r"\\(begin|end)\{.*?\}", " ", text)

    # 6️⃣ Remove residual backslashes (for commands like \mathrm, \text)
    text = re.sub(r"\\[a-zA-Z]+", "", text)

    # 7️⃣ Replace multiple spaces and normalize quotes
    text = text.replace("``", '"').replace("''", '"')
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def apply_clean_text_to_df(df):
    print("Cleaning text")
    df["clean_abstract"] = df["abstract"].progress_apply(clean_abstract)
    return df


def classify_document_type(df: pd.DataFrame) -> pd.DataFrame:
    # Define regex patterns for known diagnostic or operational report types
    diagnostic_patterns = [
        r'\blhd\s*bolometer\b',
        r'\blhd\s*flxloop\b',
        r'\blhd\s*fpellet\b',
        r'\blhd\s*interferometer\b',
        r'\blhd\s*cxrs\b',
        r'\blhd\s*ece\b',
        r'\blhd\s*diagnostic\b',
        r'\blhd\s*probe\b'
    ]

    # Combine into one regex
    diagnostic_regex = re.compile("|".join(diagnostic_patterns), flags=re.IGNORECASE)

    def detect_type(title: str) -> str:
        if pd.isna(title):
            return "unknown"
        title_clean = str(title).strip().lower()
        # Identify diagnostic/operational reports
        if diagnostic_regex.search(title_clean):
            return "diagnostic_report"
        # Short generic patterns like "LHD #123" also likely reports
        if re.search(r'\blhd\s*#?\d+', title_clean):
            return "diagnostic_report"
        # Default case
        return "scientific_paper"

    print("Classifying types")
    df["type"] = df["title"].progress_apply(detect_type)
    return df


def add_abstract_language(df):
    print("Detecting and applying languages")

    def detect_lang(text):
        try:
            return detect(text)
        except LangDetectException:
            print("Error in detecting Language")
            return None

    # Detect language for each abstract
    df['abstract_lang'] = df['clean_abstract'].astype(str).progress_apply(detect_lang)
    df['title_lang'] = df['title'].astype(str).progress_apply(detect_lang)
    return df


def save(df, path="final/data.csv.gz"):
    df.to_csv(path, index=False, compression="gzip")
    print(f"✅ Saved to {path}")


def drop_empty_rows(df):
    # Drop rows where title or abstract is missing or empty
    print("Dropping empty rows")
    df_clean = df.dropna(subset=['title', 'abstract'])
    df_clean = df_clean[df_clean['title'].str.strip() != '']
    df_clean = df_clean[df_clean['abstract'].str.strip() != '']
    return df_clean


def add_embeddings_to_df(df, device="mps",
                         text_columns=['title', 'clean_abstract'],
                         model_name="paraphrase-multilingual-MiniLM-L12-v2",
                         batch_size=64):
    print("Adding embeddings")
    # Combine specified text columns into one string per row
    texts = (df[text_columns[0]].fillna('') + ' ' + df[text_columns[1]].fillna('')).tolist()

    # Load embedding model
    model = SentenceTransformer(model_name, device=device)

    embeddings = []

    # Compute embeddings in batches with progress bar
    for i in tqdm(range(0, len(texts), batch_size), desc="Computing embeddings"):
        batch_texts = texts[i:i+batch_size]
        batch_embeddings = model.encode(batch_texts, show_progress_bar=False)
        embeddings.extend(batch_embeddings)

    embeddings = normalize(embeddings)

    df['embedding'] = [emb.tolist() for emb in embeddings]  # Convert np.array to list for storage

    return df


def split_by_language(df, title_col='title_lang', abstract_col='abstract_lang'):
    print("Splitting df based on languages")
    # Full English: both title and abstract are English
    df_full_en = df[(df[title_col] == 'en') & (df[abstract_col] == 'en')].reset_index(drop=True)
    # Full Japanese: both title and abstract are Japanese
    df_full_jp = df[(df[title_col] == 'ja') & (df[abstract_col] == 'ja')].reset_index(drop=True)
    # Japanese title, English abstract
    df_title_jp_abstract_en = df[(df[title_col] == 'ja') & (df[abstract_col] == 'en')].reset_index(drop=True)
    # English title, Japanese abstract
    df_title_en_abstract_jp = df[(df[title_col] == 'en') & (df[abstract_col] == 'ja')].reset_index(drop=True)
    return df_full_en, df_full_jp, df_title_jp_abstract_en, df_title_en_abstract_jp


def remove_empty_lang(df, title_col='title_lang', abstract_col='abstract_lang'):
    """
    Drop rows where title_lang or abstract_lang is None/NaN.
    """
    return df.dropna(subset=[title_col, abstract_col])


def load_sample_random_rows(path, n, random_state=None):
    return pd.read_csv(path, compression="gzip").sample(n=n, random_state=random_state).reset_index(drop=True)


def load_df(path="processed/rdf_results_final.csv.gz"):
    print("Loading df")
    return pd.read_csv(path, compression="gzip")


def show_len(df: pd.DataFrame):
    print(len(df["file"]))


def main():
    # df = load_sample_random_rows("processed/rdf_results_final.csv.gz", 10000, random_state=42)
    df = load_df("processed/rdf_results_final.csv.gz")
    df = drop_empty_rows(df)
    df = apply_clean_text_to_df(df)
    df = add_abstract_language(df)
    df = remove_empty_lang(df)
    df = add_embeddings_to_df(df)
    df = classify_document_type(df)
    save(df, "final/data.csv.gz")


if __name__ == "__main__":
    # main()
    df = load_df("final/data.csv.gz")
    print(df.columns)