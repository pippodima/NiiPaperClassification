import re
from html import unescape
import pandas as pd
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


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
    df["clean_abstract"] = df["abstract"].apply(clean_abstract)
    return df


def add_abstract_language(df):
    def detect_lang(text):
        try:
            return detect(text)
        except LangDetectException:
            return None

    # Detect language for each abstract
    df['abstract_lang'] = df['clean_abstract'].astype(str).apply(detect_lang)
    return df


def save(df, path="final/data.csv.gz"):
    df.to_csv(path, index=False, compression="gzip")
    print(f"✅ Saved to {path}")


def drop_empty_rows(df):
    # Drop rows where title or abstract is missing or empty
    df_clean = df.dropna(subset=['title', 'abstract'])
    df_clean = df_clean[df_clean['title'].str.strip() != '']
    df_clean = df_clean[df_clean['abstract'].str.strip() != '']
    return df_clean


def add_embeddings_to_df(df, device="mps", text_columns=['title', 'clean_abstract'], model_name="all-MPNet-base-v2", batch_size=64):
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

    df['embedding'] = [emb.tolist() for emb in embeddings]  # Convert np.array to list for storage

    return df


def split_by_language(df, lang_column='abstract_lang'):
    df_en = df[df[lang_column] == 'en'].reset_index(drop=True)
    df_jp = df[df[lang_column] == 'ja'].reset_index(drop=True)
    return df_en, df_jp


def load_sample_random_rows(path, n, random_state=None):
    return pd.read_csv(path, compression="gzip").sample(n=n, random_state=random_state).reset_index(drop=True)


def load_df(path="processed/rdf_results_final.csv.gz"):
    return pd.read_csv(path, compression="gzip")


def main():
    # df = load_sample_random_rows("processed/rdf_results_final.csv.gz", 100, random_state=42)
    df = load_df("processed/rdf_results_final.csv.gz")
    df = drop_empty_rows(df)
    df = apply_clean_text_to_df(df)
    df = add_abstract_language(df)
    df = add_embeddings_to_df(df)
    df_en, df_jp = split_by_language(df)
    save(df_en, "final/data_eng.csv.gz")
    save(df_jp, "final/data_jp.csv.gz")


if __name__ == "__main__":
    main()