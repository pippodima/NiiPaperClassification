import re
from html import unescape
import pandas as pd


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


def apply_clean_text_to_df(df_or_path):
    if isinstance(df_or_path, str):
        df = pd.read_csv(df_or_path)
    else:
        df = df_or_path.copy()
    df["clean_abstract"] = df["abstract"].apply(clean_abstract)
    save(df)


def save(df, path="final/data.csv"):
    df.to_csv(path, index=False)


def main():
    apply_clean_text_to_df("processed/openalex_concepts_and_majorConcepts.csv")


if __name__ == "__main__":
    main()
