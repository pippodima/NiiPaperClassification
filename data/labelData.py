import pandas as pd
from concepts import FOS_MAJOR_CONCEPT_IDS


def add_major_topics_column(df_or_path):
    # ====== Load data if path provided ======
    if isinstance(df_or_path, str):
        df = pd.read_csv(df_or_path)
    else:
        df = df_or_path.copy()

    # ====== Build reverse lookup ======
    concept_to_major = {
        cid: major
        for major, ids in FOS_MAJOR_CONCEPT_IDS.items()
        for cid in ids
    }

    # ====== Map each concept_id to major topic ======
    df["major_topics"] = df["concept_id"].map(concept_to_major)
    df["major_topics"].fillna("Other / Unclassified")

    # ===== Save df =====
    save(df)

    return df


def save(df, path="processed/openalex_concepts_and_majorConcepts.csv"):
    df.to_csv(path, index=False)


def main():
    add_major_topics_column("raw/openalex_fos_concepts.csv")


if __name__ == "__main__":
    main()
