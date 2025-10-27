import requests
import pandas as pd
from tqdm import tqdm
import langid
import os
from sentence_transformers import SentenceTransformer

CSV_PATH = "raw/cinii_random_1000.csv"
OUTPUT_FILE = "raw/nii_big.csv"

def safe_get_json(url):
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"❌ Error fetching {url}: {e}")
        return None

def extract_metadata(json_data):
    """Extract title, abstract, and keywords from CiNii JSON"""
    if not json_data:
        return None, None, None
    title = json_data.get("title") or json_data.get("title_ja")
    abstract = json_data.get("description") or json_data.get("description_ja")
    kw = json_data.get("keywords") or []
    if isinstance(kw, dict):
        kw = list(kw.values())
    return title, abstract, kw

def detect_language(text):
    if not text or pd.isna(text): return None
    return langid.classify(str(text))[0]

def main():
    df = pd.read_csv(CSV_PATH)
    titles, abstracts, keywords = [], [], []

    for uri in tqdm(df["id"]):
        # Convert to JSON endpoint if necessary
        if "ci.nii.ac.jp" in uri:
            uri = uri.replace("ci.nii.ac.jp/naid", "cir.nii.ac.jp/crid") + ".json"
        elif not uri.endswith(".json"):
            uri = uri.rstrip("/") + ".json"

        data = safe_get_json(uri)
        t, a, k = extract_metadata(data)
        titles.append(t)
        abstracts.append(a)
        keywords.append(k)

    df["titles"] = titles
    df["abstracts"] = abstracts
    df["keywords"] = keywords

    df = df.dropna(subset=["titles", "abstracts"])
    df = df[(df["titles"].str.strip() != "") & (df["abstracts"].str.strip() != "")]

    df["language"] = df["abstracts"].apply(detect_language)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Saved {len(df)} papers to {OUTPUT_FILE}")

def embed_title_and_abstract():
    df = pd.read_csv(OUTPUT_FILE)
    df = df[df["language"] != "ja"]
    df["text"] = df["titles"].astype(str) + ". " + df["abstracts"].astype(str)
    model = SentenceTransformer("allenai-specter")
    df["embedding"] = df["text"].apply(lambda x: model.encode(x))
    os.makedirs("final", exist_ok=True)
    df.to_csv("final/nii.csv", index=False)
    print("✅ Embeddings saved to final/nii_big.csv")

if __name__ == "__main__":
    main()
    embed_title_and_abstract()
