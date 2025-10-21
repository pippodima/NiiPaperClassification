import os
import requests
import pandas as pd
from tqdm import tqdm
import xml.etree.ElementTree as ET

OUTPUT_DIR = "raw"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "nii.csv")
CSV_PATH = "raw/article_labels.csv"


def get_paper_metadata(uri):
    r = requests.get(uri)
    if r.status_code == 404:
        print(f"⚠️ Not found: {uri}")
        return None, None, None
    r.raise_for_status()
    root = ET.fromstring(r.text)

    ns = {
        "dc": "http://purl.org/dc/elements/1.1/",
        "jpcoar": "https://github.com/JPCOAR/schema/blob/master/2.0/",
        "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "dcterms": "http://purl.org/dc/terms/",
        "datacite": "https://schema.datacite.org/meta/kernel-4/",
        "foaf": "http://xmlns.com/foaf/0.1/"
    }

    # --- Titles ---
    titles = [el.text for el in root.findall(".//dc:title", ns) if el.text]
    title_en = next((t for t in titles if "en" in (t or "").lower()), titles[0] if titles else None)

    # --- Abstract ---
    abstract = None
    for desc_path in [
        ".//description/notation",
        ".//datacite:description",
        ".//dc:description"
    ]:
        el = root.find(desc_path, ns)
        if el is not None and el.text:
            abstract = el.text.strip()
            break

    # --- Keywords / Subjects ---
    keywords = None

    # 1. Check dcterms:subject
    for subj in root.findall(".//dcterms:subject", ns):
        notation_el = subj.find("./notation")
        if notation_el is not None and notation_el.text:
            keywords = notation_el.text.strip()
            break

    # 2. If nothing, check foaf:topic titles
    if not keywords:
        for topic in root.findall(".//foaf:topic", ns):
            title = topic.attrib.get("{http://purl.org/dc/elements/1.1/}title")
            if title:
                keywords = title.strip()
                break

    return title_en, abstract, keywords


def main():
    df = pd.read_csv(CSV_PATH)
    titles = []
    abstracts = []
    keywords = []

    for uri in tqdm(df["uri"]):
        title, abstract, keywords = get_paper_metadata(uri)
        titles.append(title)
        abstracts.append(abstract)

    df["titles"] = titles
    df["abstracts"] = abstracts
    df["keywords"] = keywords
    df.to_csv(OUTPUT_FILE, index=False)
    print("✅ Finished! Saved to papers_with_metadata.csv")


if __name__ == "__main__":
    main()
