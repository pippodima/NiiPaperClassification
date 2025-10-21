import os
import requests
import pandas as pd
from tqdm import tqdm
import xml.etree.ElementTree as ET
from sentence_transformers import SentenceTransformer

OUTPUT_DIR = "raw"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "nii.csv")
CSV_PATH = "raw/article_labels.csv"


ns = {
    'dc': 'http://purl.org/dc/elements/1.1/',
    'dcterms': 'http://purl.org/dc/terms/',
    'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
    'cinii': 'http://ci.nii.ac.jp/ns/1.0/',
    'default': 'https://cir.nii.ac.jp/schema/1.0/'
}


# Load XML
def load_xml(xml_text):
    return ET.fromstring(xml_text)


# Extract title
def extract_title(root):
    title_elem = root.find('.//dc:title', ns)
    return title_elem.text if title_elem is not None else None


# Extract abstract
def extract_abstract(root):
    # Find description element whose type is Abstract
    for desc in root.findall('.//default:description', ns):
        type_elem = desc.find('default:type', ns)
        if type_elem is not None and type_elem.text == 'Abstract':
            notation_elem = desc.find('default:notation', ns)
            if notation_elem is not None:
                return notation_elem.text
    return None


# Extract keywords (subjects)
def extract_keywords(root):
    keywords = []
    # Extract all notation under dcterms:subject
    for subject in root.findall('.//dcterms:subject', ns):
        for notation in subject.findall('.//*'):
            if notation.tag.endswith('notation') and notation.text:
                keywords.append(notation.text)
    return keywords


def main():
    df = pd.read_csv(CSV_PATH)
    titles = []
    abstracts = []
    keywords = []

    for uri in tqdm(df["uri"]):
        try:
            r = requests.get(uri)
            if r.status_code == 404:
                print(f"⚠️ Not found: {uri}")
                titles.append(None)
                abstracts.append(None)
                keywords.append(None)
                continue  # Skip this URI and go to the next

            r.raise_for_status()  # Raise exception for other HTTP errors
            xml_text = r.text

            root = load_xml(xml_text)
            title = extract_title(root)
            abstract = extract_abstract(root)
            keyword = extract_keywords(root)

            titles.append(title)
            abstracts.append(abstract)
            keywords.append(keyword)

        except requests.exceptions.RequestException as e:
            print(f"❌ Error fetching {uri}: {e}")
            titles.append(None)
            abstracts.append(None)
            keywords.append(None)
            continue  # Skip to the next URI

    # Save results
    df["titles"] = titles
    df["abstracts"] = abstracts
    df["keywords"] = keywords

    df = df.dropna(subset=["titles", "abstracts"])
    df = df[(df['titles'].str.strip() != '') & (df['abstracts'].str.strip() != '')]  # removes empty strings

    df = df[df['confidence'].str.lower() != 'skip']

    df.to_csv(OUTPUT_FILE, index=False)
    print("✅ Finished! Saved to papers_with_metadata.csv")


def embed_title_and_abstract():
    df = pd.read_csv(OUTPUT_FILE)
    df["embedding"] = df["titles"].astype(str) + ". " + df["abstracts"].astype(str)

    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    # model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

    df["embedding"] = df["embedding"].apply(lambda x: model.encode(x))

    df.to_csv("final/nii.csv")


if __name__ == "__main__":
    # main()
    embed_title_and_abstract()
