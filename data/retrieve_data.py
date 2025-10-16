import os
import time
import requests
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from concepts import FOS_CONCEPT_IDS

# ====== PARAMETERS ======
OUTPUT_DIR = "raw"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "openalex_fos_concepts.csv")
WORKS_PER_CONCEPT = 2000
PER_PAGE = 200
MAX_RETRIES = 3
REQUEST_DELAY = 1.2
MAX_WORKERS = 4  # number of parallel threads

# ====== LOCK for thread-safe I/O ======
lock = Lock()


# ====== HELPER: Fetch concept metadata ======
def get_concept_metadata(concept_id, cache={}):
    if concept_id in cache:
        return cache[concept_id]

    base_url = f"https://api.openalex.org/concepts/{concept_id}"
    try:
        response = requests.get(base_url, timeout=20)
        response.raise_for_status()
        data = response.json()
        name = data.get("display_name", f"Unknown_{concept_id}")
        description = data.get("description", "")
    except requests.exceptions.RequestException as e:
        print(f"⚠️ Failed to fetch metadata for {concept_id}: {e}")
        name, description = f"Unknown_{concept_id}", ""

    cache[concept_id] = {"name": name, "description": description}
    time.sleep(0.2)
    return cache[concept_id]


# ====== FETCH PAPERS ======
def fetch_openalex_data(concept_id, max_records=2000):
    base_url = "https://api.openalex.org/works"
    works = []
    collected = 0

    for page_start in range(0, max_records, PER_PAGE):
        if collected >= max_records:
            break

        page_num = page_start // PER_PAGE + 1
        params = {
            "filter": f"has_abstract:true,concepts.id:{concept_id}",
            "per-page": PER_PAGE,
            "sort": "cited_by_count:desc",
            "page": page_num,
        }

        for attempt in range(MAX_RETRIES):
            try:
                response = requests.get(base_url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json().get("results", [])
                break
            except requests.exceptions.RequestException as e:
                if attempt < MAX_RETRIES - 1:
                    print(f"⚠️ Retry {attempt + 1}/{MAX_RETRIES} for {concept_id} page {page_num}: {e}")
                    time.sleep(5)
                else:
                    print(f"❌ Failed {concept_id} page {page_num} after {MAX_RETRIES} retries.")
                    data = []
        if not data:
            continue

        for item in data:
            abstract = item.get("abstract_inverted_index")
            if not abstract:
                continue

            n_words = max(pos for positions in abstract.values() for pos in positions) + 1
            words_list = [None] * n_words
            for word, positions in abstract.items():
                for pos in positions:
                    words_list[pos] = word
            abstract_text = " ".join(w for w in words_list if w)

            works.append({
                "id": item.get("id"),
                "title": item.get("title"),
                "abstract": abstract_text,
                "publication_year": item.get("publication_year"),
                "concept_id": concept_id,
            })
            collected += 1
            if collected >= max_records:
                break

        time.sleep(REQUEST_DELAY)

    return works


# ====== FETCH SINGLE TOPIC ======
def fetch_topic(topic_name, concept_id, cache):
    """Fetch data for one topic (concept). Returns a list of records."""
    print(f"\n📘 Fetching: {topic_name}")
    meta = get_concept_metadata(concept_id, cache=cache)
    official_name = meta["name"]
    official_desc = meta["description"]
    print(f"➡️ {concept_id}: {official_name}")

    records = fetch_openalex_data(concept_id, WORKS_PER_CONCEPT)
    for rec in records:
        rec["topic"] = topic_name
        rec["concept_name"] = official_name
        rec["concept_description"] = official_desc
    return records


# ====== MAIN ======
def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    temp_file = OUTPUT_FILE + ".tmp"

    # Resume if checkpoint exists
    if os.path.exists(temp_file):
        df_existing = pd.read_csv(temp_file)
        completed = set(df_existing["topic"].unique())
        print(f"🔄 Resuming from checkpoint. Already fetched: {len(completed)} topics.")
        all_records = df_existing.to_dict("records")
    else:
        completed = set()
        all_records = []

    concept_cache = {}

    topics_to_run = {k: v for k, v in FOS_CONCEPT_IDS.items() if k not in completed}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(fetch_topic, topic, cid, concept_cache): topic
            for topic, cid in topics_to_run.items()
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching Concepts"):
            topic = futures[future]
            try:
                records = future.result()
                with lock:
                    all_records.extend(records)
                    pd.DataFrame(all_records).to_csv(temp_file, index=False)
                    print(f"💾 Checkpoint saved after {topic} ({len(all_records)} records).")
            except Exception as e:
                print(f"❌ Error fetching {topic}: {e}")

    # Final save
    df_final = pd.DataFrame(all_records)
    df_final.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ Finished! Saved {len(df_final)} records to {OUTPUT_FILE}")

    if os.path.exists(temp_file):
        os.remove(temp_file)
        print("🧹 Temporary file removed.")


if __name__ == "__main__":
    main()



