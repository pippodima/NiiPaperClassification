#!/usr/bin/env python3
"""
Fetch ~1000 random CiNii records via OpenSearch and save metadata to CSV.
"""
import csv
import random
import time
import requests
import xml.etree.ElementTree as ET

USER_AGENT = "cinii-fetch/1.0 (+https://example.local/)"
URL = "https://ci.nii.ac.jp/opensearch/search"

# Some broad keywords to randomize queries
KEYWORDS = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "l", "m", "n", "o", "p", "q", "r", "s", "t",
            "u", "v", "w", "x", "y", "z", "j", "k"]

N_DOCS = 1000
COUNT_PER_QUERY = 100
OUTPUT_CSV = "raw/cinii_random_1000.csv"


def search_opensearch(query, count=50):
    params = {"q": query, "count": str(count), "format": "atom"}
    headers = {"User-Agent": USER_AGENT}
    r = requests.get(URL, params=params, headers=headers, timeout=30)
    r.raise_for_status()
    root = ET.fromstring(r.content)
    return root.findall(".//{http://www.w3.org/2005/Atom}entry")


def extract_from_atom_entry(entry):
    ns = "{http://www.w3.org/2005/Atom}"

    def text_tag(tag):
        el = entry.find(ns + tag)
        return el.text.strip() if el is not None and el.text else ""
    title = text_tag("title")
    id_ = text_tag("id")
    published = text_tag("published") or text_tag("updated")
    authors = [a.find(ns + "name").text for a in entry.findall(ns + "author") if a.find(ns + "name") is not None]
    links = [l.attrib.get("href") for l in entry.findall(ns + "link") if l.attrib.get("href")]
    return {"title": title, "id": id_, "published": published, "authors": authors, "urls": links}


def main():
    results, seen = [], set()
    while len(results) < N_DOCS:
        query = random.choice(KEYWORDS)
        print(f"Searching for keyword: {query!r}")
        try:
            entries = search_opensearch(query, count=COUNT_PER_QUERY)
        except Exception as e:
            print("Request failed:", e)
            time.sleep(2)
            continue

        for entry in entries:
            rec = extract_from_atom_entry(entry)
            if rec["id"] in seen:
                continue
            seen.add(rec["id"])
            results.append(rec)
            if len(results) >= N_DOCS:
                break
        print(f"Collected so far: {len(results)}")
        time.sleep(1.0)

    fields = ["title", "id", "published", "authors", "urls"]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rec in results:
            writer.writerow({
                "title": rec.get("title", ""),
                "id": rec.get("id", ""),
                "published": rec.get("published", ""),
                "authors": "; ".join(rec.get("authors", [])),
                "urls": "; ".join(rec.get("urls", [])),
            })
    print(f"\n✅ Saved {len(results)} records to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
