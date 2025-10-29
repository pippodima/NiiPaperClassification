import xml.etree.ElementTree as ET
import glob
import os
import random
import csv
import gzip

# Define namespaces
ns = {
    'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
    'dc': 'http://purl.org/dc/elements/1.1/',
    'dcterms': 'http://purl.org/dc/terms/',
    'cinii': 'https://cir.nii.ac.jp/schema/1.0/'
}


def parse_rdf(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Extract title (prefer Japanese)
    title = None
    titles = root.findall('.//dc:title', ns)
    if titles:
        for t in titles:
            if t.attrib.get('{http://www.w3.org/XML/1998/namespace}lang') != 'en':
                title = t.text
                break
        if not title:
            title = titles[0].text

    # Extract abstract
    abstract_elem = root.find('.//cinii:description/cinii:notation', ns)
    abstract = abstract_elem.text.strip() if abstract_elem is not None else None

    return {
        'file': file_path,
        'title': title,
        'abstract': abstract
    }


def find_rdf_files(root_folder, pattern="*.rdf"):
    """Recursively find all RDF files in a folder and its subfolders."""
    search_pattern = os.path.join(root_folder, '**', pattern)
    return glob.glob(search_pattern, recursive=True)


def save_results_compressed(results, output_file):
    """Save results to a compressed CSV (.csv.gz)."""
    with gzip.open(output_file, mode='wt', newline='', encoding='utf-8') as gzfile:
        writer = csv.DictWriter(gzfile, fieldnames=['file', 'title', 'abstract'])
        writer.writeheader()
        for row in results:
            writer.writerow(row)


def append_to_checkpoint(checkpoint_file, batch_number):
    """Record that a batch was completed."""
    with open(checkpoint_file, 'a', encoding='utf-8') as f:
        f.write(f"{batch_number}\n")


def load_checkpoint(checkpoint_file):
    """Load completed batch numbers from checkpoint."""
    if not os.path.exists(checkpoint_file):
        return set()
    with open(checkpoint_file, 'r', encoding='utf-8') as f:
        return set(int(line.strip()) for line in f if line.strip().isdigit())


def merge_batches(output_folder, final_output, checkpoint_file):
    """Merge all batch CSVs into one compressed CSV, then clean up."""
    batch_files = sorted(glob.glob(os.path.join(output_folder, 'rdf_results_batch_*.csv.gz')))
    if not batch_files:
        print("No batch files found to merge.")
        return

    with gzip.open(final_output, mode='wt', newline='', encoding='utf-8') as gzout:
        writer = csv.DictWriter(gzout, fieldnames=['file', 'title', 'abstract'])
        writer.writeheader()

        for batch_file in batch_files:
            with gzip.open(batch_file, mode='rt', newline='', encoding='utf-8') as gzin:
                reader = csv.DictReader(gzin)
                for row in reader:
                    writer.writerow(row)

    print(f"Merged {len(batch_files)} batch files into {final_output}")

    # ✅ Cleanup: remove temporary batch files and checkpoint
    for bf in batch_files:
        try:
            os.remove(bf)
        except Exception as e:
            print(f"Warning: could not delete {bf}: {e}")

    if os.path.exists(checkpoint_file):
        try:
            os.remove(checkpoint_file)
            print("Removed checkpoint file.")
        except Exception as e:
            print(f"Warning: could not delete checkpoint file: {e}")

    print("Temporary batch files deleted. Cleanup complete.")


def process_rdf_folder_batches(root_folder, batch_size=1000, max_docs=None, output_folder="processed"):
    """Process RDF files recursively in batches with crash recovery."""
    os.makedirs(output_folder, exist_ok=True)
    checkpoint_file = os.path.join(output_folder, 'checkpoint.txt')
    completed_batches = load_checkpoint(checkpoint_file)

    rdf_files = find_rdf_files(root_folder)

    if max_docs is not None and max_docs < len(rdf_files):
        random.seed(42)
        rdf_files = random.sample(rdf_files, max_docs)

    total_files = len(rdf_files)
    print(f"Found {total_files} files. Processing in batches of {batch_size}...")

    batch_number = 1
    for i in range(0, total_files, batch_size):
        if batch_number in completed_batches:
            print(f"Skipping batch {batch_number} (already processed)")
            batch_number += 1
            continue

        batch_files = rdf_files[i:i + batch_size]
        results = [parse_rdf(f) for f in batch_files]

        output_file = f"{output_folder}/rdf_results_batch_{batch_number}.csv.gz"
        save_results_compressed(results, output_file)
        append_to_checkpoint(checkpoint_file, batch_number)
        print(f"Saved batch {batch_number} ({len(results)} files)")

        batch_number += 1

    # ✅ After all batches are processed, merge and clean up
    final_output = os.path.join(output_folder, "rdf_results_final.csv.gz")
    merge_batches(output_folder, final_output, checkpoint_file)
    print(f"All results saved to {final_output}")


if __name__ == "__main__":
    root_folder = "raw"
    batch_size = 10000
    max_documents = None  # or None for all (0-150.000)

    process_rdf_folder_batches(root_folder, batch_size=batch_size, max_docs=max_documents, output_folder="processed")
