import ast
import re
import unicodedata
import warnings
import mojimoji
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from umap import UMAP
import hdbscan
from tqdm import tqdm
import ollama

from plot_clusters import plot_embedding, plot_embedding_interactive

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================================
# CONSTANTS
# ============================================================================

BOILERPLATE_WORDS = {
    "pdf", "article", "text", "type", "source", "application",
    "identifier", "pp", "doc", "document", "資料番号", "論文",
    "著者", "journal", "abstract", "file", "doi", "url"
}

# Compile regex patterns once at module level
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
EMAIL_PATTERN = re.compile(r"\S+@\S+")
FILE_PATTERN = re.compile(r"\b\w+\.(pdf|docx?|xlsx?|txt)\b")
NON_USEFUL_CHARS = re.compile(r"[^ぁ-んァ-ン一-龥a-z0-9\s]")
SINGLE_LETTER = re.compile(r"\b[a-z]\b")
WHITESPACE = re.compile(r"\s+")


# ============================================================================
# DATA LOADING
# ============================================================================

def load_embeddings(path: str, subset: str = "full") -> tuple:
    """
    Load gzipped CSV with stringified embeddings and convert to numpy array.

    Args:
        path: Path to the gzipped CSV file
        subset: Data subset - 'full', 'scientific_paper', or 'diagnostic_report'

    Returns:
        Tuple of (DataFrame, embeddings array)
    """
    print(f"📂 Loading dataset from {path} ...")
    df = pd.read_csv(path, compression="gzip")
    print(f"✅ Loaded {len(df)} rows")

    # Apply subset filter if needed
    if subset != "full":
        if "type" not in df.columns:
            raise ValueError("❌ Dataset must have a 'type' column for subsetting.")
        df = df[df["type"] == subset]
        print(f"🔍 Subset to {subset}: {len(df)} rows")

    # Parse embeddings from string to numpy array
    tqdm.pandas(desc="Parsing embeddings")
    embeddings = np.stack(
        df["embedding"].progress_apply(lambda x: np.array(ast.literal_eval(x), dtype=np.float32))
    )

    # Create combined text field for analysis
    df = create_combined_text(df)

    return df, embeddings


def create_combined_text(df: pd.DataFrame) -> pd.DataFrame:
    """Combine title and abstract into a single text field."""
    if "title" in df.columns and "abstract" in df.columns:
        df["combined_text"] = df["title"].fillna("") + ". " + df["abstract"].fillna("")
    elif "title" in df.columns:
        df["combined_text"] = df["title"]
    else:
        raise ValueError("❌ Dataset must contain at least a 'title' column.")

    return df


# ============================================================================
# TEXT CLEANING
# ============================================================================

def clean_text_for_tfidf(text: str) -> str:
    """
    Clean text by removing HTML, normalizing Unicode, and filtering boilerplate.

    Args:
        text: Raw text string

    Returns:
        Cleaned text string
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # Remove HTML/XML tags
    text = BeautifulSoup(text, "lxml").get_text(separator=" ")

    # Normalize Unicode and convert fullwidth to halfwidth
    text = unicodedata.normalize("NFKC", text)
    text = mojimoji.zen_to_han(text, kana=False)

    # Remove URLs, emails, and file references
    text = URL_PATTERN.sub(" ", text)
    text = EMAIL_PATTERN.sub(" ", text)
    text = FILE_PATTERN.sub(" ", text)

    # Lowercase for English text
    text = text.lower()

    # Keep only useful characters (Japanese, English, digits)
    text = NON_USEFUL_CHARS.sub(" ", text)

    # Remove boilerplate terms
    boilerplate_pattern = r"\b(" + "|".join(map(re.escape, BOILERPLATE_WORDS)) + r")\b"
    text = re.sub(boilerplate_pattern, " ", text)

    # Remove isolated single letters
    text = SINGLE_LETTER.sub(" ", text)

    # Normalize whitespace
    text = WHITESPACE.sub(" ", text).strip()

    return text


def clean_dataframe_for_tfidf(df: pd.DataFrame, text_col: str = "combined_text") -> pd.DataFrame:
    """Apply text cleaning to specified column in dataframe."""
    df["cleaned_for_tfidf"] = df[text_col].astype(str).map(clean_text_for_tfidf)
    return df


# ============================================================================
# DIMENSIONALITY REDUCTION
# ============================================================================

def perform_umap(embeddings: np.ndarray, n_neighbors: int = 100, min_dist: float = 0.1,
                 n_components: int = 10, random_state: int = 42) -> np.ndarray:
    """
    Apply UMAP dimensionality reduction to embeddings.

    Args:
        embeddings: Input embedding vectors
        n_neighbors: Number of neighbors for UMAP
        min_dist: Minimum distance for UMAP
        n_components: Target dimensionality
        random_state: Random seed for reproducibility

    Returns:
        Reduced embeddings
    """
    umap_model = UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=random_state
    )
    return umap_model.fit_transform(embeddings)


# ============================================================================
# CLUSTERING
# ============================================================================

def perform_hdbscan(embeddings: np.ndarray, min_cluster_size: int = 300,
                    min_samples: int = 10, epsilon: float = 0.3) -> np.ndarray:
    """
    Apply HDBSCAN clustering to embeddings.

    Args:
        embeddings: Input vectors to cluster
        min_cluster_size: Minimum cluster size
        min_samples: Minimum samples for core points
        epsilon: Cluster selection epsilon

    Returns:
        Cluster labels (-1 for outliers)
    """
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=epsilon,
        cluster_selection_method='leaf'
    )
    return clusterer.fit_predict(embeddings)


# ============================================================================
# CLUSTER LABELING
# ============================================================================

def extract_top_tfidf_terms(texts: list, top_n: int = 10) -> list:
    """Extract top TF-IDF terms from a collection of texts."""
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    x = vectorizer.fit_transform(texts)
    tfidf_sum = np.asarray(x.sum(axis=0)).ravel()
    terms = np.array(vectorizer.get_feature_names_out())
    top_indices = np.argsort(tfidf_sum)[::-1][:top_n]
    return terms[top_indices].tolist()


def generate_llm_cluster_name(keywords: str, llm_model: str = "qwen3:1.7b") -> str:
    """
    Generate a semantic cluster name using Ollama LLM.

    Args:
        keywords: Comma-separated keywords representing the cluster
        llm_model: Ollama model to use

    Returns:
        Generated topic name
    """
    prompt = f"""You are a research topic summarizer.
The following keywords are representative of a cluster of academic papers:
{keywords}

Suggest a concise, human-readable topic name (2–4 words),
e.g. "Cancer Biology", "Antarctic Climate Studies", "Japanese Linguistics".
Only return the topic name."""

    try:
        response = ollama.chat(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}]
        )
        topic_name = response["message"]["content"].strip()
        # Clean up potential thinking tags
        topic_name = topic_name.replace("<think>", "").split("</think>")[-1].strip()
        return topic_name
    except Exception as e:
        print(f"⚠️ Ollama naming failed: {e}")
        return None


def name_clusters(df: pd.DataFrame, labels: np.ndarray, text_col: str = "combined_text",
                  method: str = "llm", top_n_words: int = 10,
                  llm_model: str = "qwen3:1.7b") -> tuple:
    """
    Assign meaningful names to clusters using TF-IDF or LLM.

    Args:
        df: DataFrame containing text data
        labels: Cluster labels from clustering algorithm
        text_col: Column name containing text to analyze
        method: Naming method - 'tfidf' (keyword-based) or 'llm' (semantic)
        top_n_words: Number of top TF-IDF terms to extract
        llm_model: Ollama model to use for LLM naming

    Returns:
        Tuple of (updated DataFrame, cluster names dict)
    """
    df["cluster_id"] = labels
    cluster_names = {}

    for cluster_id in tqdm(sorted(set(labels)), desc="🧠 Naming clusters"):
        # Handle outliers
        if cluster_id == -1:
            cluster_names[cluster_id] = "Outliers"
            continue

        # Get texts for this cluster
        cluster_texts = df.loc[df["cluster_id"] == cluster_id, text_col].dropna().tolist()
        if not cluster_texts:
            cluster_names[cluster_id] = "Unknown"
            continue

        # Extract representative keywords via TF-IDF
        top_terms = extract_top_tfidf_terms(cluster_texts, top_n_words)

        # Use TF-IDF method
        if method == "tfidf":
            cluster_names[cluster_id] = " / ".join(top_terms[:3])
            continue

        # Use LLM method
        keywords = ", ".join(top_terms)
        llm_name = generate_llm_cluster_name(keywords, llm_model)
        cluster_names[cluster_id] = llm_name or " / ".join(top_terms[:3])

    df["cluster_name"] = df["cluster_id"].map(cluster_names)
    return df, cluster_names


# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

def run_single_configuration(df: pd.DataFrame, embeddings: np.ndarray, config: dict,
                             name_method: str = "llm", savefig: bool = False) -> dict:
    """
    Run clustering pipeline with a single configuration.

    Args:
        df: Input dataframe
        embeddings: Embedding vectors
        config: Configuration dictionary with UMAP and HDBSCAN parameters
        name_method: Cluster naming method ('tfidf' or 'llm')
        savefig: Whether to save generated figures

    Returns:
        Dictionary containing results (df, labels, cluster_names)
    """
    # Perform dimensionality reduction
    reduced_embeddings = perform_umap(
        embeddings,
        n_neighbors=config.get("umap_neighbors", 100),
        min_dist=config.get("umap_min_dist", 0.4),
        n_components=config.get("umap_components", 10)
    )

    # Perform clustering
    labels = perform_hdbscan(
        reduced_embeddings,
        min_cluster_size=config.get("hdb_min_cluster_size", 300),
        min_samples=config.get("hdb_min_samples", 10),
        epsilon=config.get("hdb_epsilon", 0.3)
    )

    # Calculate metrics
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_outliers = list(labels).count(-1)
    print(f"📊 Clusters: {n_clusters} | 🚫 Outliers: {n_outliers}")

    # Prepare dataframe
    df_temp = df.copy()
    df_temp["cluster_id"] = labels
    df_temp = clean_dataframe_for_tfidf(df_temp, text_col="combined_text")

    # Generate cluster names
    df_labeled, cluster_names = name_clusters(
        df_temp,
        labels,
        text_col="cleaned_for_tfidf",
        method=name_method,
        llm_model="qwen3:1.7b"
    )

    # Display cluster names
    print("🏷️ Cluster names:")
    for cid, name in cluster_names.items():
        if cid != -1:
            print(f"  - {cid}: {name}")

    # Generate visualizations
    if savefig or True:  # Always attempt plotting
        try:
            umap_2d = perform_umap(embeddings, n_neighbors=15, min_dist=0.1, n_components=2)
            plot_embedding(
                umap_embeddings=umap_2d,
                labels=labels,
                neighbors=config.get("umap_neighbors"),
                cluster_size=config.get("hdb_min_cluster_size"),
                save=savefig
            )
            plot_embedding_interactive(
                df=df_labeled,
                umap_embeddings=umap_2d,
                labels=labels,
                neighbors=config.get("umap_neighbors"),
                cluster_size=config.get("hdb_min_cluster_size"),
                save=savefig
            )
        except Exception as e:
            print(f"⚠️ Skipped plotting: {e}")

    return {
        "config": config,
        "df": df_labeled,
        "labels": labels,
        "cluster_names": cluster_names
    }


def try_multiple_configurations(df: pd.DataFrame, embeddings: np.ndarray, configs: list,
                                name_method: str = "llm", savefig: bool = False) -> list:
    """
    Run clustering under multiple configurations and compare results.

    Args:
        df: Input dataframe
        embeddings: Embedding vectors
        configs: List of configuration dictionaries
        name_method: Cluster naming method ('tfidf' or 'llm')
        savefig: Whether to save generated figures

    Returns:
        List of result dictionaries, one per configuration
    """
    results = []

    for i, config in enumerate(configs, start=1):
        print(f"\n{'=' * 60}")
        print(f"⚙️  Running configuration {i}/{len(configs)}: {config}")
        print(f"{'=' * 60}")

        result = run_single_configuration(df, embeddings, config, name_method, savefig)
        results.append(result)

    return results


# ============================================================================
# I/O OPERATIONS
# ============================================================================

def save_results(results: list, output_dir: str = "data/final") -> None:
    """
    Save clustering results to compressed CSV files.

    Args:
        results: List of result dictionaries from clustering experiments
        output_dir: Directory to save output files
    """
    for i, res in enumerate(results, start=1):
        neighbors = res['config']['umap_neighbors']
        cluster_size = res['config']['hdb_min_cluster_size']
        filename = f"{output_dir}/clusters_config_{neighbors}neighbors_{cluster_size}cluster_size.csv.gz"

        print(f"💾 Saving {filename}")
        res["df"].to_csv(filename, index=False, compression="gzip")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Execute the clustering pipeline."""
    # Configuration
    SAVE_RESULTS = False
    SAVE_FIGURES = False
    INPUT_PATH = "data/final/data.csv.gz"

    # Load data
    df, embeddings = load_embeddings(path=INPUT_PATH, subset="scientific_paper")

    # Define experimental configurations
    configs = [
        {"umap_neighbors": 10, "hdb_min_cluster_size": 50},
        # {"umap_neighbors": 30, "hdb_min_cluster_size": 400},
        # {"umap_neighbors": 50, "hdb_min_cluster_size": 500},
        # {"umap_neighbors": 75, "hdb_min_cluster_size": 1000},
        # {"umap_neighbors": 100, "hdb_min_cluster_size": 1250},
        # {"umap_neighbors": 125, "hdb_min_cluster_size": 1500}
    ]

    # Run experiments
    # Choose naming method: "tfidf" (fast) or "llm" (Ollama semantic)
    results = try_multiple_configurations(
        df, embeddings, configs,
        name_method="tfidf",
        savefig=SAVE_FIGURES
    )

    # Save results if configured
    if SAVE_RESULTS:
        save_results(results)

    print("\n✅ All experiments complete!")


if __name__ == "__main__":
    main()
