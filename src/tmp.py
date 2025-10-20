from typing import Tuple, List
from categories import categories, keywords
from sentence_transformers import SentenceTransformer, util
import subprocess
import ollama
import os

# --- Initialize embeddings ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
embedder = SentenceTransformer('all-MiniLM-L6-v2', device="cpu")
category_embeddings = embedder.encode(categories, convert_to_tensor=True)


CONFIDENCE_THRESHOLD = 0.6


# --- Lexical Agent ---
def lexical_agent(text: str) -> str:
    scores = {cat: 0 for cat in categories}
    text_lower = text.lower()
    for cat, keys in keywords.items():
        for k in keys:
            if k in text_lower:
                scores[cat] += 1
    return max(scores, key=scores.get)


# --- Contextual Agent ---
def contextual_agent(text: str) -> str:
    text_emb = embedder.encode(text, convert_to_tensor=True)
    scores = util.cos_sim(text_emb, category_embeddings)[0]
    return categories[scores.argmax()]


# --- Logic Agent ---
def logic_agent(text: str) -> str:
    text_lower = text.lower()
    if "ai" in text_lower or "robotics" in text_lower:
        return "Engineering and Technology"
    if "medicine" in text_lower or "clinical" in text_lower:
        return "Medical and Health Sciences"
    if "crop" in text_lower or "soil" in text_lower:
        return "Agricultural Sciences"
    return "Social Sciences"


# --- LLM Agent ---
def llm_agent(text: str) -> str:
    categories_list = [
        "Natural Sciences",
        "Engineering and Technology",
        "Medical and Health Sciences",
        "Agricultural Sciences",
        "Social Sciences",
        "Humanities"
    ]

    prompt = (
        "Classify the following abstract into one of these categories: "
        f"{', '.join(categories_list)}.\n\n"
        f"Abstract: \"{text}\"\n\n"
        "Very important: Answer with only the category name, no other words — "
        "just one of the provided categories."
    )

    try:
        # Use Ollama's Python API instead of subprocess
        response = ollama.chat(
            model='qwen3:1.7b',
            messages=[{'role': 'user', 'content': prompt}]
        )

        output = response['message']['content'].strip()

    except Exception as e:
        print(f"⚠️ Ollama error: {e}")
        return "Unknown"

    # Post-process model output
    label = output.split("\n")[-1].strip()
    categories_set = set(categories_list)

    for cat in categories_set:
        if cat.lower() in label.lower():
            return cat

    return "Unknown"


# --- Consensus Agent ---
def consensus_agent(agent_outputs: List[str]) -> str:
    return max(set(agent_outputs), key=agent_outputs.count)


# --- Explainability Agent ---
def explainability_agent(text: str, final_label: str, agent_outputs: List[str]) -> str:
    explanation = (
        f"Final label: {final_label}\n"
        f"Agent suggestions: {agent_outputs}\n"
        f"Lexical matches, embeddings, logic rules, and LLM reasoning contributed to this decision."
    )
    return explanation


# --- Multi-Agent Classifier with LLM ---
def multi_agent_classify_tmp(text: str):

    agent_outputs = [
        lexical_agent(text),
        contextual_agent(text),
        logic_agent(text),
        llm_agent(text)
    ]

    final_label = consensus_agent(agent_outputs)

    return final_label
