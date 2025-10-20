from sentence_transformers import SentenceTransformer, util
from src.categories import categories
from src.utils.device_check import get_device

device_name = get_device()
# SentenceTransformer supports "cpu" and "cuda"
embedder = SentenceTransformer('all-MiniLM-L6-v2', device=device_name)
category_embeddings = embedder.encode(categories, convert_to_tensor=True)


def embedding_agent(text):
    text_emb = embedder.encode(text, convert_to_tensor=True)
    scores = util.cos_sim(text_emb, category_embeddings)[0]
    return categories[scores.argmax()]
