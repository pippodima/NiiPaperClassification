from transformers import pipeline
from src.categories import categories
from src.utils.device_check import get_device


device_name = get_device()
# Transformers pipeline: -1 = CPU, 0 = GPU. Use -1 for Mac M1.
device_arg = 0 if device_name == "cuda" else -1

zero_shot = pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-3", device=device_arg)


def zero_shot_agent(text):
    result = zero_shot(text, candidate_labels=categories)
    return result['labels'][0]
