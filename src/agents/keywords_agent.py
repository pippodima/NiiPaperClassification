from src.categories import categories, keywords


def keyword_agent(text):
    scores = {cat: 0 for cat in categories}
    text_lower = text.lower()
    for cat, keys in keywords.items():
        for k in keys:
            if k in text_lower:
                scores[cat] += 1
    return max(scores, key=scores.get)
