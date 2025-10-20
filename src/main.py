from agents.keywords_agent import keyword_agent
from agents.zero_shot_agent import zero_shot_agent
from agents.embedding_agent import embedding_agent


def classify_abstract(text):
    agents = [keyword_agent(text), zero_shot_agent(text), embedding_agent(text)]
    final_category = max(set(agents), key=agents.count)
    return final_category


if __name__ == "__main__":
    abstract = "This paper explores machine learning techniques in robotics and electronics applications."
    category = classify_abstract(abstract)
    print("Predicted Category:", category)
