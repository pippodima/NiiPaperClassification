from agents.keywords_agent import keyword_agent
from agents.zero_shot_agent import zero_shot_agent
from agents.embedding_agent import embedding_agent
from data.utils import get_n_rows_datasets
from utils.metrics import get_metrics
from tqdm import tqdm
from tmp import multi_agent_classify_tmp


def classify_abstract(text):
    agents = [keyword_agent(text),
              zero_shot_agent(text),
              embedding_agent(text)]
    final_category = max(set(agents), key=agents.count)
    return final_category


if __name__ == "__main__":
    tqdm.pandas()
    df = get_n_rows_datasets(csv_path="data/final/data.csv", rows=20)
    df["predicted_category"] = df['abstract'].progress_apply(multi_agent_classify_tmp)
    acc, report = get_metrics(df)

    print("acc: ", acc)
    print(report)
