from sentence_transformers import SentenceTransformer, util
from typing import Iterator
import json

# Use a lightweight sentence-transformer
bert_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def bert(response, ground_truth):
    """
    @param response: the response from LLM
    @param ground_truth: the ground truth of the question
    @return: the cosine similarity
    """
    query_embedding = bert_model.encode(response, convert_to_tensor=True)
    text_embedding = bert_model.encode(ground_truth, convert_to_tensor=True)

    # Compute the cosine similarity between the query and text
    cosine_score = util.pytorch_cos_sim(query_embedding, text_embedding)

    return cosine_score.item()

def _parse_squad_data(raw):
    dataset = {"ki_text": [], "qas": []}

    for k_id, data in enumerate(raw["data"]):
        article = []
        for p_id, para in enumerate(data["paragraphs"]):
            article.append(para["context"])
            for qa in para["qas"]:
                ques = qa["question"]
                answers = [ans["text"] for ans in qa["answers"]]
                dataset["qas"].append(
                    {
                        "title": data["title"],
                        "paragraph_index": tuple((k_id, p_id)),
                        "question": ques,
                        "answers": answers,
                    }
                )
        dataset["ki_text"].append(
            {"id": k_id, "title": data["title"], "paragraphs": article}
        )

    return dataset


def get_training_data() -> tuple[list[str], Iterator[tuple[str, str]]]:
    with open("../input/get_training_data", "r") as file:
        data = json.load(file)
    parsed_data = _parse_squad_data(data)
    text_list = []
    # Get the knowledge Articles for at most max_knowledge, or all Articles if max_knowledge is None
    for article in parsed_data["ki_text"][:]:
        text_list.append(article["title"])
        text_list.append("\n".join(article["paragraphs"][0:]))
    questions = [
        qa["question"]
        for qa in parsed_data["qas"]
    ]
    answers = [
        qa["answers"][0]
        for qa in parsed_data["qas"]
    ]

    dataset = zip(questions, answers)

    return text_list, dataset