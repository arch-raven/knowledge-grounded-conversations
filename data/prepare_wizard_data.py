import os
import json
import csv

# import pandas as pd
source_path = "/home1/deeksha/aditya/multigen/data/wizard/{}/source.csv"
target_path = "/home1/deeksha/aditya/multigen/data/wizard/{}/target.csv"
knowledge_path = "/home1/deeksha/aditya/multigen/data/wizard/{}/knowledge.csv"


def write_csv_(list_of_texts, save_to_path):
    os.makedirs(os.path.dirname(save_to_path), exist_ok=True)
    print(f"=> Writing to {save_to_path}")
    data = list(enumerate(list_of_texts))
    with open(save_to_path, "w") as f:
        csvwriter = csv.writer(f)
        csvwriter.writerows(data)


def prepare_data(jsonl_file, stage: str):
    assert stage in ["train", "dev", "test"]
    source_sentences = []
    knowledge_sentences = []
    target_sentences = []

    with open(jsonl_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f.readlines()):
            data = json.loads(line)
            history = data["history"]
            response = data["response"]
            knowledge = data["knowledge"][0]  # first knowledge is Knowledge used
            assert (
                type(history) == list
                and type(response) == str
                and type(knowledge) == str
            )
            history = [" ".join(text.split()) for text in history]
            response = " ".join(response.split())
            knowledge = knowledge.split("__knowledge__")[-1].strip()

            source_sentences.append(" ".join(history))
            target_sentences.append(response)
            knowledge_sentences.append(knowledge)

    write_csv_(source_sentences, source_path.format(stage))
    write_csv_(target_sentences, target_path.format(stage))
    write_csv_(knowledge_sentences, knowledge_path.format(stage))


if __name__ == "__main__":
    prepare_data(
        jsonl_file="/home1/deeksha/aditya/KnowledGPT/wizard_of_wikipedia/data/train.jsonl",
        stage="train",
    )
    prepare_data(
        jsonl_file="/home1/deeksha/aditya/KnowledGPT/wizard_of_wikipedia/data/valid.jsonl",
        stage="dev",
    )
    prepare_data(
        jsonl_file="/home1/deeksha/aditya/KnowledGPT/wizard_of_wikipedia/data/test_seen.jsonl",
        stage="test",
    )
