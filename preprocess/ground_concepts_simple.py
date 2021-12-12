import configparser
import json
import csv
import spacy
from spacy.matcher import Matcher
import sys
import timeit
from tqdm import tqdm
import numpy as np
import multiprocessing
import sys

config = configparser.ConfigParser()
config.read("paths.cfg")

with open(config["paths"]["blacklisted_words"], "r", encoding="utf8") as f:
    blacklist = [l.strip() for l in list(f.readlines())]
blacklist = set(blacklist)


concept_vocab = set()
with open(config["paths"]["concept_vocab"], "r", encoding="utf8") as f:
    cpnet_vocab = [l.strip() for l in list(f.readlines())]
cpnet_vocab = set([c.replace("_", " ") for c in cpnet_vocab])

# nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'textcat'])
nlp = spacy.load("en_core_web_sm")


def hard_ground(sent):
    global cpnet_vocab, model_vocab
    sent = sent.lower()
    doc = nlp(sent)
    res = set()
    for t in doc:
        if (
            t.lemma_ in cpnet_vocab
            and t.lemma_ not in blacklist
            and t.lemma_ in model_vocab
        ):
            if t.pos_ == "NOUN" or t.pos_ == "VERB":
                res.add(t.lemma_)
    # sent = "_".join([t.text for t in doc])
    # if sent in cpnet_vocab and sent not in blacklist and sent in model_vocab:
    #    res.add(sent)
    return res


def match(input):
    return match_mentioned_concepts(input[0], input[1])


def match_mentioned_concepts(sents, answers):
    assert len(sents) == len(answers)
    # global nlp
    # matcher = load_matcher(nlp)

    res = []
    # print("Begin matching concepts.")

    for sid, s in tqdm(
        enumerate(sents), total=len(sents)
    ):  # , desc="grounding batch_id:%d"%batch_id):
        a = answers[sid]

        all_concepts = hard_ground(s + " " + a)
        # print(all_concepts)
        question_concepts = hard_ground(s)
        # print(question_concepts)

        answer_concepts = all_concepts - question_concepts

        res.append(
            {
                "sent": s,
                "ans": a,
                "qc": list(question_concepts),
                "ac": list(answer_concepts),
            }
        )
    return res


def lemmatize(nlp, concept):

    doc = nlp(concept.replace("_", " "))
    lcs = set()

    lcs.add("_".join([token.lemma_ for token in doc]))  # all lemma
    return lcs


def load_matcher(nlp):
    config = configparser.ConfigParser()
    config.read("paths.cfg")

    matcher = Matcher(nlp.vocab)
    for concept in cpnet_vocab:
        matcher.add(concept, None, [{"LEMMA": concept}])

    return matcher


def grounding_sentences(src, tgt, type, path):

    # nlp.add_pipe(nlp.create_pipe('sentencizer'))
    res = match_mentioned_concepts(sents=src, answers=tgt)
    # res[0]["qc"] = reduce_redundant_concepts(res[0]["qc"])

    with open(path + "/{}/concepts_nv.json".format(type), "w") as f:
        for line in res:
            json.dump(line, f)
            f.write("\n")


def read_csv(data_path="train/source.csv"):
    data = []
    with open(data_path, "r") as f:
        reader = csv.reader(f, delimiter=",")
        for i, row in enumerate(reader):
            data.append(" ".join(row[1:]))
    return data


def read_model_vocab(data_path):
    global model_vocab
    vocab_dict = json.loads(open(data_path, "r").readlines()[0])
    model_vocab = []
    for tok in vocab_dict.keys():
        model_vocab.append(tok[1:])
    print(len(model_vocab))


if __name__ == "__main__":
    dataset = sys.argv[1]
    assert dataset in ["wizard", "story", "eg", "anlg"]
    DATA_PATH = config["paths"][dataset + "_dir"]

    SRC_FILE = DATA_PATH + "/{}/source.csv"
    TGT_FILE = DATA_PATH + "/{}/target.csv"

    read_model_vocab(config["paths"]["gpt2_vocab"])

    TYPE = "train"
    src = read_csv(SRC_FILE.format(TYPE))
    tgt = read_csv(TGT_FILE.format(TYPE))
    grounding_sentences(src, tgt, TYPE, DATA_PATH)

    TYPE = "dev"
    src = read_csv(SRC_FILE.format(TYPE))
    tgt = read_csv(TGT_FILE.format(TYPE))
    grounding_sentences(src, tgt, TYPE, DATA_PATH)

    TYPE = "test"
    src = read_csv(SRC_FILE.format(TYPE))
    tgt = read_csv(TGT_FILE.format(TYPE))
    grounding_sentences(src, tgt, TYPE, DATA_PATH)
