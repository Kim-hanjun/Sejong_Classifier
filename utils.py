import random
from os import PathLike
from typing import Any, Dict

import evaluate
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import PreTrainedTokenizer

from literal import LABEL2ID, LABELS, RAW_DATA, RAW_LABELS, SENT


def seed_everything(random_seed: int) -> None:
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def get_dataset(csv_path: PathLike) -> Dataset:
    df = pd.read_csv(csv_path,sep='\t')
    sents = df[RAW_DATA].to_list()
    labels = df[RAW_LABELS].to_list()

    return Dataset.from_dict({SENT: sents, RAW_LABELS: labels})


def preprocess_dataset(raw: Dict[str, Any], tokenizer: PreTrainedTokenizer, return_token_type_ids: bool):
    tokenized_text = tokenizer(
        raw[SENT], truncation=True, max_length=tokenizer.model_max_length, return_token_type_ids=return_token_type_ids
    )
    for key, item in tokenized_text.items():
        raw[key] = item
    if RAW_LABELS in raw:
        raw[LABELS] = LABEL2ID[raw[RAW_LABELS]]
    return raw


def compute_metrics(pred):
    references = pred.label_ids
    predictions = np.argmax(pred.predictions, axis=-1)
    scores = pred.predictions
    accuracy = evaluate.load("accuracy").compute
    f1 = evaluate.load("f1").compute
    recall = evaluate.load("recall").compute
    precision = evaluate.load("precision").compute
    roc_auc = evaluate.load("roc_auc", "multiclass").compute
    metric = {
        "accuracy": accuracy(references=references, predictions=predictions)["accuracy"],
        "micro_f1": f1(references=references, predictions=predictions, average="micro")["f1"],
        "micro_recall": recall(references=references, predictions=predictions, average="micro")["recall"],
        "micro_precision": precision(references=references, predictions=predictions, average="micro")["precision"],
        "roc_auc_ovr": roc_auc(references=references, prediction_scores=scores, multi_class="ovr")["roc_auc"],
    }

    return metric


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return F.softmax(logits, dim=-1, dtype=torch.float32)
