import torch
import argparse
from enum import Enum
from typing import List
from pydantic import BaseModel
from kamino import ProdModel, Kamino
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.nn import Softmax

# [API]
# klue/bert-base 모델의 MSL : 512
MODEL_MSL = 512
# 모델의 경로
_MODEL_DIR_PREFIX = "/app/output/model"
# label_classes 경로
_LABEL_CLASS = "/app/label_classes/classes.txt"

# label_classes, 모델 등 정보 로딩
index2class = []
models_dir = _MODEL_DIR_PREFIX
with open(_LABEL_CLASS, "r") as class_file:
    label_items = [line.rstrip("\n") for line in class_file.readlines()]
index2class.append(label_items)

print(
    f"len(index2class): {len(index2class)}, models_dir: {models_dir}"
)

class Request(BaseModel):
    # API 전체 입력 파라미터 정의
    # print(Row.schema_json(indent=2))
    paragraphs: List[str]

    class Config:
        allow_population_by_field_name = True


class LabelAndScore(BaseModel):
    label: str
    score: float

class Response(BaseModel):
    # API 전체 출력 파라미터 정의
    # print(Response.schema_json(indent=2))
    answers: List[LabelAndScore]


class SeJongClassifier(ProdModel):
    input_model = Request
    output_model = Response
    _SCORE_FLOAT_ROUND_DIGITS = 5  # 스코어 값에 대해 반올림하여 소수점 N자리까지 표시

    def __init__(self, models_dir, args, device=0):
        self.models = []
        self.models.append(AutoModelForSequenceClassification.from_pretrained(models_dir))
        self.tokenizer = []
        self.tokenizer.append(AutoTokenizer.from_pretrained(models_dir, use_fast=True))

        self.device = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
        self.args = args
        print(
            f"models_dir: {models_dir}, device:{self.device, device}, "
        )
        
        for model in self.models:
            model.to(self.device)

    def predict(self, data):
        print(f"[predict] type(data): {type(data)}, data: {data}")

        softmax_func = Softmax(dim=-1)
        answers = []
        for paragraph in data.paragraphs:
            for index, model in enumerate(self.models):
                max_len = MODEL_MSL
                input = self.tokenizer[index](paragraph, truncation=True, max_length=max_len)
                # tokenizer 결과 예
                # input = {'input_ids': [2, 22141, 4058, 3, 16129, 28358, 27392, 26334, 4056, 3],
                #          'token_type_ids': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                #          'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}

                inputs = {}
                for (k1, v1) in input.items():
                    inputs[k1] = torch.tensor([v1])
                    assert inputs[k1].shape[-1] <= max_len, f"{inputs[k1].shape[-1]} <= {max_len}"
                # print(f'[predict] inputs: {inputs}, input: {input}')

                # model(**inputs) 결과
                # SequenceClassifierOutput(loss=None, logits=tensor([[-1.8032,  0.9718, -1.3259, -1.5659,  3.4244]],
                #   grad_fn=<AddmmBackward>), hidden_states=None, attentions=None)
                logits = model(**inputs).logits
                logit = logits[0].data  # batch size가 1이므로 첫번째만 사용
                max_arg = int(torch.argmax(logit))
                score = softmax_func(logit).data
                max_score = round(score[max_arg].item(), SeJongClassifier._SCORE_FLOAT_ROUND_DIGITS)
                max_label = index2class[index][max_arg]
                answers.append(LabelAndScore(label=max_label, score=max_score))

                print(
                    f"[predict] index: {index}, max_label:{max_label}, "
                    f"max_arg:{max_arg}, max_score:{max_score}, logit: {logit}"
                )
            
        return Response(answers=answers)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serving classifier with Kamino.")

    # model
    parser.add_argument(
        "--max_batch_size_per_worker", type=int, default=0, help="maximum number of passages handled by each worker"
    )

    # kamino
    parser.add_argument("--frontdoor_port", default=4242, type=int, help="front door port for Kamino")
    parser.add_argument("--queue_port", default=4646, type=int, help="queue port for Kamino")
    parser.add_argument("--gather_port", default=3642, type=int, help="gather port for Kamino")
    parser.add_argument("--result_port", default=3636, type=int, help="result port for Kamino")
    parser.add_argument("--skip_port", default=4236, type=int, help="skip port for Kamino")
    parser.add_argument("--control_port", default=4332, type=int, help="control port for Kamino")
    parser.add_argument("--port", default=5000, type=int, help="port for Kamino")
    parser.add_argument("--local_worker_devices", type=int, nargs="+", default=[-1])
    parser.add_argument("--kamino_logger", type=str, nargs="?", const="debug", default="debug", help="kamino logger")
    parser.add_argument(
        "--reinit_worker_on_exception_while_predict",
        type=str2bool,
        nargs="?",
        default=True,
        help="whether to reinitialize the worker if exception occurs while predicting",
    )

    args = parser.parse_args()

    m = Kamino(
        SeJongClassifier,
        local_worker_devices=args.local_worker_devices,
        batch_size=args.max_batch_size_per_worker,
        frontdoor_port=args.frontdoor_port,
        queue_port=args.queue_port,
        gather_port=args.gather_port,
        result_port=args.result_port,
        skip_port=args.skip_port,
        control_port=args.control_port,
        port=args.port,
        models_dir=models_dir,
        logger=args.kamino_logger,
        reinit_worker_on_exception_while_predict=args.reinit_worker_on_exception_while_predict,
        args=args,
    )

    m.run()
