import os
import torch
import argparse
from typing import List, Optional
from pydantic import BaseModel, Field
from kamino import ProdModel, Kamino
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.nn import Softmax, Hardsigmoid
from transformers_addons import BertKoreanMecabTokenizer, ElectraForMultiLabelClassification

from data_info import OutputLabel, CLUE4_MODEL_MSL, EMOTION7_MODEL_MSL

_LABEL_CLASS_DIR_CLUE4 = "/app/custom/gwanghwasidae_clue4_datasets/label_classes"
_LABEL_CLASS_DIR_EMOTION7 = "/app/custom/gwanghwasidae_emotion7_datasets/label_classes"
_MODEL_DIR_PREFIX = "/app/models/"

# label (name&key), label_classes, 모델 등 정보 로딩
LABEL_KEYS = []
LABEL_VALUES = []
index2class = []
models_dir = []

# 4단 모델
LABEL_KEYS.append(OutputLabel.CLUE4.name)
LABEL_VALUES.append(OutputLabel.CLUE4.value)
label_key_clue4 = OutputLabel.CLUE4.name
with open(os.path.join(_LABEL_CLASS_DIR_CLUE4, label_key_clue4 + '.txt'), 'r') as class_file:
    label_items = [line.rstrip("\n") for line in class_file.readlines()]
index2class.append(label_items)
models_dir.append(_MODEL_DIR_PREFIX + OutputLabel.CLUE4.value)

print(f'LABEL_KEYS: {LABEL_KEYS}, LABEL_VALUES: {LABEL_VALUES}, '
      f'len(index2class): {len(index2class)}, models_dir: {models_dir}')

# 7정 모델
_DEF_THRESHOLD = 0.4
LABEL_KEYS.append(OutputLabel.EMOTION7.name)
LABEL_VALUES.append(OutputLabel.EMOTION7.value)
label_key_emotion7 = OutputLabel.EMOTION7.name
with open(os.path.join(_LABEL_CLASS_DIR_EMOTION7, label_key_emotion7 + '.txt'), 'r') as class_file:
    label_items = [line.rstrip("\n") for line in class_file.readlines()]
index2class.append(label_items)
models_dir.append(_MODEL_DIR_PREFIX + OutputLabel.EMOTION7.value)

print(f'LABEL_KEYS: {LABEL_KEYS}, LABEL_VALUES: {LABEL_VALUES}, '
      f'len(index2class): {len(index2class)}, models_dir: {models_dir}')


class Request(BaseModel):
    # API 전체 입력 파라미터 정의
    # print(Row.schema_json(indent=2))
    paragraphs: List[str]
    threshold: Optional[float] = _DEF_THRESHOLD

    class Config:
        allow_population_by_field_name = True


class LabelAndScore(BaseModel):
    label: str
    score: float


class ResponseItem(BaseModel):
    # 각각의 paragraph에 대한 응답 단위 정의
    # 변수명을 OutputLabel 값으로 설정 불가하여 OutputLabel 내의 이름과 동일하게 설정
    CLUE4: LabelAndScore = Field(..., alias=OutputLabel.CLUE4.value)
    EMOTION7: List[LabelAndScore] = Field(..., alias=OutputLabel.EMOTION7.value)

    class Config:
        allow_population_by_field_name = True


class Response(BaseModel):
    # API 전체 출력 파라미터 정의
    # print(Response.schema_json(indent=2))
    answers: List[ResponseItem]


class GwanghwasidaeClassifier(ProdModel):
    input_model = Request
    output_model = Response
    _SCORE_FLOAT_ROUND_DIGITS = 5  # 스코어 값에 대해 반올림하여 소수점 N자리까지 표시

    def __init__(self, models_dir, args, device=0):
        self.models = []
        self.models.append(AutoModelForSequenceClassification.from_pretrained(models_dir[0]))
        self.models.append(ElectraForMultiLabelClassification.from_pretrained(models_dir[1]))

        self.tokenizer = []
        self.tokenizer.append(AutoTokenizer.from_pretrained(models_dir[0], use_fast=True))
        self.tokenizer.append(BertKoreanMecabTokenizer.from_pretrained(models_dir[1], do_lower_case=True, spacing=False,
                                                                       joining=True))

        self.device = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
        self.args = args
        self.emotion7_score_weight_list = [float(weight) for weight in args.emotion7_score_weights.split(',')]
        print(f'models_dir: {models_dir}, device:{self.device, device}, '
              f'emotion7_score_weight_list: {self.emotion7_score_weight_list}, args: {args}')

        for model in self.models:
            model.to(self.device)

    def predict(self, data):
        print(f'[predict] type(data): {type(data)}, data: {data}')

        softmax_func = Softmax(dim=-1)
        sigmoid_func = Hardsigmoid()
        answers = []
        label_and_score_dict = {}
        for paragraph in data.paragraphs:
            for index, model in enumerate(self.models):
                model_class = LABEL_VALUES[index]
                if model_class == OutputLabel.CLUE4.value:
                    max_len = CLUE4_MODEL_MSL
                elif model_class == OutputLabel.EMOTION7.value:
                    max_len = EMOTION7_MODEL_MSL
                else:
                    print(f'[predict] Invalid model class found! index: {index}, model_class: {model_class}')

                input = self.tokenizer[index](paragraph, truncation=True, max_length=max_len)

                # tokenizer 결과 예
                # input = {'input_ids': [2, 22141, 4058, 3, 16129, 28358, 27392, 26334, 4056, 3],
                #          'token_type_ids': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                #          'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
                inputs = {}
                for (k1, v1) in input.items():
                    inputs[k1] = torch.tensor([v1])
                    assert inputs[k1].shape[-1] <= max_len, f'{inputs[k1].shape[-1]} <= {max_len}'
                # print(f'[predict] inputs: {inputs}, input: {input}')

                if model_class == OutputLabel.CLUE4.value:
                    # model(**inputs) 결과
                    # SequenceClassifierOutput(loss=None, logits=tensor([[-1.8032,  0.9718, -1.3259, -1.5659,  3.4244]],
                    #   grad_fn=<AddmmBackward>), hidden_states=None, attentions=None)
                    logits = model(**inputs).logits
                    logit = logits[0].data  # batch size가 1이므로 첫번째만 사용
                    max_arg = int(torch.argmax(logit))
                    score = softmax_func(logit).data
                    max_score = round(score[max_arg].item(), GwanghwasidaeClassifier._SCORE_FLOAT_ROUND_DIGITS)
                    max_label = index2class[index][max_arg]
                    if max_label == '미분류':
                        assert max_arg == 0, f'index for 미분류 changed. It must be zero! max_arg: {max_arg}'
                        logit_except_unclassified = logit.clone().detach()[1:]  # 미분류(0)에 대한 logit 제외
                        alt_max_arg = int(torch.argmax(logit_except_unclassified))
                        alt_score = softmax_func(logit_except_unclassified)
                        alt_max_score = round(alt_score[alt_max_arg].item(),
                                              GwanghwasidaeClassifier._SCORE_FLOAT_ROUND_DIGITS)
                        if alt_max_score >= self.args.clue4_alt_score_threshold:
                            max_arg = alt_max_arg + 1  # 미분류(0)에 대한 index 추가
                            max_score = alt_max_score
                            max_label = index2class[index][max_arg]
                            print(f'[predict] Using alternative due to {alt_max_score} >= '
                                  f'{self.args.clue4_alt_score_threshold}, alt_max_arg: {alt_max_arg}')

                    label_and_score_dict[model_class] = LabelAndScore(
                        label=max_label,
                        score=max_score)

                    print(f'[predict] index: {index}, model_class: {model_class}, max_label:{max_label}, '
                          f'max_arg:{max_arg}, max_score:{max_score}, logit: {logit}')
                elif model_class == OutputLabel.EMOTION7.value:
                    # model(**inputs) 결과
                    # (tensor([[-4.4246, -3.2022, -2.2742, -4.9601, -3.5453, -2.8777,  1.8575]],
                    #   grad_fn=<AddmmBackward>),)
                    (logits,) = model(**inputs)
                    logit = logits[0].data  # batch size가 1이므로 첫번째만 사용
                    label_and_score_dict[model_class] = []
                    score = sigmoid_func(logit).data
                    weighted_scores = []
                    for label_idx, label_score in enumerate(score):
                        weighted_score = round(label_score.item() * self.emotion7_score_weight_list[label_idx],
                                               GwanghwasidaeClassifier._SCORE_FLOAT_ROUND_DIGITS)
                        weighted_scores.append(weighted_score)
                        if weighted_score >= data.threshold:
                            label_and_score_dict[model_class].append(LabelAndScore(
                                label=index2class[index][label_idx], score=weighted_score))
                    label_and_score_dict[model_class] = sorted(
                        label_and_score_dict[model_class],
                        key=lambda answer: answer.score, reverse=True)

                    print(f'[predict] index: {index}, model_class: {model_class}, '
                          f'label_and_score_dict[model_class]: {label_and_score_dict[model_class]}, '
                          f'score: {score}, weighted_scores: {weighted_scores}')
                else:
                    print(f'[predict] Invalid model class found! index: {index}, model_class: {model_class}')

            answers.append(ResponseItem(
                CLUE4=label_and_score_dict[OutputLabel.CLUE4.value],
                EMOTION7=label_and_score_dict[OutputLabel.EMOTION7.value]
            ))

        return Response(
            answers=answers
        )


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serving classifier with Kamino.")

    # model
    parser.add_argument("--max_batch_size_per_worker", type=int, default=0,
                        help="maximum number of passages handled by each worker")

    # kamino
    parser.add_argument("--frontdoor_port", default=4242, type=int, help="front door port for Kamino")
    parser.add_argument("--queue_port", default=4646, type=int, help="queue port for Kamino")
    parser.add_argument("--gather_port", default=3642, type=int, help="gather port for Kamino")
    parser.add_argument("--result_port", default=3636, type=int, help="result port for Kamino")
    parser.add_argument("--skip_port", default=4236, type=int, help="skip port for Kamino")
    parser.add_argument("--control_port", default=4332, type=int, help="control port for Kamino")
    parser.add_argument("--port", default=5000, type=int, help="port for Kamino")
    parser.add_argument("--local_worker_devices", type=int, nargs="+", default=[-1])
    parser.add_argument("--kamino_logger", type=str, nargs="?", const="debug", default="debug",
                        help="kamino logger")
    parser.add_argument("--reinit_worker_on_exception_while_predict", type=str2bool, nargs="?", default=True,
                        help="whether to reinitialize the worker if exception occurs while predicting")

    parser.add_argument("--clue4_alt_score_threshold", default=0.4, type=float,
                        help="alternative score threshold for the label of '미분류' in '4단'")
    # EMOTION7_VALID_LABELS 순서 ('기쁨', '두려움', '분노', '사랑', '슬픔', '욕망', '증오')
    parser.add_argument("--emotion7_score_weights", default="0.9,1,1,1,1,0.8,1", type=str,
                        help="score weights for the labels in '7정'")

    args = parser.parse_args()

    m = Kamino(GwanghwasidaeClassifier,
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
               args=args)

    m.run()
