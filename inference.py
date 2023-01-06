from transformers import pipeline, AutoTokenizer
from literal import RAW_DATA, RAW_LABELS
import pandas as pd
from tqdm import tqdm
import argparse

def main(args) :
    test_tsv_path = args.test_tsv_path
    model_path = args.model_path

    # tsv 파일을 sep 탭으로 불러온다
    df = pd.read_csv(test_tsv_path, sep='\t')

    # 불러온 데이터중 Data 부분만 가져와서 리스트로 변경
    df_Datas = df[RAW_DATA].values.tolist()
    df_answers = df[RAW_LABELS]
    df_origin = df["원본파일명"]

    # 텍스트 길이가 max를 넘어서서 토크나이저 설정 변경
    sent  = df_Datas
    # sent  = df_Datas[:10]
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenized_texts = tokenizer(sent,truncation=True, max_length=tokenizer.model_max_length)['input_ids']
    truncated_sents = []
    for tokenized_text in tqdm(tokenized_texts):
        truncated_sents.append(tokenizer.decode(tokenized_text,skip_special_tokens=True))

    # pipe 라이브러리로 label과 점수를 return 받는다
    pipe = pipeline("text-classification",model=model_path)
    result = pipe(truncated_sents)
    # pipe 에 넣은 텍스트와 result를 dataframe 형식으로 변경한다
    sent_dataframe = pd.DataFrame(sent, columns=["텍스트"])
    # 원래 label 과 score 였던 열 이름을 예측, 스코어로 변경
    result_dataframe = pd.DataFrame(result).rename(columns={"label": "예측", "score": "스코어"})
    # 변경된 dataframe 을 열 방향으로 합친다
    total_dataframe = pd.concat([sent_dataframe, result_dataframe, df_answers, df_origin], axis=1)
    total_dataframe.to_excel("법무법인 세종 테스트 결과.xlsx", index=False, encoding='cp949')

if __name__=="__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--test_tsv_path", type=str, default=None)
    args = parser.parse_args()
    main(args=args)


