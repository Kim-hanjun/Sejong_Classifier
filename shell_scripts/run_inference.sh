MODEL_PATH="output/klue/bert-base"
TEST_TSV_PATH="data/preprocess/test.tsv"

python -m inference \
  --model_path ${MODEL_PATH} \
  --test_tsv_path ${TEST_TSV_PATH}