MODEL_PATH="output/klue/bert-base"
TEST_TSV_PATH="data/preprocess/test.tsv"
MERGE_TSV_PATH="data/raw/merge.tsv"

python -m inference \
  --model_path ${MODEL_PATH} \
  --test_tsv_path ${TEST_TSV_PATH} \
  --merge_tsv_path ${MERGE_TSV_PATH}
