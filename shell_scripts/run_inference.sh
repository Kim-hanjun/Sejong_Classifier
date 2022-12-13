model_path="/output/klue/bert-base"
test_tsv_path="/data/preprocess/test.tsv"

python -m inference \
  --model_path ${model_path} \
  --test_tsv_path ${test_tsv_path}