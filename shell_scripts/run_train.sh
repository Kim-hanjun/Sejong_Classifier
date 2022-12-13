export OMP_NUM_THREADS=8
model_name_or_path="klue/bert-base"
python /elder_speech_emotion_classfication/train.py \
  --output_dir "output/${model_name_or_path}" \
  --train_csv_path "data/preprocess/train.tsv" \
  --test_csv_path "data/preprocess/test.tsv" \
  --num_train_epochs "1" \
  --per_device_train_batch_size "32" \
  --per_device_eval_batch_size "32" \
  --gradient_accumulation_steps "1"\
  --model_name_or_path ${model_name_or_path} \
  --evaluation_strategy "steps" \
  --save_strategy "steps" \
  --logging_strategy "steps" \
  --logging_steps "50" \
  --eval_step "100" \
  --save_step "100" \
  --save_total_limit "1" \
  --logging_strategy "steps" \
  --metric_for_best_model "micro_f1" \
  --learning_rate "2e-5" \
  --dataloader_num_workers "4" \
  --label_names "labels" \
  --load_best_model_at_end \
  --do_eval \
  --no_cuda 
