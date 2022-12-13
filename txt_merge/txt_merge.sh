input_dir='/data/sejong_data/output'
output_dir='/data/ncloud/repo/jun/elder_speech_emotion_classification/data/raw'

python txt_merge.py 
    --input_dir "$input_dir" \
    --output_dir "$output_dir"
