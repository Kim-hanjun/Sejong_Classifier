input_dir='/data/sejong_data/output'
output_dir='/home/ncloud/workspace/repo/Sejong_Classifier/data/raw'
mkdir -p "$output_dir"

python -m txt_merge.txt_merge \
    --input_dir "$input_dir" \
    --output_dir "$output_dir"
