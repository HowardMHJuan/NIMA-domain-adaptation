# project_dir="/tmp/CC2020"
project_dir="saved"

for img in "1717" "6174" "39062"; do
    echo "$project_dir/${img}_crop.png"
    python3 test.py \
        --model "$project_dir/model.pth" \
        --img "$project_dir/${img}_crop.png"
done
