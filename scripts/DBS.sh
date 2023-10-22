export CUDA_VISIBLE_DEVICES=0

python src/main.py --anormly_ratio 4 --num_epochs 10  --batch_size 1024  --mode train --dataset DBS --data_path /home/shpark/Anomaly_Explanation/dataset/dbsherlock/processed/tpce_3000 --input_c 200 --output_c 200 --win_size 30 --step_size 30 --cause "all"
python src/main.py --anormly_ratio 4 --num_epochs 10  --batch_size 1024  --mode test  --dataset DBS --data_path /home/shpark/Anomaly_Explanation/dataset/dbsherlock/processed/tpce_3000 --input_c 200 --output_c 200 --win_size 30 --step_size 30 --pretrained_model 20 --cause "all"
