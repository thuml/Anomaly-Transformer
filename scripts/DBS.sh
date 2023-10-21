export CUDA_VISIBLE_DEVICES=0

python src/main.py --anormly_ratio 4 --num_epochs 10  --batch_size 256  --mode train --dataset DBS --data_path /root/Anomaly_Explanation/dataset/dbsherlock/processed/tpcc_500w --input_c 200 --output_c 200 --win_size 25 --step_size 25 --cause "all"
python src/main.py --anormly_ratio 4 --num_epochs 10  --batch_size 256  --mode test  --dataset DBS --data_path /root/Anomaly_Explanation/dataset/dbsherlock/processed/tpcc_500w --input_c 200 --output_c 200 --win_size 25 --step_size 25 --pretrained_model 20 --cause "all"
