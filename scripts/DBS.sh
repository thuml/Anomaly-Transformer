export CUDA_VISIBLE_DEVICES=0

python main.py --anormly_ratio 1 --num_epochs 10   --batch_size 256  --mode train --dataset DBS  --data_path dataset/DBS_C_new   --input_c 200 --output_c 200 --win_size 25 --step_size 25
python main.py --anormly_ratio 1 --num_epochs 10   --batch_size 256  --mode test  --dataset DBS  --data_path dataset/DBS_C_new   --input_c 200 --output_c 200 --win_size 25 --step_size 25 --pretrained_model 20