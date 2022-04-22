export CUDA_VISIBLE_DEVICES=7

python main.py --anormly_ratio 1 --num_epochs 3   --batch_size 256  --mode train --dataset MSL  --data_path dataset/MSL --input_c 55    --output_c 55
python main.py --anormly_ratio 1  --num_epochs 10      --batch_size 256     --mode test    --dataset MSL   --data_path dataset/MSL  --input_c 55    --output_c 55  --pretrained_model 20




