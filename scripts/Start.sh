export CUDA_VISIBLE_DEVICES=0

python main.py --anormly_ratio 0.5 --num_epochs 10   --batch_size 256  --mode train --dataset SMD  --data_path dataset/SMD   --input_c 38
python main.py --anormly_ratio 0.5 --num_epochs 10   --batch_size 256     --mode test    --dataset SMD   --data_path dataset/SMD     --input_c 38     --pretrained_model 20

#python main.py --anormly_ratio 1 --num_epochs 3   --batch_size 256  --mode train --dataset MSL  --data_path dataset/MSL --input_c 55    --output_c 55
#python main.py --anormly_ratio 1  --num_epochs 10      --batch_size 256     --mode test    --dataset MSL   --data_path dataset/MSL  --input_c 55    --output_c 55  --pretrained_model 20

#python main.py --anormly_ratio 1 --num_epochs 3   --batch_size 256  --mode train --dataset SMAP  --data_path dataset/SMAP --input_c 25    --output_c 25
#python main.py --anormly_ratio 1  --num_epochs 10        --batch_size 256     --mode test    --dataset SMAP   --data_path dataset/SMAP  --input_c 25    --output_c 25  --pretrained_model 20

#python main.py --anormly_ratio 0.5 --num_epochs 3    --batch_size 256  --mode train --dataset SWaT  --data_path dataset/SWaT --input_c 51    --output_c 51
#python main.py --anormly_ratio 0.1  --num_epochs 10        --batch_size 256     --mode test    --dataset SWaT   --data_path dataset/SWaT  --input_c 51    --output_c 51  --pretrained_model 10

#python main.py --anormly_ratio 1 --num_epochs 3    --batch_size 256  --mode train --dataset PSM  --data_path dataset/PSM --input_c 25    --output_c 25
#python main.py --anormly_ratio 1  --num_epochs 10       --batch_size 256     --mode test    --dataset PSM   --data_path dataset/PSM  --input_c 25    --output_c 25  --pretrained_model 20


