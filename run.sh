# First stage training
CUDA_VISIBLE_DEVICES=1 python train_first.py --config_path ./Configs/config.yml > train_first_output.log 2>&1
# Second stage training
CUDA_VISIBLE_DEVICES=1 python train_second.py --config_path ./Configs/config.yml > train_second_output.log 2>&1