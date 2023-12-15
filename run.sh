pip install -r requirements.txt

./preprocess.sh

python train_first.py --config_path ./Configs/config.yml

python train_second.py --config_path ./Configs/config.yml