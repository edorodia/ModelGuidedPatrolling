
# Script to train DRL

# Usage: ./train_shekel.sh
python Training/Train.py --benchmark shekel --w_reward_weight 10.0 --i_reward_weight 1.0 --model deepUnet --device 0
python Training/Train.py --benchmark shekel --w_reward_weight 1.0 --i_reward_weight 10.0 --model deepUnet --device 0
python Training/Train.py --benchmark shekel --w_reward_weight 10.0 --i_reward_weight 10.0 --model deepUnet --device 0
