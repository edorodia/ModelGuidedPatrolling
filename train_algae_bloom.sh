
# Script to train DRL

# Usage: ./train_shekel.sh
python Training/Train.py --benchmark algae_bloom --w_reward_weight 10.0 --i_reward_weight 1.0 --model miopic --device 0
python Training/Train.py --benchmark algae_bloom --w_reward_weight 1.0 --i_reward_weight 10.0 --model miopic --device 0
python Training/Train.py --benchmark algae_bloom --w_reward_weight 10.0 --i_reward_weight 10.0 --model miopic --device 0
