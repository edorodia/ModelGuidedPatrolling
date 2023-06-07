
# Script to train DRL

# Usage: ./train_shekel.sh
python Training/Train.py --benchmark shekel --reward_weights 1.0 --reward_weights 1.0 --model miopic --decive "cuda:0"
python Training/Train.py --benchmark shekel --reward_weights 0.1 --reward_weights 1.0 --model miopic --decive "cuda:0" 
python Training/Train.py --benchmark shekel --reward_weights 1.0 --reward_weights 0.1 --model miopic --decive "cuda:0" 