# This script shows how to install dependency for GNeRF

# conda initialize
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"

# Activate the environment
conda activate Bundle_Adjusting_TensoRF

python train_2d.py --group=Planar_Experiments --model=planar --yaml=planar_barf  --seed=0
