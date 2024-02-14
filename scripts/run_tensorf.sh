# conda initialize
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"

# Activate the environment
conda activate BARF

cd bundle-adjusting-NeRF && python3 train_3d.py --group=Bundle_Adjusting_TensoRF\
                                    --model=tensorf\
                                    --yaml=tensorf_blender\
                                    --name=lego_nerf\
                                    --data.scene=lego\
                                    --wandb=true\
                                    --visdom=false

#--barf_c2f=[0.1,0.5]
