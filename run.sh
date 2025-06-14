  
# conda activate Bundle_Adjusting_TensoRF

NAME=planar_bat
python train_2d.py --group=planar_JFT --model=planar_svd --yaml=planar_bat  --seed=0  --name=${NAME}_h2t2 --wandb  --wandb_name=${NAME}_h2t2 --freq.vis=50  \
    --warp.noise_h=0.2 --warp.noise_t=0.2


python train_2d.py --group=planar_JFT --model=planar_svd --yaml=planar_bat  --seed=0  --name=${NAME}_h1t2 --wandb  --wandb_name=${NAME}_h1t2 --freq.vis=50  --warp.noise_h=0.1 --warp.noise_t=0.2



python train_2d.py --group=planar_JFT --model=planar_svd --yaml=planar_bat  --seed=0  --name=${NAME}_3h1t3 --wandb  --wandb_name=${NAME}_h1t3 --freq.vis=50  \
    --warp.noise_h=0.1 --warp.noise_t=0.3

python train_2d.py --group=planar_JFT --model=planar_svd --yaml=planar_bat  --seed=0  --name=${NAME}_h2t3 --wandb  --wandb_name=${NAME}_h2t3 --freq.vis=50   \
    --warp.noise_h=0.2 --warp.noise_t=0.3

# noise_h_list=(0.1 0.2 0.3 0.4 0.5)
# noise_t_list=(0.2 0.4 0.6 0.8 1.0)
# python train_2d.py --group=planar_GNN --model=planar --yaml=planar_barf  --seed=0 --gnn=true --warp.noise_h=$noise_h --warp.noise_t=$noise_t --name=gcn_h_${noise_h}_t_${noise_t}  --wandb=false

# for noise_h in ${noise_h_list[@]};
# do
#     for noise_t in ${noise_t_list[@]};
#     do
#         echo "noise_h: $noise_h, noise_t: $noise_t"
        
#         python train_2d.py --group=planar_GNN --model=planar --yaml=planar_barf  --seed=0 --gnn=true --warp.noise_h=$noise_h --warp.noise_t=$noise_t --name=gcn_h_${noise_h}_t_${noise_t}  --wandb=false
#         python train_2d.py --group=planar_GNN --model=planar --yaml=planar_barf  --seed=0 --gnn=false --warp.noise_h=$noise_h --warp.noise_t=$noise_t --name=base_h_${noise_h}_t_${noise_t}  --wandb=false
#     done
# done


# SCENE_LIST=( "orchids" "fern" "horns" "trex" "leaves" "flower"  "room" "fortress" )  

# for SCENE in "${SCENE_LIST[@]}"
# do
# python train_3d.py --group=LLFF_JTF --model=bat --yaml=bat_llff_VM_MLP  --seed=0  --name=${SCENE}  --wandb=true --visdom! --data.scene=${SCENE}
# done