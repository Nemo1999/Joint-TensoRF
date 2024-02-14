from scripts.gpu_scheduler import ModelTypes, RunConfigsGPUScheduler

# ids for available gpus
#free_gpus = set([0,1,2,3]) # use this if you have 4 gpus on the device
free_gpus = set([0]) # use this if you have 1 gpu

# blender scenes to run
scenes = ["fern", "flower", "fortress", "horns", "leaves", "orchids", "room", "trex"]
scenes = scenes[:1] # run first scene only


configs = [
    {
        "name": f"barf_llff_{scene}",
        "tags": ["llff","BARF",scene],
        "data.dataset": "llff",
        "data.scene": scene,
    }
    for scene in scenes
]

scheduler = RunConfigsGPUScheduler(
    default_model=ModelTypes.BARF,
    available_gpu_ids=free_gpus,
    default_eval_only=False,
    default_train_only=False,
    default_use_wandb=True,
    default_use_pdb=False,
    default_wandb_group_name="Bundle_Adjusting_TensoRF",
    default_dataset_name="llff",
    default_yaml_config_file="barf_llff"
)
if __name__ == "__main__":
    scheduler.run_configs(exp_configs=configs)
