from scripts.gpu_scheduler import ModelTypes, RunConfigsGPUScheduler

# ids for available gpus
#free_gpus = set([0,1,2,3]) # use this if you have 4 gpus on the device
free_gpus = set([0]) # use this if you have 1 gpu

# blender scenes to run
scenes = ["lego", "chair", "ficus", "hotdog", "drums", "materials", "mic", "ship"]
scenes = scenes[:1] # run first scene only


configs = [
    {
        "name": f"barf_blender_{scene}",
        "tags": ["blender","BARF",scene],
        "data.dataset": "blender",
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
    default_dataset_name="blender",
    default_yaml_config_file="barf_blender"
)
if __name__ == "__main__":
    scheduler.run_configs(exp_configs=configs)
