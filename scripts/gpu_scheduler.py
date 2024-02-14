import time
import subprocess
import atexit
from icecream import ic

class ModelTypes:
    NERF="nerf" # MLP architecture
    BARF="barf" # MLP architecture with pose learning + coarse to fine Positional Encoding
    TENSORF="tensorf" # Decomposed Low-Rank Tensor Architecture
    OURS="bat" # Decomposed Low-Rank Tensor Architecture (randomized 2D & 3D decomposable filtering + edge guided loss mask)

class RunConfigsGPUScheduler:
    def __init__(
            self,
            default_model=ModelTypes.OURS,
            available_gpu_ids : set[int] =set([0]),
            default_eval_only=False, # don't train but load checkpoint with current name and run evaluation only, currently the evaluation result will be uploaded to wandb as a separate run
            default_train_only=False, # don't evaluate after training (used to calculate training time), evaluation can takes longer time than training because we have to calibrate the testing pose with back_propagation for each testing image
            default_use_wandb=True,
            default_wandb_group_name=None,
            default_random_seed=0,
            default_use_visdom=False,
            default_early_stop_iters=None, # use for quickly checking experiment convergence trend
            default_yaml_config_file=None,
            default_dataset_name="blender", # valid options are blender, llff
            default_use_pdb=False # jump into pdb on error
    ):
        # gpu and process
        self.running_processes = []
        self.available_gpus = available_gpu_ids

        # default config options
        self.default_model=default_model
        self.default_eval_only=False
        self.default_use_wandb = default_use_wandb
        self.default_use_visdom = default_use_visdom
        self.default_use_pdb = default_use_pdb
        self.default_wandb_group_name = default_wandb_group_name
        self.default_random_seed = default_random_seed
        self.default_early_stop_iters=None
        self.default_yaml_config_file=default_yaml_config_file
        self.default_dataset_name=default_dataset_name
        self.default_config = {
            "model": self.default_model,
            "yaml": self.default_yaml_config_file,
            "seed": self.default_random_seed,
            "visdom": self.default_use_visdom,
            "wandb": self.default_use_wandb,
            "data.dataset": self.default_dataset_name,
            "pdb": self.default_use_pdb,
            "group": self.default_wandb_group_name,
            "evaluation_only": self.default_eval_only,
        }
        if self.default_early_stop_iters:
            self.default_config["early_stop_iter"]=self.default_early_stop_iters

    def get_cmd(
            self,
            config: dict,
            gpu_id: int,
    ):
        config["wandb_name"] = config["name"]
        config["gpu"] = gpu_id
        cmd_parts = {
            "conda_init": 'eval "$(command conda \'shell.bash\' \'hook\' 2> /dev/null)"',
            "conda_activate": 'conda activate Bundle_Adjusting_TensoRF',
            "train_cmd": ("yes |" if not config["pdb"] else "") + "python train_3d.py",
            "option_str": " ".join(f"--{key}={str(val).replace(' ','')}\\\n" for key, val in config.items())
        }
        cmd = ";".join([cmd_parts["conda_init"], cmd_parts["conda_activate"], cmd_parts["train_cmd"]+" "+cmd_parts["option_str"]])
        return cmd

    def run_configs(
            self,
            exp_configs: list[dict]
    ):
        for config in exp_configs:
            # overrite default_config with current config
            complete_config = dict(self.default_config, **config)
            # get current avalable gpus
            current_gpu = self.available_gpus.pop()

            # get command
            cmd = self.get_cmd(complete_config, current_gpu)
            # execute command
            print("Execting the following command:")
            print(cmd)
            proc = subprocess.Popen(cmd, shell=True)
            self.running_processes.append( (proc,current_gpu) )

            # wait for next available gpu
            terminated_proc = None
            while len(self.available_gpus) == 0:
                for proc, gpu in self.running_processes:
                    if proc.poll() is not None:
                        terminated_proc = proc
                        self.available_gpus.add(gpu)
                        break
                time.sleep(3)
            self.running_processes = list(filter(lambda proc_gpu: proc_gpu[0]!=terminated_proc, self.running_processes))
            time.sleep(2)
        # wait for all processes to finish
        for proc, _ in self.running_processes:
            try:
                proc.wait()
            except KeyboardInterrupt:
                for proc, _ in proc:
                    proc.kill()
