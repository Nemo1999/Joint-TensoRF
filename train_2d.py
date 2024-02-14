import numpy as np
import os,sys,time
import torch
import importlib
import wandb
import options
from util import log


def main():

    log.process(os.getpid())
    log.title("[{}] (PyTorch code for training NeRF/BARF)".format(sys.argv[0]))

    opt_cmd = options.parse_arguments(sys.argv[1:])
    opt = options.set(opt_cmd=opt_cmd)
    options.save_options_file(opt)
    if opt.wandb:
        wandb.init(
            project=f"{opt.group}",
            notes=f"planar run: model={opt.model}, name={opt.name}, yaml={opt.yaml}",
            tags=["planar", opt.name] + opt.tags,
            config=opt,
        )
    else:
        wandb.init(mode="disabled")

    wandb.run.name = f"{opt.wandb_name}"

    # train model
    with torch.cuda.device(opt.device):

        model = importlib.import_module("model.{}".format(opt.model))
        m = model.Model(opt)

        m.load_dataset(opt)
        m.build_networks(opt)
        m.setup_optimizer(opt)
        m.restore_checkpoint(opt)
        m.setup_visualizer(opt)

        m.train(opt)

if __name__ == "__main__":
    main()
