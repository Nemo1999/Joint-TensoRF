import numpy as np
import os,sys,time
import torch
import importlib
import wandb
import options
from util import log
from icecream import ic
from torch.profiler import profile, record_function, ProfilerActivity
import pdb, traceback, sys
import shutil


def main(opt):

    log.process(os.getpid())
    log.title("[{}] (PyTorch code for training NeRF/BARF)".format(sys.argv[0]))

    options.save_options_file(opt)

    # setting env variables
    if hasattr(opt, "env_variables"):
        for var in opt.env_variables.keys():
            os.environ[var] = (opt.env_variables)[var]
            # printing environment variables
        print(os.environ)

    if opt.wandb:
        wandb.init(
            project=f"{opt.group}",
            notes=f"planar run: model={opt.model}, name={opt.name}, yaml={opt.yaml}",
            tags=["planar", opt.model]+opt.tags,
            config=opt,
            resume=opt.resume
        )
    else:
        wandb.init(mode="disabled")

    wandb.run.name = f"{opt.wandb_name}"

    if opt.ic:
        ic.enable()
    else:
        ic.disable()

    # load GT scene content
    if  opt.train_pose_with_GT_scene or opt.register_new_poses:
        #opt.load = "lego_gt.ckpt"
        opt.load =  opt.GT_scene_ckpt


    if hasattr(opt,"profiling") and opt.profiling == True:
        opt["prof"] = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=5, warmup=5, active=1, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(opt.prof_path),
            record_shapes=True,
            with_stack=True
        )
        opt.prof.start()

    if not opt.evaluation_only:
        # remove existing output result
        if os.path.exists(opt.output_path):
            shutil.rmtree(opt.output_path)

        # train model
        with torch.cuda.device(opt.device):
             model = importlib.import_module("model.{}".format(opt.model))

             m = model.Model(opt)
             m.load_dataset(opt, train_split=opt.train_split)
             m.build_networks(opt)
             m.setup_optimizer(opt)
             m.restore_checkpoint(opt)
             if opt.train_pose_with_GT_scene or opt.register_new_poses:
                 assert opt.model in ["bat", "barf"]
                 m.interrupt_pose(opt)
                 m.freeze_scene(opt)
             m.setup_visualizer(opt)
             m.train(opt)

    if hasattr(opt,"profiling") and opt.profiling == True:
        opt.prof.stop()
        #print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=100))
        #prof.export_chrome_trace("./profiling_trace.json")
        raise Exception("profiling Done!!")

    if not opt.train_only:
      # evaluate model
      with torch.cuda.device(opt.device):
        opt.load = "{0}/model.ckpt".format(opt.output_path)
        model = importlib.import_module("model.{}".format(opt.model))
        m = model.Model(opt)

        m.load_dataset(opt,eval_split="test", train_split=opt.train_split) # load train_split for loading correct  pose embbeding se3 parameter shape
        m.build_networks(opt)

        if opt.model in ["barf", "bat"]:
            #m.generate_videos_pose(opt)
            pass # we have better video pose

        m.restore_checkpoint(opt)
        m.freeze_scene(opt)
        m.freeze_poses(opt)
        if opt.data.dataset in ["blender","llff"]:
            m.evaluate_full(opt)
            m.generate_videos_synthesis(opt)


if __name__ == '__main__':
    opt_cmd = options.parse_arguments(sys.argv[1:])
    opt = options.set(opt_cmd=opt_cmd)

    if "pdb" not in opt  or opt.pdb == False :
        main(opt)
    else:
        try:
            main(opt)
        except:
            extype, value, tb = sys.exc_info()
            traceback.print_exc()
            pdb.post_mortem(tb)
