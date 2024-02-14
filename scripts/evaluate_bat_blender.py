from scripts.train_and_evaluate_bat_blender import scheduler, configs

if __name__ == "__main__":
    scheduler.default_eval_only=True
    scheduler.default_train_only=False
    scheduler.run_configs(exp_configs=configs)
