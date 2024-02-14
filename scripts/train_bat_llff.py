from scripts.train_and_evaluate_bat_llff import scheduler, configs

if __name__ == "__main__":
    scheduler.default_eval_only=False
    scheduler.default_train_only=True
    scheduler.run_configs(exp_configs=configs)
