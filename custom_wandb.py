import wandb

def wandb_init(args, name):
    configs = {}
    if args is not None:
        for key, value in vars(args).items():
            configs[key] = value
        wandb.init(project='thr2rgb_benchmark',
                name=name,
                config=configs)
    
def wandb_log(category, name, value):
    wandb.log({f"{category}/{name}": value})
