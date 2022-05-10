import os
import yaml
import shutil
from argparse import ArgumentParser
from trainer import ACGANTrainer
from trainer_h5 import ACGANTrainer_h5

config_file = 'config.yaml'

def make_dirs(run_dir):
    shutil.rmtree(run_dir, ignore_errors=True)
    os.makedirs(run_dir, exist_ok = True)
    os.makedirs(os.path.join(run_dir, 'images', 'class'))
    os.makedirs(os.path.join(run_dir, 'ckpt'))
    
def main(args):
    config = yaml.load(open(config_file, 'r'), Loader = yaml.FullLoader)
    run_dir = os.path.join(config["runs_root"], str(config['run']))
    save_dir = os.path.join(config["save_root"], str(config['run']))

    config["run_dir"] = run_dir
    config["save_dir"] = save_dir
    
    if os.path.exists(run_dir) and not args.override:
        print("Run name exists! Please choose another name.")
        return

    make_dirs(run_dir)
    make_dirs(save_dir)

    shutil.copyfile(config_file, os.path.join(run_dir, config_file))
    shutil.copyfile(config_file, os.path.join(save_dir, config_file))
    
    if args.mode == "img":
        trainer = ACGANTrainer(config)
    elif args.mode == "h5":
        trainer = ACGANTrainer_h5(config)

    trainer.start()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--override', action = 'store_true', help = "Override run name.")
    parser.add_argument("--mode", type=str, choices=["img", "h5"], required=True)
    args = parser.parse_args()
    main(args)