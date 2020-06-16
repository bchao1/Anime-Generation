import os
import yaml
import shutil
from argparse import ArgumentParser
from trainer import ACGANTrainer

config_file = 'config.yaml'

def make_dirs(run_dir):
    if os.path.exists(run_dir):
        shutil.rmtree(run_dir)
    os.makedirs(run_dir, exist_ok = True)
    os.makedirs(os.path.join(run_dir, 'images', 'fix'))
    os.makedirs(os.path.join(run_dir, 'images', 'class'))
    os.makedirs(os.path.join(run_dir, 'ckpt'))
    
def main(override):
    config = yaml.load(open(config_file, 'r'), Loader = yaml.FullLoader)
    run_dir = os.path.join('runs', str(config['run']))
    
    if os.path.exists(run_dir) and not override:
        print("Run name exists! Please choose another name.")
        return

    make_dirs(run_dir)
    shutil.copyfile(config_file, os.path.join(run_dir, config_file))
    
    trainer = ACGANTrainer(config)
    trainer.start()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-r', action = 'store_true', help = "Override run name.")
    args = parser.parse_args()
    main(args.r)