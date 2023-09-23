import argparse
import os.path as osp
from tkinter.tix import Tree
import yaml
from yaml import SafeLoader
from util.helper import seed_everything
from model.COSTA import Runner
support_models = ['SFA','COSTA']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='/data/yy/potgclv2/COSTA-v2')
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--config', type=str, default='default.yaml')
    parser.add_argument('--model', type=str)
    parser.add_argument('--use_pot', default=False, action="store_true") # whether to use pot in loss
    parser.add_argument('--kappa', type=float, default=0.5)
    parser.add_argument('--pot_batch', type=int, default=-1)
    parser.add_argument('--drop_1', type=float, default=-1)
    parser.add_argument('--drop_2', type=float, default=-1)
    parser.add_argument('--tau', type=float, default=0.4)
    parser.add_argument('--num_epochs', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--save_file', type=str, default=".")


    args = parser.parse_args()

    config = yaml.load(open(osp.join(
        osp.join(args.root, 'config'), args.config)), Loader=SafeLoader)[args.dataset]
    config['dataset'] = args.dataset
    config['data_dir'] = osp.join(args.root, 'data')
    if args.drop_1 != -1:
        config['drop_edge_rate_1'] = args.drop_1
    if args.drop_2 != -1:
        config['drop_edge_rate_2'] = args.drop_2
    if args.tau != -1:
        config['tau'] = args.tau
    if args.num_epochs != -1:
        config['num_epochs'] = args.n_epochs
    print(args)
    args_dict = vars(args)
    for key in args_dict.keys():
        config[key] = args_dict[key]
    print(config)
    seed_everything(config['seed'])
    use_pot = args.use_pot
    kappa = args.kappa
    pot_batch = args.pot_batch
    res = Runner(conf=config).execute()
    print(res)
    res_file = f"res/{args.dataset}_pot_temp.csv" if use_pot else f"res/{args.dataset}_base_temp.csv"
    if args.save_file == '.':
        f = open(res_file,"a+")
    else:
        f = open(args.save_file, "a+")
    if use_pot:
        f.write(f'{config["drop_edge_rate_1"]}, {config["drop_edge_rate_2"]}, {config["tau"]}, {kappa}, {pot_batch}, '
                f'{res["micro_f1"]:.4f}, {res["macro_f1"]:.4f}\n')
    else:
        f.write(f'{config["drop_edge_rate_1"]}, {config["drop_edge_rate_2"]}, {config["tau"]}, '
                f'{res["micro_f1"]:.4f}, {res["macro_f1"]:.4f}\n')
    f.close()

    
