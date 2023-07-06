import argparse
import configparser

MODE = 'train'
DEBUG = False
DEVICE = 'cuda:4'
MODEL = 'GMAN'
DATASET = 'METRLA'  # PEMSBAY
dataset_dir = "../data/METR-LA/metr-la.h5"
SE_file = '../data/METR-LA/SE(METR).txt'

# 1. get configuration
config_file = './{}_{}.conf'.format(DATASET, MODEL)
config = configparser.ConfigParser()
config.read(config_file)

# 2. arguments parser
args = argparse.ArgumentParser(description='Arguments')
args.add_argument('--mode', default=MODE, type=str)
args.add_argument('--debug', default=DEBUG, type=eval)
args.add_argument('--device', default=DEVICE, type=str)
args.add_argument('--model', default=MODEL, type=str)
args.add_argument('--dataset', default=DATASET, type=str)
args.add_argument("--train_ratio", default=0.7, type=float)
args.add_argument("--test_ratio", default=0.2, type=float)
args.add_argument('--traffic_file', default=dataset_dir, type=str)
args.add_argument('--SE_file', default=SE_file, type=str)
args.add_argument('--num_nodes', default=config['data']['num_nodes'], type=int)
args.add_argument('--window', default=config['data']['window'], type=int)
args.add_argument('--horizon', default=config['data']['horizon'], type=int)
args.add_argument('--steps_per_day', default=config['data']['steps_per_day'], type=int)

# 3. model params
args.add_argument('--L', default=config['model']['L'], type=int)
args.add_argument('--K', default=config['model']['K'], type=int)
args.add_argument('--d', default=config['model']['d'], type=int)
args.add_argument('--bn_decay', default=config['model']['bn_decay'], type=float)
args.add_argument('--use_bias', default=config['model']['use_bias'], type=eval)
args.add_argument('--mask', default=config['model']['mask'], type=eval)

# 4. train params
args.add_argument('--cl', default=config['train']['cl'], type=eval)
args.add_argument('--seed', default=config['train']['seed'], type=int)
args.add_argument('--loss_func', default=config['train']['loss_func'], type=str)
args.add_argument('--batch_size', default=config['train']['batch_size'], type=int)
args.add_argument('--epochs', default=config['train']['epochs'], type=int)
args.add_argument('--weight_decay', default=config['train']['weight_decay'], type=float)
args.add_argument('--lr_init', default=config['train']['lr_init'], type=float)
args.add_argument('--lr_decay', default=config['train']['lr_decay'], type=eval)
args.add_argument('--lr_decay_rate', default=config['train']['lr_decay_rate'], type=float)
args.add_argument('--lr_decay_step', default=config['train']['lr_decay_step'], type=str)
args.add_argument('--early_stop', default=config['train']['early_stop'], type=eval)
args.add_argument('--early_stop_patience', default=config['train']['early_stop_patience'], type=int)
args.add_argument('--grad_norm', default=config['train']['grad_norm'], type=eval)
args.add_argument('--max_grad_norm', default=config['train']['max_grad_norm'], type=int)
args.add_argument('--step_size', default=config['train']['step_size'], type=int)
args = args.parse_args()