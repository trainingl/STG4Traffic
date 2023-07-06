import argparse
import configparser

MODE = 'train'
DEBUG = False
DEVICE = 'cuda:2'
MODEL = 'MTGNN'
DATASET = 'METRLA'  # PEMSBAY  or METRLA
dataset_dir = "../data/METR-LA/processed/"
adj_type = 'doubletransition'
graph_pkl = "../data/METR-LA/processed/adj_mx.pkl"

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
args.add_argument('--dataset_dir', default=dataset_dir, type=str)
args.add_argument('--graph_pkl', default=graph_pkl, type=str)
args.add_argument('--adjtype', default=adj_type, type=str)
args.add_argument('--num_nodes', default=config['data']['num_nodes'], type=int)
args.add_argument('--window', default=config['data']['window'], type=int)
args.add_argument('--horizon', default=config['data']['horizon'], type=int)

# 3. model params
args.add_argument('--gcn_true', default=config['model']['gcn_true'], type=eval)
args.add_argument('--buildA_true', default=config['model']['buildA_true'], type=eval)
args.add_argument('--gcn_depth', default=config['model']['gcn_depth'], type=eval)
args.add_argument('--subgraph_size', default=config['model']['subgraph_size'], type=eval)
args.add_argument('--dropout', default=config['model']['dropout'], type=eval)
args.add_argument('--node_dim', default=config['model']['node_dim'], type=eval)
args.add_argument('--dilation_exponential', default=config['model']['dilation_exponential'], type=eval)
args.add_argument('--conv_channels', default=config['model']['conv_channels'], type=eval)
args.add_argument('--residual_channels', default=config['model']['residual_channels'], type=eval)
args.add_argument('--skip_channels', default=config['model']['skip_channels'], type=eval)
args.add_argument('--end_channels', default=config['model']['end_channels'], type=eval)
args.add_argument('--input_dim', default=config['model']['input_dim'], type=eval)
args.add_argument('--output_dim', default=config['model']['output_dim'], type=eval)
args.add_argument('--layers', default=config['model']['layers'], type=eval)
args.add_argument('--propalpha', default=config['model']['propalpha'], type=eval)
args.add_argument('--tanhalpha', default=config['model']['tanhalpha'], type=eval)
args.add_argument('--layer_norm_affline', default=config['model']['layer_norm_affline'], type=eval)

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
args.add_argument('--step_size1', default=config['train']['step_size1'], type=int)
args.add_argument('--step_size2', default=config['train']['step_size2'], type=int)
args.add_argument('--num_split', default=config['train']['num_split'], type=int)
args.add_argument('--new_training_method', default=config['train']['new_training_method'], type=eval)
args = args.parse_args()