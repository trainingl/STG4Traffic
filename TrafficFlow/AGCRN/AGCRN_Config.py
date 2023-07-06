import argparse
import configparser

MODE = 'train'
DEBUG = False
DEVICE = 'cuda:4'
MODEL = 'AGCRN'
DATASET = 'PEMSD4'
GRAPH = "../data/PEMSD4/PEMSD4.csv"
FILENAME_ID = None

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
args.add_argument('--graph_path', default=GRAPH, type=str)
args.add_argument('--graph_type', default='BINARY', type=str)
args.add_argument('--normalized_k', default=0.1, type=float)
args.add_argument('--filename_id', default=FILENAME_ID, type=str)

# 3. conf params
args.add_argument('--val_ratio', default=config['data']['val_ratio'], type=float)
args.add_argument('--test_ratio', default=config['data']['test_ratio'], type=float)
args.add_argument('--window', default=config['data']['lag'], type=int)
args.add_argument('--horizon', default=config['data']['horizon'], type=int)
args.add_argument('--num_node', default=config['data']['num_nodes'], type=int)
args.add_argument('--tod', default=config['data']['tod'], type=eval)
args.add_argument('--normalizer', default=config['data']['normalizer'], type=str)
args.add_argument('--default_graph', default=config['data']['default_graph'], type=eval)

args.add_argument('--loss_func', default=config['train']['loss_func'], type=str)
args.add_argument('--seed', default=config['train']['seed'], type=int)
args.add_argument('--batch_size', default=config['train']['batch_size'], type=int)
args.add_argument('--epochs', default=config['train']['epochs'], type=int)
args.add_argument('--lr_init', default=config['train']['lr_init'], type=float)
args.add_argument('--weight_decay', default=config['train']['weight_decay'], type=float)
args.add_argument('--lr_decay', default=config['train']['lr_decay'], type=eval)
args.add_argument('--lr_decay_rate', default=config['train']['lr_decay_rate'], type=float)
args.add_argument('--lr_decay_step', default=config['train']['lr_decay_step'], type=str)
args.add_argument('--early_stop', default=config['train']['early_stop'], type=eval)
args.add_argument('--early_stop_patience', default=config['train']['early_stop_patience'], type=int)
args.add_argument('--grad_norm', default=config['train']['grad_norm'], type=eval)
args.add_argument('--max_grad_norm', default=config['train']['max_grad_norm'], type=int)
args.add_argument('--teacher_forcing', default=False, type=bool)
args.add_argument('--real_value', default=config['train']['real_value'], type=eval)
# 4. model params
args.add_argument('--input_dim', default=config['model']['input_dim'], type=eval)
args.add_argument('--output_dim', default=config['model']['output_dim'], type=eval)
args.add_argument('--embed_dim', default=config['model']['embed_dim'], type=eval)
args.add_argument('--hidden_dim', default=config['model']['rnn_units'], type=eval)
args.add_argument('--num_layers', default=config['model']['num_layers'], type=eval)
args.add_argument('--cheb_k', default=config['model']['cheb_order'], type=eval)

args.add_argument('--mae_thresh', default=config['test']['mae_thresh'], type=eval)
args.add_argument('--mape_thresh', default=config['test']['mape_thresh'], type=float)
args.add_argument('--log_dir', default='./', type=str)
args = args.parse_args()
