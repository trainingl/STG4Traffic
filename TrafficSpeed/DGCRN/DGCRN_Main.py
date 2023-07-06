import sys
sys.path.append('../')

import os
import torch
from datetime import datetime
from lib.utils import *
from lib.data_loader import *
from lib.generate_adj_mx import *
from DGCRN_Config import args
from DGCRN_Trainer import Trainer
from model.DGCRN.net import DGCRN as Network


def load_data(args):
    data_loader = load_dataset(args.dataset_dir, args.batch_size, args.batch_size, args.batch_size)
    scaler = data_loader['scaler']
    # 加载拓扑图的邻接矩阵
    _, _, adj_mx = load_pickle(args.graph_pkl)
    adj_mx = [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]
    return adj_mx, data_loader, scaler   # adj_mx type: list


def generate_model_components(args, predefined_A):
    predefined_A = [torch.tensor(adj).to(args.device) for adj in predefined_A]
    # 1. model
    model = Network(
        args.gcn_depth,
        args.num_nodes,
        args.device,
        predefined_A=predefined_A,
        dropout=args.dropout,
        subgraph_size=args.subgraph_size,
        node_dim=args.node_dim,
        middle_dim=2,
        seq_length=args.window,
        in_dim=args.input_dim,
        out_dim=args.horizon,
        layers=args.num_layers,
        list_weight=[0.05, 0.95, 0.95],
        tanhalpha=args.tanhalpha,
        cl_decay_steps=args.cl_decay_steps,
        rnn_size=args.rnn_units,
        hyperGNN_dim=args.hyperGNN_dim
    )
    model = model.to(args.device)
    # print the number of model parameters
    print_model_parameters(model, only_num=False)
    # 2. loss
    if args.loss_func == 'mask_mae':
        loss = masked_mae
    elif args.loss_func == 'mae':
        loss = torch.nn.L1Loss().to(args.device)
    elif args.loss_func == 'mse':
        loss = torch.nn.MSELoss().to(args.device)
    elif args.loss_func == 'smoothloss':
        loss = torch.nn.SmoothL1Loss().to(args.device)
    else:
        raise ValueError
    # 3. optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr_init, weight_decay=args.weight_decay)
    # 4. learning rate decay
    lr_scheduler = None
    if args.lr_decay:
        print('Applying learning rate decay.')
        lr_decay_steps = [int(i) for i in list(args.lr_decay_step.split(','))]
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                            milestones=lr_decay_steps,
                                                            gamma=args.lr_decay_rate)
    return model, loss, optimizer, lr_scheduler


def get_log_dir(model, dataset):
    current_time = datetime.now().strftime('%Y%m%d%H%M%S')
    current_dir = os.path.abspath(os.path.join(os.getcwd(), "../"))  # /GNN4Traffic/TrafficSpeed
    log_dir = os.path.join(current_dir,'log', model, dataset, current_time) 
    return log_dir

def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    setup_seed(args.seed)
    torch.set_num_threads(3)
    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.device[5]))
    else:
        args.device = 'cpu'
    supports, data_loader, scaler = load_data(args)
    args.log_dir = get_log_dir(args.model, args.dataset)
    model, loss, optimizer, lr_scheduler = generate_model_components(args, supports)
    trainer = Trainer(
        args=args, 
        data_loader=data_loader, 
        scaler=scaler, 
        model=model, 
        loss=loss, 
        optimizer=optimizer, 
        lr_scheduler=lr_scheduler,
        cl=args.cl, 
        new_training_method=args.new_training_method
    )
    
    if args.mode == "train":
        trainer.train()
    elif args.mode == 'test':
        checkpoint = "../log/DGCRN/PEMSBAY/20230205121734/PEMSBAY_DGCRN_best_model.pth"  # 20230125134933/METRLA_DGCRN_best_model
        trainer.test(args, model, data_loader, scaler, trainer.logger, save_path=checkpoint)
    else:
        raise ValueError