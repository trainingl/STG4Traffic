import sys
sys.path.append('../')

import os
import torch
import pandas as pd
from datetime import datetime
from lib.utils import *
from lib.data_loader import *
from lib.generate_adj_mx import *
from GTS_Config import args
from GTS_Trainer import Trainer
from model.GTS.gts import GTSModel as Network


def load_data(args):
    data_loader = load_dataset(args.dataset_dir, args.batch_size, args.batch_size, args.batch_size)
    scaler = data_loader['scaler']
    # Node Features
    df = pd.read_hdf(args.origin_data)
    num_samples = df.shape[0]
    num_train = round(num_samples * 0.7)
    df = df[:num_train].values
    scaler = StandardScaler(mean=df.mean(), std=df.std())
    traffic_feas = scaler.transform(df)
    node_feas = torch.Tensor(traffic_feas).to(args.device)

    # Init Graph Structure
    k = args.knn_k
    knn_metric = 'cosine'
    from sklearn.neighbors import kneighbors_graph
    g = kneighbors_graph(traffic_feas.T, k, metric=knn_metric)
    g = np.array(g.todense(), dtype=np.float32)
    adj_mx = torch.Tensor(g).to(args.device)
    return adj_mx, node_feas, data_loader, scaler

def generate_model_components(args):
    # 1. model
    model = Network(
        temperature = args.temperature, 
        input_dim = args.input_dim,
        output_dim = args.output_dim,
        rnn_units = args.rnn_units,
        seq_len = args.window,
        horizon = args.horizon,
        num_rnn_layers = args.num_rnn_layers,
        num_nodes = args.num_nodes,
        filter_type = args.filter_type,
        dim_fc = args.dim_fc,
        max_diffusion_step = args.max_diffusion_step,
        cl_decay_steps = args.cl_decay_steps,
        use_curriculum_learning = args.use_curriculum_learning
    )
    model = model.to(args.device)
    # print the number of model parameters
    print_model_parameters(model, only_num=True)
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
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr_init,eps=args.epsilon)
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
    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.device[5]))
    else:
        args.device = 'cpu'
    adj_mx, node_feas, data_loader, scaler = load_data(args)
    args.log_dir = get_log_dir(args.model, args.dataset)
    model, loss, optimizer, lr_scheduler = generate_model_components(args)
    trainer = Trainer(
        args = args, 
        data_loader = data_loader, 
        scaler = scaler, 
        model = model, 
        loss = loss, 
        optimizer = optimizer, 
        lr_scheduler = lr_scheduler,
        node_feas = node_feas,
        adj_mx = adj_mx,
        cl=args.cl
    )
    if args.mode == "train":
        trainer.train()
    elif args.mode == 'test':
        checkpoint = "../log/GTS/PEMSBAY/20230204200025/PEMSBAY_GTS_best_model.pth"
        trainer.test(args, model, data_loader, scaler, trainer.logger, save_path=checkpoint)
    else:
        raise ValueError