import sys
sys.path.append('../')

import os
import copy
import torch
import time
from tqdm import tqdm
from lib.utils import get_logger
from lib.evaluate import All_Metrics

class Trainer(object):
    def __init__(self, args, train_loader, val_loader, test_loader, scaler, model, loss, optimizer, lr_scheduler=None):
        super(Trainer, self).__init__()
        self.args = args
        self.train_loader = train_loader
        self.train_per_epoch = len(train_loader)
        self.val_loader = val_loader
        if self.val_loader != None:
            self.val_per_epoch = len(self.val_loader)
        else:
            self.val_loader = test_loader
            self.val_per_epoch = len(self.val_loader)
        self.test_loader = test_loader
        self.scaler = scaler
        # 模型、损失函数、优化器
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.batches_seen = 0
        # 日志与模型的保存路径
        self.best_path = os.path.join(args.log_dir, '{}_{}_best_model.pth'.format(args.dataset, args.model))
        if os.path.isdir(args.log_dir) == False and not args.debug:
            os.makedirs(args.log_dir, exist_ok=True)  # run.log
        self.logger = get_logger(args.log_dir, name=args.model, debug=args.debug)
        self.logger.info("Experiment log path in: {}".format(args.log_dir))


    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for _, (data, target) in enumerate(self.train_loader):
            data = data[..., :self.args.input_dim]
            label = target[..., :self.args.output_dim]
            # data and target shape: B, T, N, D; output shape: B, T, N, D
            trainx = data.transpose(0, 1).reshape(self.args.horizon, self.args.batch_size, -1)     # (T, B, 1 * N)
            trainy = label.transpose(0, 1).reshape(self.args.horizon, self.args.batch_size, -1)     # (T, B, 1 * N)
            self.optimizer.zero_grad()
            trainy = self.scaler.inverse_transform(trainy)
            output = self.model(trainx, trainy, batches_seen=self.batches_seen)  # directly predict the true value
            if self.batches_seen == 0:
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr_init, eps=1.0e-3)
            loss = self.loss(output.cuda(), trainy)
            loss.backward()
            self.batches_seen += 1

            # add max grad clipping
            if self.args.grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()
            total_loss += loss.item()
        
        train_epoch_loss = total_loss / self.train_per_epoch
        # learning rate decay
        if self.args.lr_decay:
            self.lr_scheduler.step()
        return train_epoch_loss


    def val_epoch(self):
        self.model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for _, (data, target) in enumerate(self.val_loader):
                data = data[..., :self.args.input_dim]
                label = target[..., :self.args.output_dim]
                validx = data.transpose(0, 1).reshape(self.args.horizon, self.args.batch_size, -1)     # (T, B, 1 * N)
                validy = label.transpose(0, 1).reshape(self.args.horizon, self.args.batch_size, -1)     # (T, B, 1 * N)
                output = self.model(validx)
                validy = self.scaler.inverse_transform(validy)
                loss = self.loss(output.cuda(), validy)
                if not torch.isnan(loss):
                    total_val_loss += loss.item()
        val_loss = total_val_loss / self.val_per_epoch
        return val_loss


    def train(self):
        best_model = None
        best_loss = float('inf')
        not_improved_count = 0
        train_loss_list = []
        val_loss_list = []
        start_time = time.time()
        for epoch in tqdm(range(1, self.args.epochs + 1)):
            t1 = time.time()
            train_epoch_loss = self.train_epoch()
            t2 = time.time()
            # 验证, 如果是Encoder-Decoder结构，则需要将epoch作为参数传入
            val_epoch_loss = self.val_epoch()
            t3 = time.time()
            self.logger.info('Epoch {:03d}, Train Loss: {:.4f}, Valid Loss: {:.4f}, Training Time: {:.4f} secs, Inference Time: {:.4f} secs.'.format(epoch, train_epoch_loss, val_epoch_loss, (t2 - t1), (t3 - t2)))
            train_loss_list.append(train_epoch_loss)
            val_loss_list.append(val_epoch_loss)
            if train_epoch_loss > 1e6:
                self.logger.warning("Gradient explosion detected. Ending...")
                break
            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                not_improved_count = 0
                best_state = True
            else:
                not_improved_count += 1
                best_state = False
            # is or not early stop
            if self.args.early_stop:
                if not_improved_count == self.args.early_stop_patience:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                    "Training stops.".format(self.args.early_stop_patience))
                    break
            # save the best state
            if best_state == True:
                self.logger.info("Current best model saved!")
                best_model = copy.deepcopy(self.model.state_dict())
                torch.save(best_model, self.best_path)
            
        training_time = time.time() - start_time
        self.logger.info("Total training time: {:.4f} min, best loss: {:.6f}".format((training_time / 60), best_loss))
        # save the best model to file
        self.logger.info("Saving current best model to " + self.best_path)
        # load model and test
        self.model.load_state_dict(best_model)
        self.test(self.model, self.args, self.test_loader, self.scaler, self.logger)

    def setup_graph(self):
        self.model.eval()
        with torch.no_grad():
            for _, (data, target) in enumerate(self.val_loader):
                data = data[..., :self.args.input_dim]
                label = target[..., :self.args.output_dim]
                validx = data.transpose(0, 1).reshape(self.args.horizon, self.args.batch_size, -1)     # (T, B, 1 * N)
                validy = label.transpose(0, 1).reshape(self.args.horizon, self.args.batch_size, -1)     # (T, B, 1 * N)
                self.model(validx)
                break

    def test(self, model, args, data_loader, scaler, logger, save_path=None):
        if save_path != None:
            self.setup_graph()
            model.load_state_dict(torch.load(save_path))
            model.to(args.device)
            print("load saved model...")
        model.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for _, (data, target) in enumerate(data_loader):
                data = data[..., :args.input_dim].squeeze()
                label = target[..., :args.output_dim].squeeze()
                testx = data.transpose(0, 1).reshape(args.horizon, -1, args.num_node)
                testy = label.transpose(0, 1).reshape(args.horizon, -1, args.num_node) 
                output = model(testx)
                y_true.append(testy)
                y_pred.append(output)
        y_pred = torch.cat(y_pred, dim=1)
        y_true = scaler.inverse_transform(torch.cat(y_true, dim=1))
        for t in range(y_true.shape[0]):
            mae, rmse, mape = All_Metrics(y_pred[t, ...], y_true[t, ...], args.mae_thresh, args.mape_thresh)
            logger.info("Horizon {:02d}, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%".format(t + 1, mae, rmse, mape * 100))
        mae, rmse, mape = All_Metrics(y_pred, y_true, args.mae_thresh, args.mape_thresh)
        logger.info("Average Horizon, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%".format(mae, rmse, mape * 100))