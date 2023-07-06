import sys
sys.path.append('../')

import os
import copy
import time
import torch
import torch.nn as nn
from DMSTGCN_Utils import *
from lib.utils import *
from lib.evaluate import *

class Trainer(object):
    def __init__(self, args, data, train_loader, val_loader, test_loader, scaler, model, loss, optimizer, lr_scheduler):
        super(Trainer, self).__init__()
        self.args = args
        self.dataloader = data
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scaler = scaler

        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        # 日志与模型的保存路径
        self.best_path = os.path.join(args.log_dir, '{}_{}_best_model.pth'.format(args.dataset, args.model))
        if os.path.isdir(args.log_dir) == False and not args.debug:
            os.makedirs(args.log_dir, exist_ok=True)  # run.log
        self.logger = get_logger(args.log_dir, name=args.model, debug=args.debug)
        self.logger.info("Experiment log path in: {}".format(args.log_dir))
        self.logger.info(args)
        
    def train_epoch(self):
        train_loss = []
        train_rmse = []
        train_mape = []
        self.model.train()
        self.train_loader.shuffle()
        for _, (x, y, ind) in enumerate(self.train_loader.get_iterator()):
            trainx = torch.Tensor(x).to(self.args.device)
            trainy = torch.Tensor(y).to(self.args.device)
            trainx = trainx.transpose(1, 3)  # (B, D, N, T)
            trainx = nn.functional.pad(trainx, (1, 0, 0, 0))
            self.optimizer.zero_grad()
            output = self.model(trainx, ind)
            predict = self.scaler.inverse_transform(output)
            loss = self.loss(predict, trainy[:, :, :, 0:1], 0.0)
            loss.backward()
            mae = masked_mae(predict, trainy[:, :, :, 0:1], 0.0).item()
            rmse = masked_rmse(predict, trainy[:, :, :, 0:1], 0.0).item()
            mape = masked_mape(predict, trainy[:, :, :, 0:1], 0.0).item()

            if self.args.grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()
            train_loss.append(mae)
            train_rmse.append(rmse)
            train_mape.append(mape)
        
        mtrain_loss = np.mean(train_loss)
        mtrain_rmse = np.mean(train_rmse)
        mtrain_mape = np.mean(train_mape)
        return mtrain_loss, mtrain_rmse, mtrain_mape

    def val_epoch(self):
        valid_loss = []
        valid_mape = []
        valid_rmse = []
        self.model.eval()
        with torch.no_grad():
            for _, (x, y, ind) in enumerate(self.val_loader.get_iterator()):
                trainx = torch.Tensor(x).to(self.args.device)
                trainy = torch.Tensor(y).to(self.args.device)
                trainx = trainx.transpose(1, 3)  # (B, D, N, T)
                trainx = nn.functional.pad(trainx, (1, 0, 0, 0))
                output = self.model(trainx, ind)
                predict = self.scaler.inverse_transform(output)
                mae = masked_mae(predict, trainy[:, :, :, 0:1], 0.0).item()
                rmse = masked_rmse(predict, trainy[:, :, :, 0:1], 0.0).item()
                mape = masked_mape(predict, trainy[:, :, :, 0:1], 0.0).item()
                valid_loss.append(mae)
                valid_rmse.append(rmse)
                valid_mape.append(mape)
        
        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        return mvalid_loss, mvalid_rmse, mvalid_mape

    def train(self):
        self.logger.info("start training...")
        best_model = None
        best_loss = float('inf')
        not_improved_count = 0
        train_loss_list = []
        val_loss_list = []
        start_time = time.time()
        for epoch in range(1, self.args.epochs + 1):
            t1 = time.time()
            mtrain_loss, _, _ = self.train_epoch()
            t2 = time.time()
            mvalid_loss, mvalid_rmse, mvalid_mape = self.val_epoch()
            t3 = time.time()
            self.logger.info('Epoch {:03d}, Train Loss: {:.4f}, Valid Loss: {:.4f}, Valid RMSE: {:.4f}, Valid MAPE: {:.4f}, Training Time: {:.4f} secs, Inference Time: {:.4f} secs.'.format(epoch, mtrain_loss, mvalid_loss, mvalid_rmse, mvalid_mape, (t2 - t1), (t3 - t2)))
            train_loss_list.append(mtrain_loss)
            val_loss_list.append(mvalid_loss)
            if mtrain_loss > 1e6:
                self.logger.warning("Gradient explosion detected. Ending...")
                break
            if mvalid_loss < best_loss:
                best_loss = mvalid_loss
                not_improved_count = 0
                best_state = True
            else:
                not_improved_count += 1
                best_state = False
            
            # early stop is or not
            if self.args.early_stop:
                if not_improved_count == self.args.early_stop_patience:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                    "Training stops.".format(self.args.early_stop_patience))
                    break
            # save the best model
            if best_state == True:
                # self.logger.info("Current best model saved!")
                best_model = copy.deepcopy(self.model.state_dict())
                torch.save(best_model, self.best_path)

        training_time = time.time() - start_time
        self.logger.info("Total training time: {:.4f} min, best loss: {:.6f}".format((training_time / 60), best_loss))
        # save the best model to file
        self.logger.info("Saving current best model to " + self.best_path)

        # Let' test the model
        self.model.load_state_dict(best_model)
        self.test(self.args, self.model, self.dataloader, self.scaler, self.logger)

    
    def test(self, args, model, data_loader, scaler, logger, save_path=None):
        if save_path != None:
            model.load_state_dict(torch.load(save_path))
            model.to(args.device)
            print("load saved model...")
        model.eval()
        # test
        outputs = []
        realy = torch.Tensor(data_loader['y_test']).to(args.device)
        realy = realy[:, :, :, 0:1]   # (B, T, N, 1)
        with torch.no_grad():
            for _, (x, y, ind) in enumerate(data_loader['test_loader'].get_iterator()):
                testx = torch.Tensor(x).to(args.device)
                testy = torch.Tensor(y).to(args.device)
                testx = testx.transpose(1, 3)
                preds = self.model(testx, ind)   # (B, T, N, 1)
                outputs.append(preds)
        
        yhat = torch.cat(outputs, dim=0)
        yhat = yhat[:realy.size(0), ...]   # concat at batch_size
        mae = []
        rmse = []
        mape = []
        for i in range(args.horizon):
            # 预测的是归一化的结果, 所以需要反归一化
            pred = scaler.inverse_transform(yhat[:, i, :, :])  # (B, T, N, 1)
            real = realy[:, i, :, :]  # (B, T, N, 1)
            metrics = All_Metrics(pred, real, args.mae_thresh, args.mape_thresh)
            log = 'Evaluate model for horizon {:2d}, MAE: {:.4f}, RMSE: {:.4f}, MAPE: {:.4f}'
            logger.info(log.format(i + 1, metrics[0], metrics[1], metrics[2]))
            mae.append(metrics[0].cpu().numpy())
            rmse.append(metrics[1].cpu().numpy())
            mape.append(metrics[2].cpu().numpy())
        logger.info('On average over 12 horizons, MAE: {:.4f}, RMSE: {:.4f}, MAPE: {:.4f}'.format(np.mean(mae), np.mean(rmse), np.mean(mape)))