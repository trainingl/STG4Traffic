import sys
sys.path.append('../')

import os
import copy
import time
import torch
from lib.utils import *

class Trainer(object):
    def __init__(self, args, data_loader, scaler, model, loss, optimizer, lr_scheduler, cl=True, new_training_method=True):
        super(Trainer, self).__init__()
        self.args = args
        self.data_loader = data_loader
        self.train_loader = data_loader['train_loader']  
        self.val_loader = data_loader['val_loader']
        self.test_loader = data_loader['test_loader']
        self.scaler = scaler
        # model, loss_func, optimizer, lr_scheduler
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

        self.iter = 1
        self.task_level = 1
        self.cl = cl
        self.horizon = args.horizon
        self.step = args.step_size1
        self.new_training_method = new_training_method
        self.batches_seen = 0

    def train_epoch(self):
        train_loss = []
        train_rmse = []
        train_mape = []
        self.model.train()
        self.train_loader.shuffle()
        for idx, (x, y, ycl) in enumerate(self.train_loader.get_iterator()):
            self.batches_seen += 1
            trainx = torch.Tensor(x).to(self.args.device)
            trainy = torch.Tensor(y).to(self.args.device)
            trainycl = torch.Tensor(ycl).to(self.args.device)

            self.optimizer.zero_grad()
            if self.cl:
                # curriculum learning
                trainx = trainx.transpose(1, 3)     # (B, 1, N, T)
                trainy = trainy.transpose(1, 3)     # (B, 1, N, T)
                trainycl = trainycl.transpose(1, 3)
                if idx % self.args.step_size2 == 0:
                    perm = np.random.permutation(range(self.args.num_nodes))
                num_sub = int(self.args.num_nodes / self.args.num_split)
                for j in range(self.args.num_split):
                    if j != self.args.num_split - 1:
                        id = perm[j * num_sub:(j + 1) * num_sub]
                    else:
                        id = perm[j * num_sub:]
                    id = torch.tensor(id).to(self.args.device)
                    tx = trainx[:, :, id, :]
                    ty = trainy[:, :, id, :]

                    real_val = ty[:, 0, :, :]
                    output = self.model(tx, idx=id)
                    output = output.transpose(1, 3)  # (B, 1, N, T)
                    real = torch.unsqueeze(real_val, dim=1)
                    # 预测的是归一化的结果, 所以需要反归一化
                    predict = self.scaler.inverse_transform(output)
                    if self.iter % self.step == 0 and self.task_level <= self.horizon:
                        self.task_level += 1
                    loss = self.loss(predict[:, :, :, :self.task_level],
                                    real[:, :, :, :self.task_level], 0.0)
                    mape = masked_mape(predict[:, :, :, :self.task_level],
                                            real[:, :, :, :self.task_level], 0.0).item()
                    rmse = masked_rmse(predict[:, :, :, :self.task_level],
                                            real[:, :, :, :self.task_level], 0.0).item()
            else:
                output = self.model(trainx)
                # 预测的是归一化的结果, 所以需要反归一化
                predict = self.scaler.inverse_transform(output)
                loss = self.loss(predict, trainy[:, :, :, 0:1], 0.0)
                mape = masked_mape(predict, trainy[:, :, :, 0:1], 0.0).item()
                rmse = masked_rmse(predict, trainy[:, :, :, 0:1], 0.0).item()                
            loss.backward()

            if self.args.grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()
            self.iter += 1
            train_loss.append(loss.item())
            train_mape.append(mape)
            train_rmse.append(rmse)
        
        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)
        return mtrain_loss, mtrain_rmse, mtrain_mape


    def val_epoch(self):
        valid_loss = []
        valid_mape = []
        valid_rmse = []
        self.model.eval()
        with torch.no_grad():
            for _, (x, y) in enumerate(self.val_loader.get_iterator()):
                validx = torch.Tensor(x).to(self.args.device)
                validy = torch.Tensor(y).to(self.args.device)
                if self.cl:
                    validx = validx.transpose(1, 3)  # (B, 1, N, T)
                    validy = validy.transpose(1, 3)  # (B, 1, N, T)
                    real_val = validy[:, 0, :, :]
                    output = self.model(validx)  # (B, T, N, 1)
                    output = output.transpose(1, 3)
                    predict = self.scaler.inverse_transform(output)
                    real = torch.unsqueeze(real_val, dim=1)
                    loss = self.loss(predict, real, 0.0)
                    mape = masked_mape(predict, real, 0.0).item()
                    rmse = masked_rmse(predict, real, 0.0).item()
                else:
                    output = self.model(validx)  # (B, T, N, 1)
                    # 预测的是归一化的结果, 所以需要反归一化
                    predict = self.scaler.inverse_transform(output)
                    loss = self.loss(predict, validy[:, :, :, 0:1], 0.0)
                    mape = masked_mape(predict, validy[:, :, :, 0:1], 0.0).item()
                    rmse = masked_rmse(predict, validy[:, :, :, 0:1], 0.0).item()  
                valid_loss.append(loss.item())
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
        self.test(self.args, self.model, self.data_loader, self.scaler, self.logger)


    def test(self, args, model, data_loader, scaler, logger, save_path=None):
        if save_path != None:
            model.load_state_dict(torch.load(save_path))
            model.to(args.device)
            print("load saved model...")
        model.eval()
        outputs = []
        realy = torch.Tensor(data_loader['y_test']).to(args.device)
        realy = realy[:, :, :, 0:1]   # (B, T, N, 1)
        with torch.no_grad():
            for _, (x, y) in enumerate(data_loader['test_loader'].get_iterator()):
                testx = torch.Tensor(x).to(args.device)
                testy = torch.Tensor(y).to(args.device)
                if self.cl:
                    testx = testx.transpose(1, 3)  # (B, 1, N, T)
                    testy = testy.transpose(1, 3)  # (B, 1, N, T)
                    preds = self.model(testx)  # (B, T, N, 1)
                else:
                    preds = self.model(testx)   # (B, T, N, 1)
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
            metrics = metric(pred, real)
            log = 'Evaluate model for horizon {:2d}, MAE: {:.4f}, MAPE: {:.4f}, RMSE: {:.4f}'
            logger.info(log.format(i + 1, metrics[0], metrics[1], metrics[2]))
            mae.append(metrics[0])
            mape.append(metrics[1])
            rmse.append(metrics[2])
        logger.info('On average over 12 horizons, MAE: {:.4f}, MAPE: {:.4f}, RMSE: {:.4f}'.format(np.mean(mae), np.mean(mape), np.mean(rmse)))