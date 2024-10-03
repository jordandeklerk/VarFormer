from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
import os
import time
import numpy as np


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        self.args = args
        self.label_len = args.label_len

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        # Use weight_decay only for BayesFormer
        if self.args.model == 'BayesFormer':
            model_optim = optim.Adam(
                self.model.parameters(),
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay
            )
        else:
            model_optim = optim.Adam(
                self.model.parameters(),
                lr=self.args.learning_rate
            )
        return model_optim

    def _select_criterion(self):
        return nn.MSELoss()

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # Decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # Encoder-Decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.model == 'BayesFormer':
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            if isinstance(outputs, tuple):
                                outputs = outputs[0]
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.model == 'BayesFormer':
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        if isinstance(outputs, tuple):
                            outputs = outputs[0]
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                # Determine feature dimension
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

                # Compute loss
                if self.args.model == 'BayesFormer':
                    pred = outputs
                    true = batch_y
                    loss = criterion(pred, true)
                else:
                    pred = outputs
                    true = batch_y
                    loss = criterion(pred, true)

                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        # Initialize scheduler based on model type
        if self.args.model == 'BayesFormer':
            scheduler = lr_scheduler.OneCycleLR(
                optimizer=model_optim,
                steps_per_epoch=train_steps,
                pct_start=self.args.pct_start,
                epochs=self.args.train_epochs,
                max_lr=self.args.learning_rate
            )
        else:
            scheduler = None  # Will use adjust_learning_rate for other models

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # Decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # Encoder-Decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.model == 'BayesFormer':
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            loss = self.model.calculate_loss(*outputs, batch_y)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            f_dim = -1 if self.args.features == 'MS' else 0
                            outputs = outputs[:, -self.args.pred_len:, f_dim:]
                            loss = criterion(outputs, batch_y[:, -self.args.pred_len:, f_dim:])
                else:
                    if self.args.model == 'BayesFormer':
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        loss = self.model.calculate_loss(*outputs, batch_y)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        loss = criterion(outputs, batch_y[:, -self.args.pred_len:, f_dim:])

                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print(f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f}")
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print(f'\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s')
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                # Update scheduler if using OneCycleLR
                if self.args.model == 'BayesFormer' and scheduler is not None:
                    scheduler.step()

            if self.args.model != 'BayesFormer' and scheduler is not None:
                adjust_learning_rate(model_optim, epoch + 1, self.args)

            print(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time}")

            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print(f"Epoch: {epoch + 1}, Steps: {train_steps} | "
                  f"Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f}")

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if scheduler is not None and self.args.model == 'BayesFormer':
                print(f'Updating learning rate to {scheduler.get_last_lr()[0]}')

        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test_flag=0):
        test_data, test_loader = self._get_data(flag='test')
        if test_flag:
            print('Loading model')
            self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')))

        preds = []
        trues = []
        oris = [] if self.args.model != 'BayesFormer' else None
        folder_path = os.path.join('./test_results', self.args.exp_type, setting)
        os.makedirs(folder_path, exist_ok=True)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # Decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # Encoder-Decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.model == 'BayesFormer':
                            outputs = self.model.forecast(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.model == 'BayesFormer':
                        outputs = self.model.forecast(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                # Determine feature dimension
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

                if self.args.model == 'BayesFormer':
                    recon_seq = outputs.detach().cpu().numpy()  # Reconstructed sequence
                    true = batch_y.detach().cpu().numpy()
                else:
                    recon_seq = outputs.detach().cpu().numpy()
                    true = batch_y.detach().cpu().numpy()

                preds.append(recon_seq)
                trues.append(true)

                if self.args.model != 'BayesFormer':
                    ori_seq = batch_x.detach().cpu().numpy()
                    oris.append(ori_seq)

                if i % 20 == 0:
                    input_np = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input_np[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input_np[0, :, -1], recon_seq[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, f'{i}.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        if self.args.model != 'BayesFormer' and oris is not None:
            oris = np.concatenate(oris, axis=0)

        print('Test shape:', preds.shape, trues.shape)

        if self.args.model != 'BayesFormer':
            oris = oris.reshape(-1, oris.shape[-2], oris.shape[-1])
            print('Original shape:', oris.shape)

        # Ensure preds and trues have the same shape
        min_len = min(preds.shape[1], trues.shape[1])
        preds = preds[:, :min_len, :]
        trues = trues[:, :min_len, :]

        print('Test shape after trimming:', preds.shape, trues.shape)

        # Save results
        results_folder = os.path.join('./results', self.args.exp_type, setting)
        os.makedirs(results_folder, exist_ok=True)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print(f'mse: {mse}, mae: {mae}')

        with open(os.path.join(results_folder, "result_long_term_forecast.txt"), 'a') as f:
            f.write(f"{setting}\n")
            f.write(f'mse: {mse}, mae: {mae}\n\n')

        np.save(os.path.join(results_folder, 'metrics.npy'), np.array([mae, mse, rmse, mape, mspe]))
        np.save(os.path.join(results_folder, 'pred.npy'), preds)
        np.save(os.path.join(results_folder, 'true.npy'), trues)
        
        if self.args.model != 'BayesFormer' and oris is not None:
            np.save(os.path.join(results_folder, 'ori.npy'), oris)

        return
