import os
import numpy as np
import torch
import matplotlib.pyplot as plt



# def adjust_learning_rate(optimizer, epoch, args):
#     # lr = args.learning_rate * (0.2 ** (epoch // 2))
#     if args.lradj == 'type1':
#         lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
#     elif args.lradj == 'type2':
#         lr_adjust = {
#             2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
#             10: 5e-7, 15: 1e-7, 20: 5e-8
#         }
#     if epoch in lr_adjust.keys():
#         lr = lr_adjust[epoch]
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = lr
#         print('Updating learning rate to {}'.format(lr))


def adjust_learning_rate(optimizer, scheduler, epoch, args, printout=True):
    """
    Adjusts the learning rate based on the specified strategy in args.lradj.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer whose learning rate needs to be adjusted.
        scheduler (torch.optim.lr_scheduler): The scheduler associated with the optimizer.
        epoch (int): The current epoch number.
        args (Namespace): Parsed command-line arguments.
        printout (bool): Whether to print the updated learning rate.
    """
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    elif args.lradj == '3':
        lr_adjust = {epoch: args.learning_rate if epoch < 10 else args.learning_rate * 0.1}
    elif args.lradj == '4':
        lr_adjust = {epoch: args.learning_rate if epoch < 15 else args.learning_rate * 0.1}
    elif args.lradj == '5':
        lr_adjust = {epoch: args.learning_rate if epoch < 25 else args.learning_rate * 0.1}
    elif args.lradj == '6':
        lr_adjust = {epoch: args.learning_rate if epoch < 5 else args.learning_rate * 0.1}
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    else:
        lr_adjust = {}

    if epoch in lr_adjust:
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout:
            print(f'Updating learning rate to {lr}')


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """

    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                           Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        """
        Call method to check if early stopping should be triggered.

        Args:
            val_loss (float): Current validation loss.
            model (torch.nn.Module): Model being trained.
            path (str): Path to save the best model.
        """
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        """
        Saves model when validation loss decreases.

        Args:
            val_loss (float): Current validation loss.
            model (torch.nn.Module): Model being trained.
            path (str): Path to save the best model.
        """
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(path, 'checkpoint.pth'))
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler:
    """
    Standardize features by removing the mean and scaling to unit variance.
    """

    def __init__(self, mean, std):
        """
        Args:
            mean (np.array): Mean of the data.
            std (np.array): Standard deviation of the data.
        """
        self.mean = mean
        self.std = std

    def transform(self, data):
        """
        Transform the data using the stored mean and std.

        Args:
            data (np.array): Data to be transformed.

        Returns:
            np.array: Transformed data.
        """
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        """
        Inverse transform the data to original scale.

        Args:
            data (np.array): Data to be inverse transformed.

        Returns:
            np.array: Inverse transformed data.
        """
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def adjustment(gt, pred):
    """
    Adjusts the predictions based on ground truth anomalies.

    Args:
        gt (list or np.array): Ground truth anomaly labels.
        pred (list or np.array): Predicted anomaly labels.

    Returns:
        tuple: Adjusted ground truth and predictions.
    """
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    """
    Calculates the accuracy of predictions.

    Args:
        y_pred (np.array): Predicted labels.
        y_true (np.array): True labels.

    Returns:
        float: Accuracy score.
    """
    return np.mean(y_pred == y_true)
