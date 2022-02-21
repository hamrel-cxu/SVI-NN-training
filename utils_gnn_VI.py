import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score
from matplotlib.ticker import MaxNLocator
from scipy.linalg import block_diag
from scipy.linalg import sqrtm
import networkx as nx
import matplotlib.pyplot as plt
import sys
import pickle
import importlib as ipb
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import torch
import pickle
import networkx as nx
import random
import matplotlib.ticker as mtick
import seaborn as sns
import scipy
from scipy.sparse import csr_matrix, hstack, kron
import matplotlib.transforms as transforms
from sklearn.metrics import classification_report
import numpy as np
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
import matplotlib
import scipy.io
import os
import numpy as np
import matplotlib
import time
import matplotlib as mpl
mpl.use('pgf')
mpl.rc('text', usetex=True)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams['font.size'] = 24
plt.rcParams['axes.titlesize'] = 24
plt.rcParams['figure.titlesize'] = 30

# Put everything together
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GCN_train():
    # NOTE: it can generically be anything but for simplicity, assume it is a two-layer GCN
    def __init__(self, model_to_train, train_loader, test_loader, model_get_data=None, test_loader_true=None, more_layers=False):
        '''
        Input:
            model_to_train: initialized two-layer GCN model.
            train_loader or test_loader: a dataloader of a list of graph data
            model_get_data: ground_truth model, which can be empty.
            test_loader_true: test_data_loader that know the true graph; only used in simulation and can be empty
        '''
        self.model_to_train = model_to_train
        para_dict = model_to_train.state_dict()
        self.H = para_dict['conv1.lin.weight'].shape[0]
        data = next(iter(train_loader))
        self.batch_size = data.num_graphs
        self.C = data.num_features
        self.model_get_data = model_get_data
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.test_loader_true = test_loader_true
        self.more_layers = more_layers
        if more_layers:
            # Not actually used anymore
            self.H2 = para_dict['conv2.lin.weight'].shape[0]

    def training_and_eval(self, num_epochs, compute_para_err=False, output_dim=1, model_to_feature_ls=[], loss_type='MSE', Adam=False, freq=1, freeze_BN=False):
        '''
        Output:
            Either training loss, test loss, and test error for real data
            Or parameter recovery error, l2/linf prediction error of mean, and test loss for simulation
        '''
        if self.model_get_data == None:
            # We are in real-data experiment, so that what we will return differ
            train_loss_vanilla = []
            test_loss_vanilla = []
            train_error_vanilla = []
            test_error_vanilla = []
            train_f1weight_vanilla = []
            test_f1weight_vanilla = []
            for epoch in range(num_epochs):
                if freeze_BN and epoch >= int(num_epochs/2):
                    self.model_to_train.freeze_BN = True
                train_loss = train_revised_all_layer(self.train_loader,
                                                     model_to_train=self.model_to_train, output_dim=output_dim, model_to_feature_ls=model_to_feature_ls, more_layers=self.more_layers, loss_type=loss_type, Adam=Adam)
                print(train_loss)
                if np.mod(epoch+1, freq) == 0:
                    loss_train, loss_test, error_train, error_test, f1weight_train, f1weight_test = evaluation(
                        self.train_loader, self.test_loader, model_to_eval=self.model_to_train, output_dim=output_dim, loss_type=loss_type)
                    train_loss_vanilla.append(loss_train)
                    test_loss_vanilla.append(loss_test)
                    train_error_vanilla.append(error_train)
                    test_error_vanilla.append(error_test)
                    train_f1weight_vanilla.append(f1weight_train)
                    test_f1weight_vanilla.append(f1weight_test)
            return [train_loss_vanilla, test_loss_vanilla, train_error_vanilla, test_error_vanilla, train_f1weight_vanilla, test_f1weight_vanilla]
        else:
            # In simulation
            para_dict = self.model_get_data.state_dict()
            if compute_para_err:
                W2 = para_dict['conv2.lin.weight'].cpu().detach().numpy()
                b2 = para_dict['conv2.bias'].cpu().detach().numpy()
            else:
                W2 = []
                b2 = []
            para_error_vanilla, pred_l2error_vanilla, pred_linferror_vanilla, pred_loss_vanilla = full_SGDtrain_simulation(
                num_epochs, self.train_loader, self.test_loader, self.model_get_data, self.model_to_train, W2, b2, self.test_loader_true, model_to_feature_ls=model_to_feature_ls, loss_type=loss_type)
            return [para_error_vanilla, pred_l2error_vanilla, pred_linferror_vanilla, pred_loss_vanilla]


def get_train_test_loader(X_train, X_test, Y_train, Y_test, edge_index, batch_size):
    train_data_torch = []
    test_data_torch = []
    for Xtrain, Ytrain in zip(X_train, Y_train):
        train_temp = Data(x=torch.from_numpy(Xtrain.copy()).type(
            torch.float), edge_index=edge_index, y=torch.from_numpy(Ytrain).type(torch.float))
        train_data_torch.append(train_temp)
    for Xtest, Ytest in zip(X_test, Y_test):
        test_temp = Data(x=torch.from_numpy(Xtest.copy()).type(
            torch.float), edge_index=edge_index, y=torch.from_numpy(Ytest).type(torch.float))
        test_data_torch.append(test_temp)
    train_loader = DataLoader(train_data_torch, batch_size=batch_size)
    test_loader = DataLoader(test_data_torch, batch_size=batch_size)
    return [train_loader, test_loader]

# Data generation


def get_traffic_train_test(num_neighbor=3, d=5, sub=False):
    # Traffic flow multi-class detection
    with open(f'Data/flow_frame_train_0.7_no_drop_data.p', 'rb') as fp:
        Xtrain = pickle.load(fp)
    with open(f'Data/flow_frame_test_0.7_no_drop_data.p', 'rb') as fp:
        Xtest = pickle.load(fp)
    with open(f'Data/true_anomalies.p', 'rb') as fp:
        Yvals = pickle.load(fp)
    Ytrain = Yvals.iloc[:Xtrain.shape[0], :]
    Ytest = Yvals.iloc[Xtrain.shape[0]:, :]
    if sub:
        N = int(Xtrain.shape[0]/10)  # 50% or /2 already pretty good
        N1 = int(Xtest.shape[0]/10)
        Xtrain = Xtrain.iloc[-N:]
        Xtest = Xtest.iloc[:N1]
        Ytrain = Ytrain.iloc[-N:]
        Ytest = Ytest.iloc[:N1]
    with open(f'Data/sensor_neighbors.p', 'rb') as fp:
        neighbor_dict = pickle.load(fp)
    # Define edge index
    sensors = list(Xtrain.columns)
    sensors_dict = {i: j for (i, j) in zip(sensors, range(len(sensors)))}
    edge_index = []
    # num_neighbor = 3
    for k, sensor in enumerate(sensors):
        neighbors = neighbor_dict[sensor]
        for p in range(num_neighbor):
            edge_index.append([k, sensors_dict[neighbors[p]]])
    edge_index = torch.from_numpy(np.array(edge_index).T).type(torch.long)
    # Define graphs, similarly as the solar data
    X_train = []
    X_test = []
    Y_train = []
    Y_test = []
    # d = 5
    for t in range(d, Xtrain.shape[0]):
        X_train.append(np.flip(Xtrain.iloc[t - d:t].to_numpy().T, 1))
        Y_train.append(Ytrain.iloc[t].to_numpy())
    for t in range(Xtest.shape[0]):
        if t < d:
            temp = np.c_[np.flip(Xtest.iloc[:t].to_numpy().T, 1),
                         np.flip(Xtrain.iloc[-(d - t):].to_numpy().T, 1)]
        else:
            temp = np.flip(Xtest.iloc[t - d:t].to_numpy().T, 1)
        X_test.append(temp)
        Y_test.append(Ytest.iloc[t].to_numpy())
    return [X_train, X_test, Y_train, Y_test, edge_index]


def get_solar_train_test(train_data, test_data, days, d=5):
    # d is the dimension of the input signal, which intuitively is the memory depth
    # The training data starts at index d, where each row is X=\omega^-d_t=[\omega_t-1,...,\omega_t-d] \in R^{K-by-d}
    X_train = []
    X_test = []
    Y_train = []
    Y_test = []
    for t in range(d, days):
        X_train.append(np.flip(train_data[t - d:t], 0).T)
        Y_train.append(train_data[t])
    for t in range(days):
        if t < d:
            temp = temp = np.r_[np.flip(test_data[:t], 0), np.flip(train_data[-(d - t):], 0)]
        else:
            temp = np.flip(test_data[t - d:t], 0)
        X_test.append(temp.T)
        Y_test.append(test_data[t])
    return [X_train, X_test, Y_train, Y_test]


def get_simulation_data(model_get_data, Nnow, edge_index, n, C, train=True, torch_seed=0):
    seed = 121232212+torch_seed+123323
    if train:
        seed = 1212312+torch_seed+123323
    X = []
    Y = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for i in range(Nnow):
        np.random.seed(i+seed)
        Xtrain = np.random.random((n, C)).astype(np.float32)
        X.append(Xtrain)
        train_temp = Data(x=torch.from_numpy(Xtrain), edge_index=edge_index).to(device)
        # Y_train.append(torch.round(model_get_data(train_temp)).cpu().detach().numpy().flatten())
        pred_prob = model_get_data(train_temp).cpu().detach().numpy()
        if i <= 2:
            print(f'True Prob is {pred_prob}')
        Y_train_temp = np.array([np.random.choice([0, 1], size=1, p=[1-i[0], i[0]])[0]
                                 for i in pred_prob]).astype(np.float32)
        # Y_train_temp = np.round(pred_prob).flatten().astype(np.float32)
        Y.append(Y_train_temp)
    return [X, Y]


# Train the network

def train_revised_all_layer(train_loader, model_to_train=None, output_dim=1, model_to_feature_ls=[],  more_layers=False, loss_type='MSE', Adam=False):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if model_to_train is None:
        raise ValueError('No model specified')
    # Model in training mode, see https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch
    model_to_train.train()
    # Cannot use weight decay, o/w the frozen parameters are still updated
    if Adam:
        lr = 0.001  # was 0.001
        optimizer = torch.optim.Adam(model_to_train.parameters(), lr=lr)
    else:
        lr = 0.001  # was 0.001
        optimizer = torch.optim.SGD(model_to_train.parameters(),
                                    lr=lr, momentum=0.99, nesterov=True)
        # optimizer = torch.optim.SGD(model_to_train.parameters(),
        #                             lr=lr, momentum=0.99, nesterov=True)
    # optimizer = torch.optim.Adam(model_to_train.parameters(), lr=0.005, weight_decay=0.001)
    loss_all = 0
    if output_dim == 1:
        crit = torch.nn.MSELoss() if loss_type == 'MSE' else torch.nn.BCELoss()
    else:
        crit = torch.nn.MSELoss() if loss_type == 'MSE' else F.nll_loss
    print(f'Loss function is {crit}')
    tot_num = 0
    for batch, data in enumerate(train_loader):
        data = data.to(device)
        # Otherwise, the gradient would be a combination of the old gradient, which you have already used to update your model parameters, and the newly-computed gradient.
        optimizer.zero_grad()
        if model_to_train.freeze_BN:
            # Set to evaluation mode, so that running mean and var no longer update
            if more_layers:
                model_to_train.BN1.eval()
                model_to_train.BN2.eval()
            else:
                model_to_train.BN.eval()
        output = model_to_train(data).to(device)
        if output_dim == 1:
            label = data.y.to(device).unsqueeze(1)
        else:
            label = F.one_hot(data.y.to(torch.int64)).to(
                torch.float) if loss_type == 'MSE' else data.y.type(torch.LongTensor).to(device)
        loss = crit(output, label)
        # loss.requires_grad = True
        loss.backward()  # Back Propagation. Computes dloss/dx for every parameter x which has requires_grad=True
        # if len(model_to_feature_ls) > 0 and model_to_train.batch_norm:
        #     print(model_to_train.BN.weight.grad)
        if len(model_to_feature_ls) > 0:
            # 1. Collect all parameters in model_to_train and all layers
            Theta_dict = model_to_train.state_dict()
            all_layers = list(model_to_train.children())
            # Because we have the feature layers, which are not actually what we need
            # One BN layer
            tot_layer = int((len(all_layers)+1) /
                            3) if model_to_train.batch_norm else int(len(all_layers)/2)
            for (curr_layer, child) in enumerate(model_to_train.children()):
                if curr_layer >= tot_layer:
                    continue
                # 2. Get the nonlinear feature at each layer:
                feature = eval(f'model_to_train(data,feature{curr_layer+1}=True)').to(device)
                n = feature.shape[0]  # Numer of samples
                # For the bias
                feature = torch.cat((feature, torch.ones(n, 1).to(device)),
                                    1).to(device)  # n-by-(H+1)
                # 5. Compute gradient and update
                if curr_layer == tot_layer-1:
                    output = model_to_train(data).to(device)
                    Y = data.y.reshape(n, 1).to(device) if output_dim == 1 else F.one_hot(
                        data.y.to(torch.int64)).to(device)
                    res = (output - Y).to(device)
                else:
                    res = eval(
                        f'model_to_train.layer{curr_layer+1}_x.grad').to(device)
                # breakpoint()
                grad = torch.transpose(torch.matmul(torch.transpose(
                    feature, 0, 1), res), 0, 1).clone().detach()/data.num_graphs
                # NOTE: this part is for batch normalization!
                # If before previous layer activation
                if model_to_train.batch_norm and curr_layer < tot_layer-1:
                    rvar = model_to_train.rvar if curr_layer == 0 else eval(
                        f'model_to_train.rvar{curr_layer}')
                    weight = model_to_train.weight if curr_layer == 0 else eval(
                        f'model_to_train.weight{curr_layer}')
                    grad[:, -1] = grad[:, -1] * (rvar/weight)
                    grad[:, :-1] = grad[:, :-1] * (rvar/weight)[:, None]
                # # If after previous layer activation
                # if model_to_train.batch_norm and curr_layer > 0:
                #     # Adam already do normalization, so we just skip this
                #     grad[:, :-1] = grad[:, :-1] * \
                #         (model_to_train.rvar/model_to_train.weight)[None, :]
                # 6. Update gradient
                i = 0
                for param in child.parameters():
                    if i == 0:
                        param.grad = grad[:, -
                                          1].clone().detach().to(device)
                        # if curr_layer == 0:
                        #     print('Current bias')
                        #     print(param)
                        #     print(param.grad)
                    else:
                        param.grad = grad[:, :-1].clone().detach().to(device)
                        # if curr_layer == 0:
                        #     print('Current weight')
                        #     print(param)
                        #     print(param.grad)
                    i += 1
                # # NOTE: due to batch normalization, we need to change parameter from W to \tilde W and then change back after optimizer.step()
                # If after previous layer activation
                # if model_to_train.batch_norm and curr_layer < tot_layer-1:
                #     old_dict = model_to_train.state_dict()
                #     old_dict['conv1.bias'] = (old_dict['conv1.bias']-model_to_train.rmean) * \
                #         (model_to_train.weight/model_to_train.rvar)+model_to_train.bias
                #     old_dict['conv1.lin.weight'] = old_dict['conv1.lin.weight'] * \
                #         (model_to_train.weight/model_to_train.rvar)[:, None]
                #     model_to_train.load_state_dict(old_dict)
        loss_all += data.num_graphs * loss.item()
        optimizer.step()  # Gardient Descent
        # if len(model_to_feature_ls) > 0:
        #     # NOTE: due to batch normalization, we need to change parameter from W to \tilde W and then change back after optimizer.step()
        #     # If after previous layer activation
        #     if model_to_train.batch_norm and curr_layer < tot_layer-1:
        #         old_dict = model_to_train.state_dict()
        #         old_dict['conv1.bias'] = (old_dict['conv1.bias']-model_to_train.bias) * \
        #             (model_to_train.rvar/model_to_train.weight)+model_to_train.rmean
        #         old_dict['conv1.lin.weight'] = old_dict['conv1.lin.weight'] * \
        #             (model_to_train.rvar/model_to_train.weight)[:, None]
        #         model_to_train.load_state_dict(old_dict)
        tot_num += data.num_graphs
    return loss_all / tot_num


def evaluation(train_loader, test_loader, model_to_eval=None, output_dim=1, loss_type='MSE'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if model_to_eval is None:
        raise ValueError('No model specified')
    model_to_eval.eval()  # Model in evaluation mode, dropout deprecated
    losses = []
    errors = []
    f1weights = []
    for data_loader in [train_loader, test_loader]:
        loss_all = 0
        correct_all = 0
        tot_num = 0
        tot_num1 = 0
        y_pred = []
        y_true = []
        if output_dim == 1:
            crit = torch.nn.MSELoss() if loss_type == 'MSE' else torch.nn.BCELoss()
        else:
            crit = torch.nn.MSELoss() if loss_type == 'MSE' else F.nll_loss
        print(f'Loss function is {crit}')
        # print(f'Output dimension is {output_dim}')
        for data in data_loader:
            data = data.to(device)
            output = model_to_eval(data)
            if output_dim == 1:
                pred_label = torch.round(output)
            else:
                pred_label = torch.argmax(output, dim=1)
            y_pred += pred_label.cpu().detach().numpy().tolist()
            if output_dim == 1:
                label = data.y.to(device).unsqueeze(1)
            else:
                label = F.one_hot(data.y.to(torch.int64)).to(
                    torch.float) if loss_type == 'MSE' else data.y.type(torch.LongTensor).to(device)
            y_true += data.y.cpu().detach().numpy().tolist()
            # basically it is N^{-1} \sum_i Y_i*ln(P(Y_i=1))+(1-Y_i)*ln(1-P(Y_i=1)). Here, the P(Y_i=1) is the output by the model.
            loss = crit(output, label)
            loss_all += data.num_graphs * loss.item()
            correct_all += (pred_label.flatten() == data.y).sum().float()
            tot_num += len(label)
            tot_num1 += data.num_graphs
        losses.append(loss_all / tot_num1)
        errors.append((1-correct_all/tot_num).item())
        f1weights.append(f1_score(y_true, y_pred, average='weighted'))
        print(f'{loss_type} loss is {loss_all}')
        print(f'Prediction error is {(1-correct_all/tot_num).item()}')
    loss_train, loss_test = losses
    error_train, error_test = errors
    f1weight_train, f1weight_test = f1weights
    return [loss_train, loss_test, error_train, error_test, f1weight_train, f1weight_test]


def full_SGDtrain_simulation(num_epochs, train_loader, test_loader, model_get_data, model_to_train, W2=[], b2=[], data_loader_true=None, model_to_feature_ls=[], loss_type='MSE', Adam=False):
    # NOTE, the main thing is that error are computed differently from real data, becauase true data are randomly draws from posterior, rather than rounded.
    para_error_vanilla = []
    pred_l2error_vanilla = []
    pred_linferror_vanilla = []
    pred_loss_vanilla = []
    for epoch in range(num_epochs):
        print(f'{epoch} epochs ran')
        train_revised_all_layer(train_loader, model_to_train,
                                model_to_feature_ls=model_to_feature_ls, more_layers=False, loss_type=loss_type, Adam=Adam)
        para_err, l2_err, linf_err, loss_actual = evaluation_simulation(
            test_loader, model_get_data, model_to_train, W2, b2, data_loader_true, loss_type=loss_type)
        if np.mod(epoch, int(num_epochs//10)) == 0:
            print(
                f'[Para err, l2 err, linf err, entropy loss] at {epoch} is \n {[para_err, l2_err, linf_err, loss_actual]}')
        para_error_vanilla.append(para_err)
        pred_l2error_vanilla.append(l2_err)
        pred_linferror_vanilla.append(linf_err)
        pred_loss_vanilla.append(loss_actual)
    return [para_error_vanilla, pred_l2error_vanilla, pred_linferror_vanilla, pred_loss_vanilla]


def evaluation_simulation(data_loader, model_get_data, model_to_eval, W2, b2, data_loader_true=None, loss_type='MSE'):
    '''
    Input:
        When we estimate the graph incorrectly, we should not apply model_get_data on data_loader, but instead on data_loader_true, because the former uses incorrect data_loader
    '''
    old_dict = model_to_eval.state_dict()
    # If really evaluate, it will be the last layer bias and weight
    b2_est = old_dict['conv1.bias'].cpu().detach().numpy()
    W2_est = old_dict['conv1.lin.weight'].cpu().detach().numpy()
    para_err = 10000
    if W2 != []:
        para_err = np.linalg.norm(np.append(W2, b2)-np.append(W2_est, b2_est)) / \
            np.linalg.norm(np.append(W2, b2))
    l2_err_ls = []
    linf_err_ls = []
    loss_ls = []
    model_get_data.eval()
    model_to_eval.eval()
    if data_loader_true is not None:
        data_loader_true1 = iter(data_loader_true)
    data = next(iter(data_loader))
    n = int(len(data.y)/data.num_graphs)
    crit = torch.nn.MSELoss() if loss_type == 'MSE' else torch.nn.BCELoss()
    print(f'Loss function is {crit}')
    for data in data_loader:
        if data_loader_true is not None:
            data_true = next(data_loader_true1)
        else:
            data_true = data
        true_posterior = model_get_data(data_true).cpu().detach().numpy()
        pred_posterior = model_to_eval(data).cpu().detach().numpy()
        for i in range(data.num_graphs):
            l2_err_ls.append(np.linalg.norm(true_posterior[i*n:(i+1)*n]-pred_posterior[i*n:(i+1)*n]) /
                             np.linalg.norm(true_posterior[i*n:(i+1)*n]))
            linf_err_ls.append(np.linalg.norm(true_posterior[i*n:(i+1)*n]-pred_posterior[i*n:(i+1)*n],
                                              ord=np.inf))
        true_y = data.y.to(device).unsqueeze(1)
        # test_loss_true = crit(model_get_data(data_true), true_y).cpu().detach().numpy()
        test_loss_pred = crit(model_to_eval(data), true_y).cpu().detach().numpy()
        # loss_ls.append(np.abs(test_loss_true-test_loss_pred)/np.abs(test_loss_true))
        loss_ls.append(test_loss_pred)
    l2_err = np.mean(l2_err_ls)
    linf_err = np.mean(linf_err_ls)
    loss = np.mean(loss_ls)
    return [para_err, l2_err, linf_err, loss]


# Plot


def GNN_VI_layer_plt_real_data(num_epochs, losses_vanilla,
                               losses_VI, fully_connected, city, H_vanilla, H_VI, savefig=True, no_layer=False, loss_type='MSE', Adam=False, single_plot=False, batch_norm=False, freq=1, freeze_BN=False):
    train_loss_vanilla, train_loss_vanillaSE, test_loss_vanilla, test_loss_vanillaSE, train_error_vanilla, train_error_vanillaSE, test_error_vanilla, test_error_vanillaSE, train_f1weight_vanilla, train_f1weight_vanillaSE, test_f1weight_vanilla, test_f1weight_vanillaSE = losses_vanilla
    train_loss_VI, train_loss_VISE, test_loss_VI, test_loss_VISE, train_error_VI, train_error_VISE, test_error_VI, test_error_VISE, train_f1weight_VI, train_f1weight_VISE, test_f1weight_VI, test_f1weight_VISE = losses_VI
    alpha = 0.1
    xaxis = range(len(train_loss_vanilla))
    SGD_label = 'Adam' if Adam else 'SGD'
    opt_type = '_Adam' if Adam else ''
    if single_plot:
        fig, ax = plt.subplots(figsize=(10, 6))
        loctype = 'upper right'
        ax.plot(train_error_vanilla, linestyle='dashed',
                label=f'{SGD_label} Training', color='black')
        ax.fill_between(xaxis, train_error_vanilla-train_error_vanillaSE, train_error_vanilla+train_error_vanillaSE,
                        color='black', alpha=0.1)
        ax.plot(test_error_vanilla, linestyle='solid',
                label=f'{SGD_label} Test', color='black')
        ax.fill_between(xaxis, test_error_vanilla-test_error_vanillaSE, test_error_vanilla+test_error_vanillaSE,
                        color='black', alpha=0.1)
        ax.plot(train_error_VI, linestyle='dashed',
                label=f'SVI Training', color='orange')
        ax.fill_between(xaxis, train_error_VI-train_error_VISE, train_error_VI+train_error_VISE,
                        color='orange', alpha=0.1)
        ax.plot(test_error_VI, linestyle='solid',
                label=f'SVI Test', color='orange')
        ax.fill_between(xaxis, test_error_VI-test_error_VISE, test_error_VI+test_error_VISE,
                        color='orange', alpha=0.1)
        # ax.set_title(
        #     r'Classification Error')
        ax.set_ylabel('Error')
        ax.set_xlabel('Epoch')
        labels = [item*freq for item in ax.get_xticks()]
        labels = np.array(labels, dtype=int)
        ax.set_xticklabels(labels)
        ax.grid()
        ax.legend(ncol=2, loc='lower right', bbox_to_anchor=(1, -0.45), fontsize=20)
    else:
        tot = 3
        fig, ax = plt.subplots(1, tot, figsize=(10*tot, 5))
        # NOTE: training error is evaluated at EACH epoch where test error only every several epoches, so I decide to not plot the training error now
        # ax[0].plot(train_loss_vanilla,label='Train Loss')
        if city == 'solar_CA':
            start = 0.17
            end = 0.3
        if city == 'solar_LA':
            start = 0.18
            end = 0.3
        if city == 'Traffic':
            start = 0.75
            end = 1.1
        ax[0].set_ylim([start, end])
        ax[0].yaxis.set_ticks(np.arange(start, end, (end-start)/10))
        loctype = 'upper right'
        if no_layer:
            SGD_label += ' (one layer)'
        ax[0].plot(train_loss_vanilla, linestyle='dashed',
                   label=f'{SGD_label} Training', color='black')
        ax[0].fill_between(xaxis, train_loss_vanilla-train_loss_vanillaSE, train_loss_vanilla+train_loss_vanillaSE,
                           color='black', alpha=0.1)
        ax[0].plot(test_loss_vanilla, linestyle='solid',
                   label=f'{SGD_label} Test', color='black')
        ax[0].fill_between(xaxis, test_loss_vanilla-test_loss_vanillaSE, test_loss_vanilla+test_loss_vanillaSE,
                           color='black', alpha=0.1)
        ax[0].plot(train_loss_VI, linestyle='dashed',
                   label=f'SVI Training', color='orange')
        ax[0].fill_between(xaxis, train_loss_VI-train_loss_VISE, train_loss_VI+train_loss_VISE,
                           color='orange', alpha=0.1)
        ax[0].plot(test_loss_VI, linestyle='solid',
                   label=f'SVI Test', color='orange')
        ax[0].fill_between(xaxis, test_loss_VI-test_loss_VISE, test_loss_VI+test_loss_VISE,
                           color='orange', alpha=0.1)
        # ax[0].set_title(
        #     f'{loss_type} loss')
        ax[0].set_ylabel('Loss')
        ax[0].set_xlabel('Epoch')
        ax[0].grid(which='both')
        # ax[0].set_yscale('log')
        # ax[0].set_xscale('log')
        # ax[0].legend(loc=loctype)
        if city == 'solar_CA':
            start = 0.25
            end = 0.6
        if city == 'solar_LA':
            start = 0.25
            end = 0.6
        if city == 'Traffic':
            start = 0.32
            end = 0.57
        ax[1].set_ylim([start, end])
        ax[1].yaxis.set_ticks(np.arange(start, end, (end-start)/10))
        loctype = 'upper right'
        ax[1].plot(train_error_vanilla, linestyle='dashed',
                   label=f'{SGD_label} Training', color='black')
        ax[1].fill_between(xaxis, train_error_vanilla-train_error_vanillaSE, train_error_vanilla+train_error_vanillaSE,
                           color='black', alpha=0.1)
        ax[1].plot(test_error_vanilla, linestyle='solid',
                   label=f'{SGD_label} Test', color='black')
        ax[1].fill_between(xaxis, test_error_vanilla-test_error_vanillaSE, test_error_vanilla+test_error_vanillaSE,
                           color='black', alpha=0.1)
        ax[1].plot(train_error_VI, linestyle='dashed',
                   label=f'SVI Training', color='orange')
        ax[1].fill_between(xaxis, train_error_VI-train_error_VISE, train_error_VI+train_error_VISE,
                           color='orange', alpha=0.1)
        ax[1].plot(test_error_VI, linestyle='solid',
                   label=f'SVI Test', color='orange')
        ax[1].fill_between(xaxis, test_error_VI-test_error_VISE, test_error_VI+test_error_VISE,
                           color='orange', alpha=0.1)
        # ax[1].set_title(
        #     r'Classification Error')
        ax[1].set_ylabel('Error')
        ax[1].set_xlabel('Epoch')
        ax[1].grid()
        # ax[1].set_yscale('log')
        # ax[1].set_xscale('log')
        # ax[1].legend(loc=loctype)
        if city == 'solar_CA':
            start = 0.4
            end = 0.75
        if city == 'solar_LA':
            start = 0.3
            end = 0.75
        if city == 'Traffic':
            start = 0.4
            end = 0.65
        ax[2].set_ylim([start, end])
        ax[2].yaxis.set_ticks(np.arange(start, end, (end-start)/10))
        loctype = 'lower right'
        ax[2].plot(train_f1weight_vanilla, linestyle='dashed',
                   label=f'{SGD_label} Training', color='black')
        ax[2].fill_between(xaxis, train_f1weight_vanilla-train_f1weight_vanillaSE, train_f1weight_vanilla+train_f1weight_vanillaSE,
                           color='black', alpha=0.1)
        ax[2].plot(test_f1weight_vanilla, linestyle='solid',
                   label=f'{SGD_label} Test', color='black')
        ax[2].fill_between(xaxis, test_f1weight_vanilla-test_f1weight_vanillaSE, test_f1weight_vanilla+test_f1weight_vanillaSE,
                           color='black', alpha=0.1)
        ax[2].plot(train_f1weight_VI, linestyle='dashed',
                   label=f'SVI Training', color='orange')
        ax[2].fill_between(xaxis, train_f1weight_VI-train_f1weight_VISE, train_f1weight_VI+train_f1weight_VISE,
                           color='orange', alpha=0.1)
        ax[2].plot(test_f1weight_VI, linestyle='solid',
                   label=f'SVI Test', color='orange')
        ax[2].fill_between(xaxis, test_f1weight_VI-test_f1weight_VISE, test_f1weight_VI+test_f1weight_VISE,
                           color='orange', alpha=0.1)
        # ax[2].set_title(
        #     r'Weighted $F_1$ score')
        ax[2].set_ylabel(r'Weighted $F_1$ score')
        ax[2].set_xlabel('Epoch')
        ax[2].grid()
        fig.tight_layout()
        ax[2].legend(ncol=4, loc='lower right', bbox_to_anchor=(1, -0.46))
        for curr_fig in ax:
            curr_fig.set_xlim(0, len(xaxis))
            labels = [item*freq for item in curr_fig.get_xticks()]
            labels = np.array(labels, dtype=int)
            print(labels)
            print(curr_fig.get_xticks())
            curr_fig.set_xticklabels(labels)
    if savefig:
        layer = '_no_layer' if no_layer else '_all_layer'
        sing_idx = '_single' if single_plot else ''
        bnorm = '_bnorm' if batch_norm else ''
        half = '_half' if freeze_BN else ''
        if fully_connected:
            suffix = '_graph_fully_connected'
        else:
            suffix = '_graph_inferred'
        fig.savefig(f'Vanilla_vs_vanilla_VI_{suffix}_{city}_vanillahidden_{H_vanilla}_VIhidden_{H_VI}_new_{layer}_{loss_type}{opt_type}{sing_idx}{bnorm}{half}.pdf',
                    dpi=300, bbox_inches='tight', pad_inches=0)
    else:
        return fig


def simulation_plot(para_error_vanilla, pred_l2error_vanilla, pred_loss_vanilla, pred_linferror_vanilla, para_error_VI, pred_l2error_VI, pred_loss_VI, pred_linferror_VI, para_error_vanillaSE, pred_l2error_vanillaSE, pred_loss_vanillaSE, pred_linferror_vanillaSE, para_error_VISE, pred_l2error_VISE, pred_loss_VISE, pred_linferror_VISE, plot_para_recovery=True, loss_type='MSE', Adam=False):
    plt.rcParams['font.size'] = 26
    plt.rcParams['axes.titlesize'] = 24
    ncol = 3
    if plot_para_recovery:
        ncol = 4
    fig, ax = plt.subplots(1, ncol, figsize=(10*ncol, 6), sharex=True)
    xaxis = range(1, len(pred_l2error_VI)+1)
    i = 0
    loctype = 'upper right'
    SGD_label = 'Adam' if Adam else 'SGD'
    if plot_para_recovery:
        ax[0].plot(para_error_vanilla,
                   label=f'{SGD_label}', color='black')
        ax[0].fill_between(xaxis, para_error_vanilla-para_error_vanillaSE, para_error_vanilla+para_error_vanillaSE,
                           color='black', alpha=0.1)
        ax[0].plot(para_error_VI,
                   label=f'SVI', color='orange')
        ax[0].fill_between(xaxis, para_error_VI-para_error_VISE, para_error_VI+para_error_VISE,
                           color='orange', alpha=0.1)
        ax[0].set_title(
            r'Relatie parameter recovery error'+'\n' + r'$||\hat{\Theta}-\Theta||_2/||\Theta||_2$')
        i = 1
        ax[0].set_ylabel('Relative Error')
        ax[0].set_xlabel('Epoch')
        ax[0].grid()
        # ax[0].set_yscale('log')
        # ax[0].set_xscale('log')
        ax[0].legend(ncol=2, loc=loctype)
    ax[i].plot(pred_l2error_vanilla,
               label=f'{SGD_label}', color='black')
    ax[i].fill_between(xaxis, pred_l2error_vanilla-pred_l2error_vanillaSE, pred_l2error_vanilla+pred_l2error_vanillaSE,
                       color='black', alpha=0.1)
    ax[i].plot(pred_l2error_VI,
               label=f'SVI', color='orange')
    ax[i].fill_between(xaxis, pred_l2error_VI-pred_l2error_VISE, pred_l2error_VI+pred_l2error_VISE,
                       color='orange', alpha=0.1)
    ax[i].set_title(
        r'Relative test error in posterior prediction---$l_2$ norm'+'\n' + r'$||P_{\hat{\Theta}}(Y|X)-P_{\Theta}(Y|X)||_2/||P_{\Theta}(Y|X)||_2$')
    ax[i].set_ylabel('Relative Error')
    ax[i].set_xlabel('Epoch')
    ax[i].grid()
    # ax[i].set_yscale('log')
    # ax[i].set_xscale('log')
    ax[i].legend(ncol=2, loc=loctype)
    ax[i+1].plot(pred_loss_vanilla,
                 label=f'{SGD_label}', color='black')
    ax[i+1].fill_between(xaxis, pred_loss_vanilla-pred_loss_vanillaSE, pred_loss_vanilla+pred_loss_vanillaSE,
                         color='black', alpha=0.1)
    ax[i+1].plot(pred_loss_VI,
                 label=f'SVI', color='orange')
    ax[i+1].fill_between(xaxis, pred_loss_VI-pred_loss_VISE, pred_loss_VI+pred_loss_VISE,
                         color='orange', alpha=0.1)
    ax[i +
        1].set_title(f'Relative test error in {loss_type} loss \n' + r'$|L(\hat{\Theta})-L(\Theta)|/|L(\Theta)|$')
    ax[i+1].set_ylabel('Loss')
    ax[i+1].set_xlabel('Epoch')
    ax[i+1].grid()
    # ax[i+1].set_yscale('log')
    # ax[i+1].set_xscale('log')
    ax[i+1].legend(ncol=2, loc=loctype)
    ax[i+2].plot(pred_linferror_vanilla,
                 label=f'{SGD_label}', color='black')
    ax[i+2].fill_between(xaxis, pred_linferror_vanilla-pred_linferror_vanillaSE, pred_linferror_vanilla+pred_linferror_vanillaSE,
                         color='black', alpha=0.1)
    ax[i+2].plot(pred_linferror_VI,
                 label=f'SVI', color='orange')
    ax[i+2].fill_between(xaxis, pred_linferror_VI-pred_linferror_VISE, pred_linferror_VI+pred_linferror_VISE,
                         color='orange', alpha=0.1)
    ax[i+2].set_title(r'Relative test error in posterior prediction---$l_{\infty}$ norm'+'\n' +
                      r'$||P_{\hat{\Theta}}(Y|X)-P_{\Theta}(Y|X)||_{\infty}$')
    ax[i+2].set_ylabel('Absolute Error')
    ax[i+2].set_xlabel('Epoch')
    ax[i+2].grid()
    # ax[i+2].set_yscale('log')
    # ax[i+2].set_xscale('log')
    ax[i+2].legend(ncol=2, loc=loctype)
    # fig.suptitle('Relative test error of predmeter recovery error and \n posterior probability prediction errors over graph nodes')
    fig.tight_layout()
    return fig


def simulation_plot_know_est_graph(SGD_know_graph, VI_know_graph, SGD_est_graph, VI_est_graph, loss_type='MSE', Adam=False, make_fig=False, single_plot=False):
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 20
    plt.rcParams['axes.labelsize'] = 24
    plt.rcParams['legend.fontsize'] = 21
    fig = 0
    para_error_vanilla, para_error_vanillaSE, pred_l2error_vanilla, pred_l2error_vanillaSE, pred_linferror_vanilla, pred_linferror_vanillaSE, pred_loss_vanilla, pred_loss_vanillaSE = SGD_know_graph
    para_error_VI, para_error_VISE, pred_l2error_VI, pred_l2error_VISE, pred_linferror_VI, pred_linferror_VISE, pred_loss_VI, pred_loss_VISE = VI_know_graph
    para_error_vanilla_est, para_error_vanilla_estSE, pred_l2error_vanilla_est, pred_l2error_vanilla_estSE, pred_linferror_vanilla_est, pred_linferror_vanilla_estSE, pred_loss_vanilla_est, pred_loss_vanilla_estSE = SGD_est_graph
    para_error_VI_est, para_error_VI_estSE, pred_l2error_VI_est, pred_l2error_VI_estSE, pred_linferror_VI_est, pred_linferror_VI_estSE, pred_loss_VI_est, pred_loss_VI_estSE = VI_est_graph
    SGD_label = 'Adam (Graph Known)' if Adam else 'SGD (Graph Known)'
    SGD_label1 = 'Adam (Graph Est)' if Adam else 'SGD (Graph Est)'
    xaxis = range(1, len(pred_l2error_VI)+1)
    if single_plot:
        fig, ax = plt.subplots(figsize=(10, 5))
        start = 0.06
        end = 0.26
        ax.set_ylim([start, end])
        ax.yaxis.set_ticks(np.arange(start, end, (end-start)/10))
        ax.plot(pred_linferror_vanilla,
                label=f'{SGD_label}', color='black')
        ax.fill_between(xaxis, pred_linferror_vanilla-pred_linferror_vanillaSE, pred_linferror_vanilla+pred_linferror_vanillaSE,
                        color='black', alpha=0.1)
        ax.plot(pred_linferror_vanilla_est,
                label=f'{SGD_label1}', color='black', linestyle='dotted')
        ax.fill_between(xaxis, pred_linferror_vanilla_est-pred_linferror_vanilla_estSE, pred_linferror_vanilla_est+pred_linferror_vanilla_estSE,
                        color='black', alpha=0.1)
        ax.plot(pred_linferror_VI,
                label=f'SVI {SGD_label[3:]}', color='orange')
        ax.fill_between(xaxis, pred_linferror_VI-pred_linferror_VISE, pred_linferror_VI+pred_linferror_VISE,
                        color='orange', alpha=0.1)
        ax.plot(pred_linferror_VI_est,
                label=f'SVI {SGD_label1[3:]}', color='orange', linestyle='dotted')
        ax.fill_between(xaxis, pred_linferror_VI_est-pred_linferror_VI_estSE, pred_linferror_VI_est+pred_linferror_VI_estSE,
                        color='orange', alpha=0.1)
        # ax.set_title(r'Relative test error in posterior prediction---$l_{\infty}$ norm'+'\n' +
        #                   r'$||P_{\hat{\Theta}}(Y|X)-P_{\Theta}(Y|X)||_{\infty}$')
        ax.set_ylabel(r'$l_{\infty}$ error')
        ax.set_xlabel('Epoch')
        ax.grid(which='both')
        # ax.set_yscale('log')
        # ax.set_yscale('log')
        # fig.tight_layout()
        ax.legend(ncol=2, loc='lower right', bbox_to_anchor=(1, -0.48))
    if make_fig:
        plt.rcParams['font.size'] = 26
        plt.rcParams['axes.titlesize'] = 24
        ncol = 3
        fig, ax = plt.subplots(1, ncol, figsize=(10*ncol, 6), sharex=True)
        i = 0
        start = 0.06
        end = 0.2
        ax[i].set_ylim([start, end])
        ax[i].yaxis.set_ticks(np.arange(start, end, (end-start)/10))
        ax[i].plot(pred_l2error_vanilla,
                   label=f'{SGD_label}', color='black')
        ax[i].fill_between(xaxis, pred_l2error_vanilla-pred_l2error_vanillaSE, pred_l2error_vanilla+pred_l2error_vanillaSE,
                           color='black', alpha=0.1)
        ax[i].plot(pred_l2error_vanilla_est,
                   label=f'{SGD_label1}', color='black', linestyle='dotted')
        ax[i].fill_between(xaxis, pred_l2error_vanilla_est-pred_l2error_vanilla_estSE, pred_l2error_vanilla_est+pred_l2error_vanilla_estSE,
                           color='black', alpha=0.1)
        ax[i].plot(pred_l2error_VI,
                   label=f'SVI {SGD_label[3:]}', color='orange')
        ax[i].fill_between(xaxis, pred_l2error_VI-pred_l2error_VISE, pred_l2error_VI+pred_l2error_VISE,
                           color='orange', alpha=0.1)
        ax[i].plot(pred_l2error_VI_est,
                   label=f'SVI {SGD_label1[3:]}', color='orange', linestyle='dotted')
        ax[i].fill_between(xaxis, pred_l2error_VI_est-pred_l2error_VI_estSE, pred_l2error_VI_est+pred_l2error_VI_estSE,
                           color='orange', alpha=0.1)
        # ax[i].set_title(
        #     r'Relative test error in posterior prediction---$l_2$ norm'+'\n' + r'$||P_{\hat{\Theta}}(Y|X)-P_{\Theta}(Y|X)||_2/||P_{\Theta}(Y|X)||_2$')
        ax[i].set_ylabel(r'$l_2$ error')
        ax[i].set_xlabel('Epoch')
        ax[i].grid()
        # ax[i].set_yscale('log')
        # ax[i].set_xscale('log')
        start = 0.247
        end = 0.27
        ax[i+1].set_ylim([start, end])
        ax[i+1].yaxis.set_ticks(np.arange(start, end, (end-start)/10))
        ax[i+1].plot(pred_loss_vanilla,
                     label=f'{SGD_label}', color='black')
        ax[i+1].fill_between(xaxis, pred_loss_vanilla-pred_loss_vanillaSE, pred_loss_vanilla+pred_loss_vanillaSE,
                             color='black', alpha=0.1)
        ax[i+1].plot(pred_loss_VI,
                     label=f'SVI {SGD_label[3:]}', color='orange')
        ax[i+1].plot(pred_loss_vanilla_est,
                     label=f'{SGD_label1}', color='black', linestyle='dotted')
        ax[i+1].fill_between(xaxis, pred_loss_vanilla_est-pred_loss_vanilla_estSE, pred_loss_vanilla_est+pred_loss_vanilla_estSE,
                             color='black', alpha=0.1)
        ax[i+1].fill_between(xaxis, pred_loss_VI-pred_loss_VISE, pred_loss_VI+pred_loss_VISE,
                             color='orange', alpha=0.1)
        ax[i+1].plot(pred_loss_VI_est,
                     label=f'SVI {SGD_label1[3:]}', color='orange', linestyle='dotted')
        ax[i+1].fill_between(xaxis, pred_loss_VI_est-pred_loss_VI_estSE, pred_loss_VI_est+pred_loss_VI_estSE,
                             color='orange', alpha=0.1)
        # ax[i +
        #     1].set_title(f'Relative test error in {loss_type} loss \n' + r'$|L(\hat{\Theta})-L(\Theta)|/|L(\Theta)|$')
        ax[i+1].set_ylabel('Loss')
        ax[i+1].set_xlabel('Epoch')
        ax[i+1].grid()
        # ax[i+1].set_yscale('log')
        # ax[i+1].set_xscale('log')
        start = 0.06
        end = 0.26
        ax[i+2].set_ylim([start, end])
        ax[i+2].yaxis.set_ticks(np.arange(start, end, (end-start)/10))
        ax[i+2].plot(pred_linferror_vanilla,
                     label=f'{SGD_label}', color='black')
        ax[i+2].fill_between(xaxis, pred_linferror_vanilla-pred_linferror_vanillaSE, pred_linferror_vanilla+pred_linferror_vanillaSE,
                             color='black', alpha=0.1)
        ax[i+2].plot(pred_linferror_vanilla_est,
                     label=f'{SGD_label1}', color='black', linestyle='dotted')
        ax[i+2].fill_between(xaxis, pred_linferror_vanilla_est-pred_linferror_vanilla_estSE, pred_linferror_vanilla_est+pred_linferror_vanilla_estSE,
                             color='black', alpha=0.1)
        ax[i+2].plot(pred_linferror_VI,
                     label=f'SVI {SGD_label[3:]}', color='orange')
        ax[i+2].fill_between(xaxis, pred_linferror_VI-pred_linferror_VISE, pred_linferror_VI+pred_linferror_VISE,
                             color='orange', alpha=0.1)
        ax[i+2].plot(pred_linferror_VI_est,
                     label=f'SVI {SGD_label1[3:]}', color='orange', linestyle='dotted')
        ax[i+2].fill_between(xaxis, pred_linferror_VI_est-pred_linferror_VI_estSE, pred_linferror_VI_est+pred_linferror_VI_estSE,
                             color='orange', alpha=0.1)
        # ax[i+2].set_title(r'Relative test error in posterior prediction---$l_{\infty}$ norm'+'\n' +
        #                   r'$||P_{\hat{\Theta}}(Y|X)-P_{\Theta}(Y|X)||_{\infty}$')
        ax[i+2].set_ylabel(r'$l_{\infty}$ error')
        ax[i+2].set_xlabel('Epoch')
        ax[i+2].grid()
        # ax[i+2].set_yscale('log')
        fig.tight_layout()
        ax[i+2].legend(ncol=4, loc='lower right', bbox_to_anchor=(1, -0.38))
        # ax[i+2].set_xscale('log')
    para_error_vanilla, para_error_vanillaSE, pred_l2error_vanilla, pred_l2error_vanillaSE, pred_linferror_vanilla, pred_linferror_vanillaSE, pred_loss_vanilla, pred_loss_vanillaSE = SGD_know_graph
    para_error_VI, para_error_VISE, pred_l2error_VI, pred_l2error_VISE, pred_linferror_VI, pred_linferror_VISE, pred_loss_VI, pred_loss_VISE = VI_know_graph
    para_error_vanilla_est, para_error_vanilla_estSE, pred_l2error_vanilla_est, pred_l2error_vanilla_estSE, pred_linferror_vanilla_est, pred_linferror_vanilla_estSE, pred_loss_vanilla_est, pred_loss_vanilla_estSE = SGD_est_graph
    para_error_VI_est, para_error_VI_estSE, pred_l2error_VI_est, pred_l2error_VI_estSE, pred_linferror_VI_est, pred_linferror_VI_estSE, pred_loss_VI_est, pred_loss_VI_estSE = VI_est_graph
    long_ls = []
    long_ls += get_final_and_rel_err(pred_l2error_vanilla, pred_l2error_vanillaSE,
                                     pred_l2error_vanilla_est, pred_l2error_vanilla_estSE)
    long_ls += get_final_and_rel_err(pred_l2error_VI, pred_l2error_VISE,
                                     pred_l2error_VI_est, pred_l2error_VI_estSE)
    long_ls += get_final_and_rel_err(pred_loss_vanilla, pred_loss_vanillaSE,
                                     pred_loss_vanilla_est, pred_loss_vanilla_estSE)
    long_ls += get_final_and_rel_err(pred_loss_VI, pred_loss_VISE,
                                     pred_loss_VI_est, pred_loss_VI_estSE)
    long_ls += get_final_and_rel_err(pred_linferror_vanilla, pred_linferror_vanillaSE,
                                     pred_linferror_vanilla_est, pred_linferror_vanilla_estSE)
    long_ls += get_final_and_rel_err(pred_linferror_VI, pred_linferror_VISE,
                                     pred_linferror_VI_est, pred_linferror_VI_estSE)
    return [fig, long_ls]


def get_final_and_rel_err(array1, array11, array2, array22):
    # array1 contains good results
    return [array1[-1], array11[-1], array2[-1], array22[-1]]


def get_mean_and_se(array_ls):
    mean_vals = np.mean(array_ls, axis=0).flatten()
    se_ls = np.std(array_ls, axis=0).flatten()/np.sqrt(len(array_ls))
    return [mean_vals, se_ls]


def get_ls_err(result_dict):
    tot_result = result_dict.values()
    para_error = [res[0] for res in tot_result]
    pred_l2error = [res[1] for res in tot_result]
    pred_linferror = [res[2] for res in tot_result]
    pred_loss = [res[3] for res in tot_result]
    return [para_error, pred_l2error, pred_linferror, pred_loss]


def get_all(result_dict):
    error_ls, l2err_ls, linferr_ls, loss_ls = get_ls_err(result_dict)
    para_error, para_errorSE = get_mean_and_se(error_ls)
    pred_l2error, pred_l2errorSE = get_mean_and_se(l2err_ls)
    pred_linferror, pred_linferrorSE = get_mean_and_se(linferr_ls)
    pred_loss, pred_lossSE = get_mean_and_se(loss_ls)
    return [para_error, para_errorSE, pred_l2error, pred_l2errorSE, pred_linferror, pred_linferrorSE, pred_loss, pred_lossSE]


def get_ls_err_real(result_dict):
    tot_result = result_dict.values()
    train_loss = [res[0] for res in tot_result]
    test_loss = [res[1] for res in tot_result]
    train_error = [res[2] for res in tot_result]
    test_error = [res[3] for res in tot_result]
    train_f1weight = [res[4] for res in tot_result]
    test_f1weight = [res[5] for res in tot_result]
    return [train_loss, test_loss, train_error, test_error, train_f1weight, test_f1weight]


def get_all_real(result_dict):
    train_loss, test_loss, train_error, test_error, train_f1weight, test_f1weight = get_ls_err_real(
        result_dict)
    train_loss, train_lossSE = get_mean_and_se(train_loss)
    test_loss, test_lossSE = get_mean_and_se(test_loss)
    train_error, train_errorSE = get_mean_and_se(train_error)
    test_error, test_errorSE = get_mean_and_se(test_error)
    train_f1weight, train_f1weightSE = get_mean_and_se(train_f1weight)
    test_f1weight, test_f1weightSE = get_mean_and_se(test_f1weight)
    return [train_loss, train_lossSE, test_loss, test_lossSE, train_error, train_errorSE, test_error, test_errorSE, train_f1weight, train_f1weightSE, test_f1weight, test_f1weightSE]


def draw_graph(edge_index, edge_index_est, graph_type):
    G = nx.Graph()
    G.add_edges_from(edge_index.cpu().detach().numpy().T)
    pos = nx.circular_layout(G)
    fig_network, ax1 = plt.subplots(1, 2, figsize=(8, 4))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', ax=ax1[0])
    # ax1[0].set_title('True Graph')
    G = nx.Graph()
    G.add_edges_from(edge_index_est.cpu().detach().numpy().T)
    pos = nx.circular_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', ax=ax1[1])
    # ax1[1].set_title('Estimated Graph')
    fig_network.savefig(f'simulated_graph_{graph_type}.pdf',
                        dpi=300, bbox_inches='tight', pad_inches=0)


def get_edge_list(Y_train, n=10, fully_connected=True):
    if fully_connected:
        edge_index = torch.from_numpy(
            np.array([[a, b] for a in range(n) for b in range(n)]).T).type(torch.long)
    else:
        # Infer edge connection in a nearest neighbor fashion, by including connection among node k and all nodes whose training labels are the most similar to k (e.g., in terms of equality). The reason is that this likely indicates influence.
        # Always include itself
        Y_temp = np.array(Y_train)
        edge_index = []
        num_include = 4  # four nodes, including itself
        for k in range(n):
            same_num = np.array([np.sum(Y_temp[:, k] == Y_temp[:, j])
                                 for j in range(Y_temp.shape[1])])
            include_ones = same_num.argsort()[-num_include:][::-1]
            for j in include_ones:
                edge_index.append([k, j])
        # Also, to ensure the connection is symmetric, I delet all edges (k,j) if (j,k) not in graph. Although in reality, the solar radiation is likely directed so directed graph should be used.
        print(f'{len(edge_index)} directed edges initially')
        m = 0
        while m < len(edge_index):
            edge = edge_index[m]
            k, j = edge
            if [j, k] in edge_index:
                # print(f'{[j, k]}' in edge')
                m += 1
                continue
            else:
                # print(f'{edge} deleted b/c {[j, k]}' not in edge')
                idx = edge_index.index(edge)
                if idx > m:
                    m += 1
                edge_index.remove(edge)
        print(f'{len(edge_index)} undirected edges after deletion')
        edge_index = torch.from_numpy(np.array(edge_index).T)
    return edge_index


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def get_graph_conv(n, edge_index, output='GCN'):
    if output != 'GCN':
        raise ValueError('Not yet implemented')
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(list(map(tuple, edge_index.cpu().detach().numpy().T)))
    adjacency_matrix = nx.linalg.graphmatrix.adjacency_matrix(G)
    W = adjacency_matrix.toarray()
    if W.shape[0] < 15:
        print(f'Adjacency matrix is \n {W}')
    if check_symmetric(W) == False:
        raise ValueError('Matrix non-symmetric')
    if output == 'GCN':
        I_n = np.diag(np.ones(n))
        Atilde = W + I_n
        degrees = Atilde.sum(axis=1)
        Dtilde = np.diag(degrees)
        Dinvsqrttilde = sqrtm(np.linalg.inv(Dtilde))
        return Dinvsqrttilde.dot(Atilde.dot(Dinvsqrttilde))


def concatenat_to_one(Table, new_colindex, round_more=False):
    cols = list(Table.columns)
    tot = len(cols)
    r = 5 if round_more else 3
    stop = int(tot/2)
    rows = Table.shape[0]
    Table_new = np.zeros((rows, stop), dtype=object)
    for i in range(stop):
        if max(Table.iloc[:, 2*i]) < 0.0001:
            Table_new[:, i] = Table.iloc[:, 2*i].map(lambda x: str('{:.4f}'.format(
                x)))+Table.iloc[:, 2*i+1].map(lambda x: ' ('+str('{:.1e}'.format(x))+')')
        else:
            Table_new[:, i] = Table.iloc[:, 2 *
                                         i].map(lambda x: str('{:.4f}'.format(
                                             x)))+Table.iloc[:, 2*i+1].map(lambda x: ' ('+str('{:.1e}'.format(x))+')')
    Table_new = pd.DataFrame(Table_new, index=Table.index, columns=new_colindex)
    return Table_new


def G_reformat(G_oracle, percent_perturb=0.1, return_G=False):
    n = len(G_oracle.nodes)
    edges = list(G_oracle.edges)
    num_choose = int(percent_perturb*len(edges))
    print(f'Chose {num_choose} out of {len(edges)} edges for {n} nodes')
    remove_edges = random.sample(edges, num_choose)
    for edge in remove_edges:
        G_oracle.remove_edge(edge[0], edge[1])
    edges_to_add = [(i, j) for i in range(n) for j in range(i+1, n)]
    for edge in edges:
        edges_to_add.remove(edge)
    edges_to_add = random.sample(edges_to_add, num_choose)
    for edge in edges_to_add:
        G_oracle.add_edge(edge[0], edge[1])
    if return_G:
        return G_oracle
    else:
        pos = nx.circular_layout(G_oracle)
        fig_network, ax1 = plt.subplots(figsize=(6, 4))
        nx.draw(G_oracle, pos, with_labels=True, node_color='lightblue', ax=ax1)
        fig_network.savefig('random_graph_sensitive.pdf', dpi=300,
                            bbox_inches='tight', pad_inches=0)
        adjacency_matrix = nx.linalg.graphmatrix.adjacency_matrix(G_oracle)
        W = adjacency_matrix.toarray()
        return W
