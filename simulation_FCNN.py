import pandas as pd
import json
import ast
from torch.autograd import Variable
from torch.utils.data import TensorDataset
import time
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import utils_gnn_VI_layer as utils_layer
from torch import nn
import sys
import torch
import importlib as ipb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats import norm
import matplotlib as mpl
mpl.use('pgf')
mpl.rc('text', usetex=True)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
'''Code Layout:
    The exact line numbers can be a little off
    1. One-layer Probit model: line 313-end. Change the data argument after rand_states as 'probit'
    2. Two-layer FCNN: line 313-end. Change the data argument after rand_states as 'moon'
'''

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def probit(seed, size, beta, bias, train=False):
    X = []
    Y = []
    rv = norm()
    for i in range(size):
        if train:
            np.random.seed(seed+i)
        else:
            np.random.seed(seed+12312+i)
        X_i = np.random.normal(0.05, 1, size=p).astype(np.float32)
        pred_prob = rv.cdf(X_i.dot(beta)+bias)[0]
        Y_i = np.random.choice([0, 1], size=1, p=[1-pred_prob, pred_prob])[0].astype(np.float32)
        X.append(X_i)
        Y.append(Y_i)
    X_t = torch.from_numpy(np.array(X)).to(torch.float32)
    Y_t = torch.from_numpy(np.array(Y)).to(torch.float32)
    return [X_t, Y_t]


class FC_one_layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(p, 1)

    def forward(self, x):
        x = self.fc1(x)
        probit = torch.distributions.normal.Normal(0, 1)
        return probit.cdf(x)


class FC_one_layer_feature(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(p, p)

    def forward(self, x):
        x = self.fc1(x)
        return x


# Below for change last layer, fully connected case


class CLF_NN_one_layer_relu(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.fc1 = nn.Linear(2, H)
        self.fc2 = nn.Linear(H, 2)

    def forward(self, x):
        # x = nn.ReLU(self.fc1(x))#2->32
        x = F.relu(self.fc1(x))  # 2->32
        self.layer1_x = Variable(x, requires_grad=True)
        layer2_x = self.fc2(self.layer1_x)  # 32->2
        return F.softmax(layer2_x, dim=1)


class CLF_NN_one_layer_relu_feature(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.fc1 = nn.Linear(2, H)
        self.fc2 = nn.Linear(H, H)

    def forward(self, x):
        # x = nn.ReLU(self.fc1(x))#2->32
        x = F.relu(self.fc1(x))  # 2->32
        x = self.fc2(x)  # 32->2
        return x


class CLF_NN_one_layer_relu_feature1(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 2)

    def forward(self, x):
        x = self.fc1(x)
        return x


def train_loop(dataloader, model, loss_fn, optimizer, verbose, model_to_feature_ls=[], batch_size=1, data='moon'):
    size = len(dataloader.dataset)
    train_loss, correct = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X).to(torch.float)
        if data != 'moon':
            Y = y.reshape(len(y), 1)
        else:
            Y = F.one_hot(y).to(torch.float)
        loss = loss_fn(pred, Y)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        if len(model_to_feature_ls) > 0:
            # 1. Collect all parameters in model and all layers
            Theta_dict = model.state_dict()
            all_layers = list(model.children())
            tot_layer = len(all_layers)
            for (curr_layer, child) in enumerate(model.children()):
                # 2. Get the nonlinear feature at each layer:
                model_to_feature_sub = model_to_feature_ls[curr_layer]
                if curr_layer > 0:
                    # Update previous weights only for 1st hidden layer onward
                    old_dict = model_to_feature_sub.state_dict()
                    for prev_layer in range(1, curr_layer+1):
                        old_dict[f'fc{prev_layer}.bias'] = Theta_dict[f'fc{prev_layer}.bias']
                        old_dict[f'fc{prev_layer}.weight'] = Theta_dict[f'fc{prev_layer}.weight']
                    model_to_feature_sub.load_state_dict(old_dict)
                feature = model_to_feature_sub(X).to(device)
                n = feature.shape[0]  # Numer of samples
                # For the bias
                feature = torch.cat((feature, torch.ones(n, 1).to(device)),
                                    1).to(device)  # n-by-(H+1)
                # 5. Compute gradient and update
                if curr_layer == tot_layer-1:
                    # if batch == 0:
                    #     print(f'Layer {curr_layer}')
                    output = model(X).to(device)
                    Y_new = y.reshape(n, 1).to(device) if data == 'probit' else F.one_hot(
                        y.to(torch.int64)).to(device)
                    res = (output - Y_new).to(device)
                else:
                    # if batch == 0:
                    #     print(f'Layer {curr_layer}')
                    res = eval(
                        f'model.layer{curr_layer+1}_x.grad').to(device)
                grad = torch.transpose(torch.matmul(torch.transpose(
                    feature, 0, 1), res), 0, 1).clone().detach()/batch_size
                # print(grad.shape)
                # 6. Update gradient
                i = 0
                for param in child.parameters():
                    if i == 1:
                        param.grad = grad[:, -
                                          1].clone().detach().to(device)
                    else:
                        param.grad = grad[:, :-1].clone().detach().to(device)
                    i += 1
        optimizer.step()
        if data == 'moon':
            train_loss += loss.item()
        else:
            train_loss += loss.item()*len(y)
        if data == 'moon':
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        else:
            correct += (torch.round(pred).flatten() == y).type(torch.float).sum().item()
        if verbose and (batch % 50 == 0):
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    # print(f'Train correct {correct}')
    train_loss /= size
    correct /= size
    if verbose:
        print(
            f"Train Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {train_loss:>8f} \n")
    return train_loss, correct


def test_loop(dataloader, model, loss_fn, verbose):
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            # pred = model(X)
            pred = model(X)
            if data != 'moon':
                Y = y.reshape(len(y), 1)
            else:
                Y = F.one_hot(y).to(torch.float)
            test_loss += loss_fn(pred, Y)*len(y)
            if data == 'moon':
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            else:
                correct += (torch.round(pred).flatten() == y).type(torch.float).sum().item()
    # print(f'Test correct {correct}')
    test_loss /= size
    correct /= size
    if verbose:
        print(
            f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss, correct


# (a)
def get_param_err(model, beta_true):
    beta_est = list(model.parameters())
    beta_est_np = torch.flatten(beta_est[0]).detach().numpy()
    beta_est_np = np.append(beta_est_np, torch.flatten(beta_est[1]).detach().numpy())
    return np.linalg.norm(beta_true-beta_est_np)/np.linalg.norm(beta_true)


def simulation_plot(train_loss_vanilla, train_acc_vanilla, test_acc_vanilla, test_loss_vanilla, train_loss_VI, train_acc_VI, test_acc_VI, test_loss_VI, train_loss_vanillaSE, train_acc_vanillaSE, test_acc_vanillaSE, test_loss_vanillaSE, train_loss_VISE, train_acc_VISE, test_acc_VISE, test_loss_VISE, data):
    plt.rcParams['font.size'] = 23
    plt.rcParams['axes.titlesize'] = 24
    ncol = 2
    height = 8 if data == 'moon' else 6
    fig, ax = plt.subplots(1, ncol, figsize=(8*ncol, height), sharex=True)
    xaxis = np.arange(len(train_loss_vanilla))
    loctype = 'upper right'
    if data == 'moon':
        start = 0
        end = 0.18
    else:
        start = 0
        end = 0.2
    ax[0].set_ylim([start, end])
    ax[0].yaxis.set_ticks(np.arange(start, end, (end-start)/10))
    ax[0].plot(train_loss_vanilla, linestyle='dashed',
               label='SGD Training', color='black')
    ax[0].fill_between(xaxis, train_loss_vanilla-train_loss_vanillaSE, train_loss_vanilla+train_loss_vanillaSE,
                       color='black', alpha=0.1)
    ax[0].plot(test_loss_vanilla, linestyle='solid',
               label='SGD Test', color='black')
    ax[0].fill_between(xaxis, test_loss_vanilla-test_loss_vanillaSE, test_loss_vanilla+test_loss_vanillaSE,
                       color='black', alpha=0.1)
    ax[0].plot(train_loss_VI, linestyle='dashed',
               label='SVI Training', color='orange')
    ax[0].fill_between(xaxis, train_loss_VI-train_loss_VISE, train_loss_VI+train_loss_VISE,
                       color='orange', alpha=0.1)
    ax[0].plot(test_loss_VI, linestyle='solid',
               label='SVI Test', color='orange')
    ax[0].fill_between(xaxis, test_loss_VI-test_loss_VISE, test_loss_VI+test_loss_VISE,
                       color='orange', alpha=0.1)
    # ax[0].set_title(
    #     r'MSE loss')
    ax[0].set_ylabel('Loss')
    ax[0].set_xlabel('Epoch')
    ax[0].grid()
    # ax[0].set_yscale('log')
    # ax[0].set_xscale('log')
    # ax[0].legend(loc=loctype)
    loctype = 'upper right'
    if data == 'moon':
        start = 0
        end = 0.45
    else:
        start = 0
        end = 0.55
    ax[1].set_ylim([start, end])
    ax[1].yaxis.set_ticks(np.arange(start, end, (end-start)/10))
    ax[1].plot(1-train_acc_vanilla, linestyle='dashed',
               label='SGD Training', color='black')
    ax[1].fill_between(xaxis, 1-train_acc_vanilla-train_acc_vanillaSE, 1-train_acc_vanilla+train_acc_vanillaSE,
                       color='black', alpha=0.1)
    ax[1].plot(1-test_acc_vanilla, linestyle='solid',
               label='SGD Test', color='black')
    ax[1].fill_between(xaxis, 1-test_acc_vanilla-test_acc_vanillaSE, 1-test_acc_vanilla+test_acc_vanillaSE,
                       color='black', alpha=0.1)
    ax[1].plot(1-train_acc_VI, linestyle='dashed',
               label='SVI Training', color='orange')
    ax[1].fill_between(xaxis, 1-train_acc_VI-train_acc_VISE, 1-train_acc_VI+train_acc_VISE,
                       color='orange', alpha=0.1)
    ax[1].plot(1-test_acc_VI, linestyle='solid',
               label='SVI Test', color='orange')
    ax[1].fill_between(xaxis, 1-test_acc_VI-test_acc_VISE, 1-test_acc_VI+test_acc_VISE,
                       color='orange', alpha=0.1)
    # ax[1].set_title(
    #     r'Classification Error')
    ax[1].set_ylabel('Error')
    ax[1].set_xlabel('Epoch')
    ax[1].grid()
    # ax[1].set_yscale('log')
    # ax[1].set_xscale('log')
    ax[1].legend(loc=loctype)
    fig.tight_layout()
    ax[1].legend(ncol=4, loc='lower right', bbox_to_anchor=(1, -0.35))
    # if para_recov_VI is not None:
    #     fig.suptitle(
    #         f'Average (SE) SGD and VI parameter recovery errors are \n {np.round(para_recov_SGD,2)} ({np.round(para_recov_SGDSE,5)}) and {np.round(para_recov_VI,2)} ({np.round(para_recov_VISE,5)})', y=1.15)
    # fig.subplots_adjust(bottom=-0.25)
    return fig


# NOTE; it depends on learning rate indeed

momen = 0.9
verbose = 0
rand_states = [1103, 1111, 1214]
data = 'moon'  # moon or probit
# loss_fn = nn.CrossEntropyLoss() if data == 'moon' else nn.MSELoss()
loss_fn = nn.MSELoss()
# In the latter case, it is p, the dimension of input feature, not hidden neuron number
H_ls = [8, 16, 32, 64] if data == 'moon' else [50, 100, 200]
lr = 0.15
lr_dict_moon = {64: lr, 32: lr, 8: lr, 16: lr}
rand_states = [1103, 1111, 1214]
optimizer_name = 'vanilla'
for H in H_ls:
    result_dict_SGD_para = []
    result_dict_VI_para = []
    result_dict_SGD = {}
    result_dict_VI = {}
    ipb.reload(sys.modules['utils_gnn_VI_layer'])
    for random_state in rand_states:
        if data == 'probit':
            p = H
            N = 2000
            N1 = 500
            learning_rate = 0.005
            epochs = 200  # Increase it
            batch_size = int(N/10)
            np.random.seed(1)
            beta = np.random.normal(-0.05, 1, p)  # So class zero and one are balanced
            bias = np.random.normal(-0.1, 1, 1)
            beta_true = np.append(beta, bias)
            X_train_t, y_train_t = probit(random_state, N, beta, bias, train=True)
            print(np.unique(y_train_t, return_counts=True))
            X_test_t, y_test_t = probit(random_state, N1, beta, bias, train=False)
            # np.unique(Y_test, return_counts=True)
        else:
            N = 1000
            batch_size = int(N/10)
            learning_rate = lr_dict_moon[H]
            epochs = 100
            X, y = make_moons(noise=0.1,
                              n_samples=N,
                              random_state=random_state)
            # standarize the features
            X = StandardScaler().fit_transform(X)

            # split training and testing
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=.5,
                random_state=random_state)
            X_train_t = torch.from_numpy(X_train).to(torch.float32)
            y_train_t = torch.from_numpy(y_train).to(torch.long)
            X_test_t = torch.from_numpy(X_test).to(torch.float32)
            y_test_t = torch.from_numpy(y_test).to(torch.long)
            plt.rcParams['font.size'] = 20
            h = .02  # step size in the mesh
            x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
            y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                 np.arange(y_min, y_max, h))
            fig = plt.figure(figsize=(4, 12))
            ax = fig.add_subplot(2, 1, 1)
            # Plot the training points
            ax.scatter(X_train[:, 0], X_train[:, 1],
                       c=y_train, cmap='jet', edgecolors='k')
            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
            plt.grid(color='0.95')
            # plt.title('training')

            ax = fig.add_subplot(2, 1, 2)
            # Plot the testing points
            ax.scatter(X_test[:, 0], X_test[:, 1],
                       c=y_test, cmap='jet', alpha=0.6, edgecolors='k')
            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
            # ax.set_xticks(())
            # ax.set_yticks(())
            plt.grid(color='0.95')
            # plt.title('testing')
            fig.savefig(f'Moon.pdf', dpi=300, bbox_inches='tight', pad_inches=0)
        train_dataset = TensorDataset(X_train_t, y_train_t)
        test_dataset = TensorDataset(X_test_t, y_test_t)
        # SGD
        start = time.time()
        train_loss_all, train_acc_all = np.zeros(epochs), np.zeros(epochs)
        test_loss_all, test_acc_all = np.zeros(epochs), np.zeros(epochs)
        torch.manual_seed(random_state)
        if data == 'probit':
            model = FC_one_layer()
        else:
            model = CLF_NN_one_layer_relu(H)
        # print(list(model.parameters()))
        optimizer = torch.optim.SGD(
            model.parameters(), lr=learning_rate, momentum=momen)
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=True)

        for t in range(epochs):
            if verbose:
                print(f"Epoch {t+1}\n-------------------------------")

            train_loss_all[t], train_acc_all[t] = train_loop(
                train_dataloader, model, loss_fn, optimizer, verbose, data=data)
            test_loss_all[t], test_acc_all[t] = test_loop(
                test_dataloader, model, loss_fn, verbose)

        end = time.time()
        print('SGD')
        print('elapsed time = %.4f s' % (end - start))
        print('learning rate=', learning_rate, ', train accuracy=',
              train_acc_all[t] * 100, '%, test accuracy=', test_acc_all[t] * 100, '%')
        result_dict_SGD[random_state] = [train_loss_all.tolist(
        ), train_acc_all.tolist(), test_loss_all.tolist(), test_acc_all.tolist()]
        if data != 'moon':
            result_dict_SGD_para.append(get_param_err(model, beta_true))
        # VI
        start = time.time()
        train_loss_all, train_acc_all = np.zeros(epochs), np.zeros(epochs)
        test_loss_all, test_acc_all = np.zeros(epochs), np.zeros(epochs)
        torch.manual_seed(random_state)
        if data == 'probit':
            model_VI = FC_one_layer()
            mod_feature = FC_one_layer_feature()
            old_dict = mod_feature.state_dict()
            old_dict['fc1.bias'] = torch.zeros(p)
            old_dict['fc1.weight'] = torch.diag(torch.ones(p))
            mod_feature.load_state_dict(old_dict)
            model_to_feature_ls = [mod_feature]
        else:
            model_VI = CLF_NN_one_layer_relu(H)
            # print(list(model_VI.parameters()))
            mod_feature = CLF_NN_one_layer_relu_feature(H)
            old_dict = mod_feature.state_dict()
            old_dict['fc2.bias'] = torch.zeros(H)
            old_dict['fc2.weight'] = torch.diag(torch.ones(H))
            mod_feature.load_state_dict(old_dict)
            mod_feature1 = CLF_NN_one_layer_relu_feature1()
            old_dict = mod_feature1.state_dict()
            old_dict['fc1.bias'] = torch.zeros(2)
            old_dict['fc1.weight'] = torch.diag(torch.ones(2))
            mod_feature1.load_state_dict(old_dict)
            model_to_feature_ls = [mod_feature1, mod_feature]
        optimizer = torch.optim.SGD(
            model_VI.parameters(), lr=learning_rate, momentum=momen)
        for t in range(epochs):
            if verbose:
                print(f"Epoch {t+1}\n-------------------------------")
            train_loss_all[t], train_acc_all[t] = train_loop(
                train_dataloader, model_VI, loss_fn, optimizer, verbose, model_to_feature_ls=model_to_feature_ls, batch_size=batch_size, data=data)
            test_loss_all[t], test_acc_all[t] = test_loop(
                test_dataloader, model_VI, loss_fn, verbose)
        if data != 'moon':
            result_dict_VI_para.append(get_param_err(model_VI, beta_true))
        end = time.time()
        print('VI')
        print('elapsed time = %.4f s' % (end - start))
        print('learning rate=', learning_rate, ', train accuracy=',
              train_acc_all[t] * 100, '%, test accuracy=', test_acc_all[t] * 100, '%')
        result_dict_VI[random_state] = [train_loss_all.tolist(), train_acc_all.tolist(),
                                        test_loss_all.tolist(), test_acc_all.tolist()]
    method = 'vanilla' if optimizer_name == 'vanilla' else 'Adam'
    train_loss_vanilla, train_loss_vanillaSE, train_acc_vanilla, train_acc_vanillaSE, test_loss_vanilla, test_loss_vanillaSE, test_acc_vanilla, test_acc_vanillaSE = utils_layer.get_all(
        result_dict_SGD)
    para_recov_SGD = None
    para_recov_SGDSE = None
    if len(result_dict_SGD_para) > 0:
        para_recov_SGD = np.mean(result_dict_SGD_para)
        para_recov_SGDSE = np.std(result_dict_SGD_para)
    train_loss_VI, train_loss_VISE, train_acc_VI, train_acc_VISE, test_loss_VI, test_loss_VISE, test_acc_VI, test_acc_VISE = utils_layer.get_all(
        result_dict_VI)
    para_recov_VI = None
    para_recov_VISE = None
    if len(result_dict_VI_para) > 0:
        para_recov_VI = np.mean(result_dict_VI_para)
        para_recov_VISE = np.std(result_dict_VI_para)
    if data == 'moon':
        Hmid = f'_H={H}'
    else:
        Hmid = f'_p={p}'
    # Save results
    if data != 'moon':
        result_dict_SGD['para_err'] = result_dict_SGD_para
        result_dict_VI['para_err'] = result_dict_VI_para
    name = f'FC_SGD_{method}Training_{data}{Hmid}_lr={learning_rate}'
    name1 = f'FC_VI_{method}Training_{data}{Hmid}_lr={learning_rate}'
    json_SGD = json.dumps(str(result_dict_SGD))
    json_VI = json.dumps(str(result_dict_VI))
    f = open(f"{name}.json", "w")
    # write json object to file
    f.write(json_SGD)
    # close file
    f.close()
    # open file for writing, "w"
    f = open(f"{name1}.json", "w")
    # write json object to file
    f.write(json_VI)
    # close file
    f.close()

# Generate table with standard errors
method = 'vanilla'
lr_dict_moon = {64: lr, 32: lr, 8: lr, 16: lr}
H_ls = [8, 16, 32, 64] if data == 'moon' else [50, 100, 200]
# Times 2 because the standard errors are also stored.
Table = np.zeros((len(H_ls), 2*8)) if data == 'moon' else np.zeros((len(H_ls), 2*10))
for data in [data]:
    for k, H in enumerate(H_ls):
        print(f'{data} data with H={H}')
        if data == 'moon':
            Hmid = f'_H={H}'
            columns = np.tile(['SGD train mean', 'SGD train SE', 'VI train mean', 'VI train SE',
                               'SGD test mean', 'SGD test SE', 'VI test mean', 'VI test SE'], 2)
            # To be used later, but divide in half
            type = np.repeat(['MSE loss', 'Classification error'], 8)
        else:
            p = H
            Hmid = f'_p={p}'
            columns1 = np.array(['SGD mean', 'SGD SE', 'VI mean', 'VI SE'])
            columns2 = np.tile(['SGD train mean', 'SGD train SE', 'VI train mean', 'VI train SE',
                                'SGD test mean', 'SGD test SE', 'VI test mean', 'VI test SE'], 2)
            columns = np.append(columns1, columns2)
            type1 = np.repeat(['Para recovery error'], 4)
            type2 = np.repeat(['MSE loss', 'Classification error'], 8)
            type = np.append(type1, type2)  # To be used later, but divide in half
        tuples = list(zip(*[type, columns]))
        index = pd.MultiIndex.from_tuples(tuples)
        if data == 'probit':
            learning_rate = 0.005
        else:
            learning_rate = lr_dict_moon[H]
        name = f'FC_SGD_{method}Training_{data}{Hmid}_lr={learning_rate}.json'
        name1 = f'FC_VI_{method}Training_{data}{Hmid}_lr={learning_rate}.json'
        with open(name, 'r') as j:
            result_dict_SGD = json.loads(j.read())
            result_dict_SGD = ast.literal_eval(result_dict_SGD)
        with open(name1, 'r') as j:
            result_dict_VI = json.loads(j.read())
            result_dict_VI = ast.literal_eval(result_dict_VI)
        if data == 'probit':
            key_val = 'para_err'
            result_dict_SGD_para = result_dict_SGD[key_val]
            para_recov_SGD, para_recov_SGDSE = np.mean(result_dict_SGD_para), np.std(
                result_dict_SGD_para)/len(result_dict_SGD_para)
            result_dict_VI_para = result_dict_VI[key_val]
            para_recov_VI, para_recov_VISE = np.mean(result_dict_VI_para), np.std(
                result_dict_VI_para)/len(result_dict_VI_para)
            result_dict_SGD.pop(key_val)
            result_dict_VI.pop(key_val)
        train_loss_vanilla, train_loss_vanillaSE, train_acc_vanilla, train_acc_vanillaSE, test_loss_vanilla, test_loss_vanillaSE, test_acc_vanilla, test_acc_vanillaSE = utils_layer.get_all(
            result_dict_SGD)
        train_loss_VI, train_loss_VISE, train_acc_VI, train_acc_VISE, test_loss_VI, test_loss_VISE, test_acc_VI, test_acc_VISE = utils_layer.get_all(
            result_dict_VI)
        fig = simulation_plot(train_loss_vanilla, train_acc_vanilla, test_acc_vanilla, test_loss_vanilla, train_loss_VI, train_acc_VI, test_acc_VI, test_loss_VI, train_loss_vanillaSE,
                              train_acc_vanillaSE, test_acc_vanillaSE, test_loss_vanillaSE, train_loss_VISE, train_acc_VISE, test_acc_VISE, test_loss_VISE, data)
        if data == 'moon':
            Hmid = f'_H={H}'
        else:
            Hmid = f'_p={p}'
        fig.savefig(f'FC_{method}Training_{data}{Hmid}_lr={learning_rate}.pdf',
                    dpi=300, bbox_inches='tight', pad_inches=0)
        if data == 'probit':
            long_ls = [para_recov_SGD, para_recov_SGDSE, para_recov_VI, para_recov_VISE, train_loss_vanilla[-1], train_loss_vanillaSE[-1], train_loss_VI[-1], train_loss_VISE[-1], test_loss_vanilla[-1], test_loss_vanillaSE[-1],
                       test_loss_VI[-1], test_loss_VISE[-1], 1-train_acc_vanilla[-1], train_acc_vanillaSE[-1], 1-train_acc_VI[-1], train_acc_VISE[-1], 1-test_acc_vanilla[-1], test_acc_vanillaSE[-1], 1-test_acc_VI[-1], test_acc_VISE[-1]]
        else:
            long_ls = [train_loss_vanilla[-1], train_loss_vanillaSE[-1], train_loss_VI[-1], train_loss_VISE[-1], test_loss_vanilla[-1], test_loss_vanillaSE[-1], test_loss_VI[-1], test_loss_VISE[-1],
                       1-train_acc_vanilla[-1], train_acc_vanillaSE[-1], 1-train_acc_VI[-1], train_acc_VISE[-1], 1-test_acc_vanilla[-1], test_acc_vanillaSE[-1], 1-test_acc_VI[-1], test_acc_VISE[-1]]
        Table[k] = long_ls
    Table = pd.DataFrame(Table, index=H_ls, columns=index)
    if data == 'probit':
        Table.index.name = r'Feature dimension'
    else:
        Table.index.name = '# Hidden nodes'
Table
columns = np.tile(['SGD train', 'VI train',
                   'SGD test', 'VI test'], 2)
type = np.repeat(['MSE loss', 'Classification error'], 4)  # To be used later, but divide in half
if data != 'moon':
    columns1 = np.array(['SGD', 'VI'])
    columns2 = np.tile(['SGD train', 'VI train',
                        'SGD test', 'VI test'], 2)
    columns = np.append(columns1, columns2)
    type1 = np.repeat(['Para recovery error'], 2)
    type2 = np.repeat(['MSE loss', 'Classification error'], 4)
    type = np.append(type1, type2)  # To be used later, but
tuples = list(zip(*[type, columns]))
new_colindex = pd.MultiIndex.from_tuples(tuples)
ipb.reload(sys.modules['utils_gnn_VI_layer'])
round_more = True if data == 'moon' else False
Table_new = utils_layer.concatenat_to_one(Table, new_colindex, round_more)

print(Table_new.to_latex(escape=False))
