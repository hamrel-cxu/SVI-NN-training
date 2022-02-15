import matplotlib
import numpy as np
import os
import pandas as pd
import scipy.io
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch.autograd import Variable
import importlib as ipb
import sys
import shutil
import ast
import json
import matplotlib.gridspec as gridspec
import copy
from scipy.stats import ortho_group
from scipy.sparse import csr_matrix
from sklearn.metrics import f1_score
import utils_gnn_VI as utils_layer
from matplotlib.ticker import MaxNLocator
from scipy.linalg import block_diag
from scipy.linalg import sqrtm
from scipy.stats import norm
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('pgf')  # Significantly reduces image size
mpl.rc('text', usetex=True)  # So label can incorporate latex as well
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams['font.size'] = 24
plt.rcParams['axes.titlesize'] = 24
plt.rcParams['figure.titlesize'] = 30

ipb.reload(sys.modules['utils_gnn_VI'])

'''Code Layout:
    The exact line numbers can be a little off
    1. Two-layer GCN, no BN, with true and estimated graphs: Line 273~496
    2. Two or Three-layer GCN, assess the effect of H and B on model performance with true graphs: Line 499~1136
    3. One-layer GCN: line 1143 till the end
'''

# Generate graph data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GCN_VI_simple(torch.nn.Module):
    def __init__(self, C, F_out=1):
        super().__init__()
        self.conv1 = GCNConv(C, F_out)
        self.freeze_BN = False
        self.batch_norm = False
        self.conv1_feature = GCNConv(C, C)
        old_dict = self.state_dict()
        old_dict['conv1_feature.bias'] = torch.zeros(C)
        old_dict['conv1_feature.lin.weight'] = torch.diag(torch.ones(C))
        self.load_state_dict(old_dict)

    def forward(self, data, feature1=False):
        x, edge_index = data.x.to(device), data.edge_index.to(device)
        if feature1:
            return self.conv1_feature(x, edge_index)
        x = self.conv1(x, edge_index)
        if F_out == 1:
            return torch.sigmoid(x)
        else:
            return F.log_softmax(x, dim=1)


class GCN_SGD(torch.nn.Module):
    def __init__(self, C, F_out=1, H=4, splus=False, beta=1, batch_norm=False):
        super().__init__()
        print(f'{H} hidden nodes')
        self.conv1 = GCNConv(C, H)
        self.conv2 = GCNConv(H, F_out)
        self.splus = splus
        self.beta = beta
        self.batch_norm = batch_norm
        self.freeze_BN = False
        self.BN = None
        if self.batch_norm:
            affine = False
            self.BN = torch.nn.BatchNorm1d(H, affine=affine).to(device)

    def forward(self, data):
        x, edge_index = data.x.to(device), data.edge_index.to(device)
        x = self.conv1(x, edge_index)
        if self.BN != None:
            x = self.BN(x)
        func = torch.nn.Softplus(beta=self.beta) if self.splus else F.relu
        x = func(x)
        # if self.BN != None:
        #     x = self.BN(x)
        x = self.conv2(x, edge_index)
        return torch.sigmoid(x)


class GCN_VI(torch.nn.Module):
    def __init__(self, C, F_out=1, H=4, splus=False, beta=1, batch_norm=False):
        super().__init__()
        print(f'{H} hidden nodes')
        self.conv1 = GCNConv(C, H)
        self.conv2 = GCNConv(H, F_out)
        self.splus = splus
        self.beta = beta
        self.batch_norm = batch_norm
        self.freeze_BN = False
        self.conv1_feature = GCNConv(C, C)
        self.conv2_feature = GCNConv(H, H)
        old_dict = self.state_dict()
        old_dict['conv1_feature.bias'] = torch.zeros(C)
        old_dict['conv1_feature.lin.weight'] = torch.diag(torch.ones(C))
        old_dict['conv2_feature.bias'] = torch.zeros(H)
        old_dict['conv2_feature.lin.weight'] = torch.diag(torch.ones(H))
        self.load_state_dict(old_dict)
        # Normalize after convolution & before activation
        self.BN = None
        if self.batch_norm:
            self.affine = False
            self.BN = torch.nn.BatchNorm1d(H, affine=self.affine).to(device)
            if self.affine:
                # TODO: this did not work. We still have no grad upon calling loss.backward()
                self.BN.weight.retain_grad()
                self.BN.bias.retain_grad()

    def forward(self, data, feature1=False, feature2=False):
        x, edge_index = data.x.to(device), data.edge_index.to(device)
        if feature1:
            return self.conv1_feature(x, edge_index)
        layer1_x = self.conv1(x, edge_index)
        # Normalize after convolution & before activation
        if self.BN != None:
            layer1_x = self.BN(layer1_x)
            H = layer1_x.shape[1]
            self.rmean = self.BN.running_mean.clone()
            self.rvar = self.BN.running_var.clone()
            self.weight = torch.ones(H).to(device)
            self.bias = torch.zeros(H).to(device)
            if self.affine:
                self.weight = self.BN.weight.clone()
                self.bias = self.BN.bias.clone()
        func = torch.nn.Softplus(beta=self.beta) if self.splus else F.relu
        layer1_x = func(layer1_x)
        if feature2:
            return self.conv2_feature(layer1_x, edge_index)
        self.layer1_x = Variable(layer1_x, requires_grad=True)
        # Second layer
        layer2_x = self.conv2(self.layer1_x, edge_index)
        return torch.sigmoid(layer2_x)  # This is CONVEX


class GCN_more_layer_SGD(torch.nn.Module):
    def __init__(self, C, F_out=1, H1=2, H2=2, splus=False, beta=1, batch_norm=False):
        super().__init__()
        print(f'{[H1,H2]} hidden nodes')
        self.conv1 = GCNConv(C, H1)
        self.conv2 = GCNConv(H1, H2)
        self.conv3 = GCNConv(H2, F_out)
        self.splus = splus
        self.beta = beta
        self.batch_norm = batch_norm
        self.freeze_BN = False
        self.BN1 = None
        self.BN2 = None
        if self.batch_norm:
            affine = False
            self.BN1 = torch.nn.BatchNorm1d(H1, affine=affine).to(device)
            self.BN2 = torch.nn.BatchNorm1d(H2, affine=affine).to(device)

    def forward(self, data):
        x, edge_index = data.x.to(device), data.edge_index.to(device)
        x = self.conv1(x, edge_index)
        if self.BN1 != None:
            x = self.BN1(x)
        func = torch.nn.Softplus(beta=self.beta) if self.splus else F.relu
        x = func(x)
        x = self.conv2(x, edge_index)
        if self.BN2 != None:
            x = self.BN2(x)
        x = func(x)
        x = self.conv3(x, edge_index)
        if F_out == 1:
            return torch.sigmoid(x)
        else:
            return F.log_softmax(x, dim=1)


class GCN_more_layer_VI(torch.nn.Module):
    def __init__(self, C, F_out=1, H1=128, H2=128, splus=False, beta=1, batch_norm=False):
        super().__init__()
        print(f'{[H1,H2]} hidden nodes')
        self.conv1 = GCNConv(C, H1)
        self.conv2 = GCNConv(H1, H2)
        self.conv3 = GCNConv(H2, F_out)
        self.splus = splus
        self.beta = beta
        self.batch_norm = batch_norm
        self.freeze_BN = False
        self.conv1_feature = GCNConv(C, C)
        self.conv2_feature = GCNConv(H1, H1)
        self.conv3_feature = GCNConv(H2, H2)
        old_dict = self.state_dict()
        old_dict['conv1_feature.bias'] = torch.zeros(C)
        old_dict['conv1_feature.lin.weight'] = torch.diag(torch.ones(C))
        old_dict['conv2_feature.bias'] = torch.zeros(H1)
        old_dict['conv2_feature.lin.weight'] = torch.diag(torch.ones(H1))
        old_dict['conv3_feature.bias'] = torch.zeros(H2)
        old_dict['conv3_feature.lin.weight'] = torch.diag(torch.ones(H2))
        self.load_state_dict(old_dict)
        self.BN1 = None
        self.BN2 = None
        if self.batch_norm:
            self.affine = False
            self.BN1 = torch.nn.BatchNorm1d(H1, affine=self.affine).to(device)
            self.BN2 = torch.nn.BatchNorm1d(H2, affine=self.affine).to(device)
            if self.affine:
                # TODO: this did not work. We still have no grad upon calling loss.backward()
                self.BN1.weight.retain_grad()
                self.BN1.bias.retain_grad()
                self.BN2.weight.retain_grad()
                self.BN2.bias.retain_grad()

    def forward(self, data, feature1=False, feature2=False, feature3=False):
        # Need to use retain_grad() to get latter grad
        x, edge_index = data.x.to(device), data.edge_index.to(device)
        if feature1:
            return self.conv1_feature(x, edge_index)
        layer1_x = self.conv1(x, edge_index)
        # Normalize after convolution & before activation
        if self.BN1 != None:
            layer1_x = self.BN1(layer1_x)
            H = layer1_x.shape[1]
            self.rmean = self.BN1.running_mean.clone()
            self.rvar = self.BN1.running_var.clone()
            self.weight = torch.ones(H).to(device)
            self.bias = torch.zeros(H).to(device)
            if self.affine:
                self.weight = self.BN1.weight.clone()
                self.bias = self.BN1.bias.clone()
        func = torch.nn.Softplus(beta=self.beta) if self.splus else F.relu
        layer1_x = func(layer1_x)
        if feature2:
            return self.conv2_feature(layer1_x, edge_index)
        self.layer1_x = Variable(layer1_x, requires_grad=True)
        layer2_x = self.conv2(self.layer1_x, edge_index)
        # Normalize after convolution & before activation
        if self.BN2 != None:
            layer1_x = self.BN2(layer1_x)
            H = layer1_x.shape[1]
            self.rmean1 = self.BN2.running_mean.clone()
            self.rvar1 = self.BN2.running_var.clone()
            self.weight1 = torch.ones(H).to(device)
            self.bias1 = torch.zeros(H).to(device)
            if self.affine:
                self.weight1 = self.BN2.weight.clone()
                self.bias1 = self.BN2.bias.clone()
        layer2_x = func(layer2_x)
        self.layer2_x = layer2_x
        self.layer2_x.retain_grad()
        if feature3:
            return self.conv3_feature(layer2_x, edge_index)
        x = self.conv3(self.layer2_x, edge_index)
        if F_out == 1:
            return torch.sigmoid(x)
        else:
            return F.log_softmax(x, dim=1)


graph_type = 'large'  # 'small' or 'large'
for pp in [1]:
    ipb.reload(sys.modules['utils_gnn_VI'])
    n = 40 if graph_type == 'large' else 15
    C = 2
    H_true = 2
    F_out = 1  # Multiple layer
    mu = 1
    # # For task type==2 or 3 & est. neuron = 4
    sigma = 1
    np.random.seed(2)
    W1 = np.random.normal(mu, sigma, H_true*C).reshape((H_true, C)
                                                       ).astype(np.float32)  # H_true-by-C
    b1 = np.random.normal(mu, sigma, H_true).astype(np.float32)
    # F-by-H_true, CRUCIAL to reset shape
    # NOTE: these parameters are used so we can have more balanced one and zero to make the problem harder
    W2 = np.random.normal(mu, sigma, F_out*H_true).reshape((F_out, H_true)).astype(np.float32)
    b2 = np.random.normal(mu, sigma, F_out).astype(np.float32)  # F-by-1
    G = nx.fast_gnp_random_graph(n=n, p=0.15, seed=1103)
    edge_index = torch.tensor(list(G.edges)).T.type(torch.long)
    pertub = 0.2 if graph_type == 'small' else 0.05
    G_est = utils.G_reformat(G, percent_perturb=pertub, return_G=True)
    edge_index_est = torch.tensor(list(G_est.edges)).T.type(torch.long)
    N = 2000  # Num training data
    N1 = 2000  # Num test data
    batch_size = int(N/20)
    utils_layer.draw_graph(edge_index, edge_index_est, graph_type)
    model_get_data = GCN_SGD(C, F_out, H_true).to(device)
    # NOTE: another way to change parameters, which FORCES me to make sure parameters match the shape I want
    old_dict = model_get_data.state_dict()
    old_dict['conv1.bias'] = torch.from_numpy(b1)
    old_dict['conv1.lin.weight'] = torch.from_numpy(W1)
    old_dict['conv2.bias'] = torch.from_numpy(b2)
    old_dict['conv2.lin.weight'] = torch.from_numpy(W2)
    model_get_data.load_state_dict(old_dict)

# Update all layers
seeds = [1103, 1111, 1214]
H_ls = [H_true, 4, 8, 16, 32]
loss_type = 'MSE'  # 'MSE' or 'Cross-Entropy'
num_epochs = 200
Adam = False
opt_type = '_Adam' if Adam else ''
for H in H_ls:
    tasks = [2, 3]  # Task = 1 just run once at H_true
    for task_type in tasks:
        compute_para_err = True if task_type == 1 else False
        plot_para_recovery = True if task_type == 1 else False
        result_SGD1_dict = {}
        result_VI1_dict = {}
        for seed in seeds:
            ipb.reload(sys.modules['utils_gnn_VI'])
            # Generate Data
            model_get_data.eval()
            print(f'True model: {list(model_get_data.parameters())}')
            X_train, Y_train = utils_layer.get_simulation_data(
                model_get_data, N, edge_index, n, C, torch_seed=seed)
            len(X_train)
            X_test, Y_test = utils_layer.get_simulation_data(
                model_get_data, N1, edge_index, n, C, train=False, torch_seed=seed)
            len(X_test)
            train_loader, test_loader = utils_layer.get_train_test_loader(
                X_train, X_test, Y_train, Y_test, edge_index, batch_size)
            test_loader_true = None
            if task_type == 3:
                train_loader, test_loader = utils_layer.get_train_test_loader(
                    X_train, X_test, Y_train, Y_test, edge_index_est, batch_size)
                _, test_loader_true = utils_layer.get_train_test_loader(
                    X_train, X_test, Y_train, Y_test, edge_index, batch_size)
            # Estimation
            torch.manual_seed(seed)  # For reproducibility
            # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model_SGD1 = GCN_SGD(C, H=H).to(device)
            # print(f'SGD: {list(model_SGD1.parameters())}')
            # SGD
            mod_SGD1 = utils_layer.GCN_train(model_SGD1, train_loader,
                                             test_loader, model_get_data, test_loader_true)
            para_error_vanilla1, pred_l2error_vanilla1, pred_linferror_vanilla1, pred_loss_vanilla1 = mod_SGD1.training_and_eval(
                num_epochs, compute_para_err=compute_para_err, loss_type=loss_type, Adam=Adam)
            result_SGD1_dict[f'Seed {seed}'] = [para_error_vanilla1,
                                                pred_l2error_vanilla1, pred_linferror_vanilla1, pred_loss_vanilla1]
            # NOW VI
            torch.manual_seed(seed)  # For reproducibility
            model_VI1 = GCN_VI(C, H=H).to(device)
            model_to_feature_ls = [0]  # Place holder, never used
            mod_VI1 = utils_layer.GCN_train(model_VI1, train_loader,
                                            test_loader, model_get_data, test_loader_true)
            para_error_VI1, pred_l2error_VI1, pred_linferror_VI1, pred_loss_VI1 = mod_VI1.training_and_eval(
                num_epochs, compute_para_err=compute_para_err, model_to_feature_ls=model_to_feature_ls, loss_type=loss_type, Adam=Adam)
            result_VI1_dict[f'Seed {seed}'] = [para_error_VI1,
                                               pred_l2error_VI1, pred_linferror_VI1, pred_loss_VI1]
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(pred_linferror_vanilla1, label='SGD')
            ax.plot(pred_linferror_VI1, label='VI')
            ax.set_title('SGD vs. VI on $l_{\infty}$ prediction error')
            ax.legend(loc='best')
            # ax.set_yscale('log')
            # ax.set_xscale('log')
            plt.show()
        json_SGD = json.dumps(str(result_SGD1_dict))
        json_VI = json.dumps(str(result_VI1_dict))
        if task_type == 2:
            name = f'SGD_Simulation_b_{graph_type}_first_layer_not_known_H={H}{loss_type}{opt_type}'
            name1 = f'VI_Simulation_b_{graph_type}_first_layer_not_known_H={H}{loss_type}{opt_type}'
        if task_type == 3:
            name = f'SGD_Simulation_c_{graph_type}_est_graph_H={H}{loss_type}{opt_type}'
            name1 = f'VI_Simulation_c_{graph_type}_est_graph_H={H}{loss_type}{opt_type}'
        # open file for writing, "w"
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
        para_error_vanilla, para_error_vanillaSE, pred_l2error_vanilla, pred_l2error_vanillaSE, pred_linferror_vanilla, pred_linferror_vanillaSE, pred_loss_vanilla, pred_loss_vanillaSE = utils_layer.get_all(
            result_SGD1_dict)
        para_error_VI, para_error_VISE, pred_l2error_VI, pred_l2error_VISE, pred_linferror_VI, pred_linferror_VISE, pred_loss_VI, pred_loss_VISE = utils_layer.get_all(
            result_VI1_dict)
        fig = utils_layer.simulation_plot(para_error_vanilla, pred_l2error_vanilla, pred_loss_vanilla, pred_linferror_vanilla, para_error_VI, pred_l2error_VI, pred_loss_VI, pred_linferror_VI,
                                          para_error_vanillaSE, pred_l2error_vanillaSE, pred_loss_vanillaSE, pred_linferror_vanillaSE, para_error_VISE, pred_l2error_VISE, pred_loss_VISE, pred_linferror_VISE, plot_para_recovery=plot_para_recovery, loss_type=loss_type, Adam=Adam)
        if task_type == 1:
            fig.savefig(f'Simulation_a_{graph_type}_first_layer_fully_known{loss_type}{opt_type}.pdf',
                        dpi=300, bbox_inches='tight', pad_inches=0)
        elif task_type == 2:
            fig.savefig(f'Simulation_b_{graph_type}_first_layer_not_known_H={H}{loss_type}{opt_type}.pdf',
                        dpi=300, bbox_inches='tight', pad_inches=0)
        else:
            fig.savefig(f'Simulation_c_{graph_type}_est_graph_H={H}{loss_type}{opt_type}.pdf',
                        dpi=300, bbox_inches='tight', pad_inches=0)


# Use saved JSON data:
# Plot prediction with and without knowing graph together, with label
# For each f, load the case in task_type b and c
# Also, save results in this Table below
# Table:
# row are hidden nueron number
# columns are three metrics, each having 6 entries: SGD vs. VI-SGD under true graph & under est. graph accuracy, and the relative error in between. In particular, present the three numbers for one method
# So 18 entries per row.
H_true = 2
H_ls = [H_true, 4, 8, 16, 32]
# H_ls = [32]
loss_type = '_MSE'
Adam = False
opt_type = '_Adam' if Adam else ''
Table_dict = {}
SGD_label = 'Adam (Known)' if Adam else 'SGD (Known)'
SGD_label1 = 'Adam (Est)' if Adam else 'SGD (Est)'
opt_type_sub = 'Adam' if Adam else 'SGD'
make_fig = True  # If just get table, do not set to True
gtypes = ['small', 'large']
for graph_type in gtypes:
    Table = np.zeros((len(H_ls), 2*12))
    columns = np.tile([SGD_label+' mean', SGD_label+' SE', SGD_label1+' mean', SGD_label1+' SE',
                       f'VI-{SGD_label}'+' mean', f'VI-{SGD_label}'+' SE', f'VI-{SGD_label1}'+' mean', f'VI-{SGD_label1}'+' SE'], 3)
    type = np.repeat(['Posterior prediction--$l_2$ norm',
                      f'{loss_type} loss', 'Posterior prediction--$l_{\infty}$ norm'], 8)
    tuples = list(zip(*[type, columns]))
    index = pd.MultiIndex.from_tuples(tuples)
    ipb.reload(sys.modules['utils_gnn_VI'])
    for k, H in enumerate(H_ls):
        print(f'{graph_type} graph, H={H}')
        # Case b, knowing graph completely
        name = f'SGD_Simulation_b_{graph_type}_first_layer_not_known_H={H}{loss_type}{opt_type}.json'
        name1 = f'VI_Simulation_b_{graph_type}_first_layer_not_known_H={H}{loss_type}{opt_type}.json'
        with open(name, 'r') as j:
            result_SGD1_dict = json.loads(j.read())
            result_SGD1_dict = ast.literal_eval(result_SGD1_dict)
        with open(name1, 'r') as j:
            result_VI1_dict = json.loads(j.read())
            result_VI1_dict = ast.literal_eval(result_VI1_dict)
        SGD_know_graph = utils_layer.get_all(
            result_SGD1_dict)
        VI_know_graph = utils_layer.get_all(
            result_VI1_dict)
        # Case c, estimate graph
        name = f'SGD_Simulation_c_{graph_type}_est_graph_H={H}{loss_type}{opt_type}.json'
        name1 = f'VI_Simulation_c_{graph_type}_est_graph_H={H}{loss_type}{opt_type}.json'
        with open(name, 'r') as j:
            result_SGD1_dict = json.loads(j.read())
            result_SGD1_dict = ast.literal_eval(result_SGD1_dict)
        with open(name1, 'r') as j:
            result_VI1_dict = json.loads(j.read())
            result_VI1_dict = ast.literal_eval(result_VI1_dict)
        SGD_est_graph = utils_layer.get_all(
            result_SGD1_dict)
        VI_est_graph = utils_layer.get_all(
            result_VI1_dict)
        single_plot = False
        if H == 32:
            single_plot = True
            if make_fig:
                single_plot = False
        fig, long_ls = utils_layer.simulation_plot_know_est_graph(
            SGD_know_graph, VI_know_graph, SGD_est_graph, VI_est_graph, loss_type=loss_type, Adam=Adam, make_fig=make_fig, single_plot=single_plot)
        Table[k] = long_ls
        if fig != 0:
            sing_idx = '_single' if single_plot else ''
            fig.savefig(f'{graph_type}_graph_know_est_graph_H={H}{loss_type}{opt_type}{sing_idx}.pdf',
                        dpi=300, bbox_inches='tight', pad_inches=0)
    Table = pd.DataFrame(Table, index=H_ls, columns=index)
    Table.index.name = '# Hidden nodes'
    Table_dict[graph_type] = Table
list(Table_dict.keys())
columns = np.tile([SGD_label, SGD_label1,
                   f'VI-{SGD_label}', f'VI-{SGD_label1}'], 3)
type = np.repeat(['Posterior prediction--$l_2$ norm',
                  f'{loss_type} loss', 'Posterior prediction--$l_{\infty}$ norm'], 4)
tuples = list(zip(*[type, columns]))
new_colindex = pd.MultiIndex.from_tuples(tuples)
round_more = False
Table_new_small = utils_layer.concatenat_to_one(Table_dict['small'], new_colindex, round_more)
Table_new_large = utils_layer.concatenat_to_one(Table_dict['large'], new_colindex, round_more)


# Test results. NOTE, MSE loss and linf error first, because l2 error is similar and the table is too wide
# if all included
print(Table_new_small.to_latex(escape=False))
print(Table_new_large.to_latex(escape=False))

# Visualize dynamics of weight update
# Setup: C=2 (so easy to see), H large (like several hundred), so each column in the matrix of shape (C-by-H)
# denotes the weight on node i
# For now, still include bias, but we just visualize weights
# 1. Start with the same initialization of parameter and store them
# 2. After estimation, retrieve the parameters and obtain the inner plot with the initial one

# Then plot both on the plot

graph_type = 'small'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
more_layers = True  # If True, data has 2 hidden layers
for pp in [1]:
    n = 40 if graph_type == 'large' else 15
    C = 2
    H_true = 2
    F_out = 1  # Multiple layer
    mu = 1
    # # For task type==2 or 3 & est. neuron = 4
    sigma = 1
    np.random.seed(2)
    W1 = np.random.normal(mu, sigma, H_true*C).reshape((H_true, C)
                                                       ).astype(np.float32)  # H_true-by-C
    b1 = np.random.normal(mu, sigma, H_true).astype(np.float32)
    # F-by-H_true, CRUCIAL to reset shape
    # NOTE: these parameters are used so we can have more balanced one and zero to make the problem harder
    W2 = np.random.normal(mu, sigma, F_out*H_true).reshape((F_out, H_true)).astype(np.float32)
    b2 = np.random.normal(mu, sigma, F_out).astype(np.float32)  # F-by-1
    model_get_data = GCN_SGD(C, F_out, H_true).to(device)

    if more_layers:
        W1_more = np.random.normal(mu, sigma, H_true*H_true).reshape((H_true, H_true)
                                                                     ).astype(np.float32)  # H_true-by-H_true
        b1_more = np.random.normal(mu, sigma, H_true).astype(np.float32)
        model_get_data = GCN_more_layer_SGD(C, F_out, H_true, H_true).to(device)
    # NOTE: another way to change parameters, which FORCES me to make sure parameters match the shape I want
    old_dict = model_get_data.state_dict()
    old_dict['conv1.bias'] = torch.from_numpy(b1)
    old_dict['conv1.lin.weight'] = torch.from_numpy(W1)
    if more_layers:
        old_dict['conv2.bias'] = torch.from_numpy(b1_more)
        old_dict['conv2.lin.weight'] = torch.from_numpy(W1_more)
        old_dict['conv3.bias'] = torch.from_numpy(b2)
        old_dict['conv3.lin.weight'] = torch.from_numpy(W2)
    else:
        old_dict['conv2.bias'] = torch.from_numpy(b2)
        old_dict['conv2.lin.weight'] = torch.from_numpy(W2)
    model_get_data.load_state_dict(old_dict)
    # Generate the ground truth graph
    G = nx.fast_gnp_random_graph(n=n, p=0.15, seed=1103)
    edge_index = torch.tensor(list(G.edges)).T.type(torch.long)
    pertub = 0.2 if graph_type == 'small' else 0.05
    G_est = utils_layer.G_reformat(G, percent_perturb=pertub, return_G=True)
    edge_index_est = torch.tensor(list(G_est.edges)).T.type(torch.long)
    N = 2000  # Num training data
    N1 = 2000  # Num test data.

batch_vs_H = False  # If True, then comprehensive results with Adam=False, splus=False, loss=MSE. If False, then illustrative examples with Adam, splus, cross-entropy loss, and fixed H, B
for _ in [1]:
    N = 2000
    Adam_ls = [True, False] if batch_vs_H else [True, False]
    splus_ls = [False] if batch_vs_H else [True]
    loss_type_ls = ['MSE'] if batch_vs_H else ['Cross-Entropy']
    bnorm_ls = [[True, False], [True, True], [False, False]]
    H_ls = [2, 4, 8, 16, 32] if batch_vs_H else [16]
    batch_size_ls = [int(N/40), int(N/20), int(N/10)] if batch_vs_H else [int(N/20)]
    more_layers_ls = [False, True]
    N = 2000
    num_epochs = 200
    num_divide = 40
    C = 2
    freq = int(num_epochs//num_divide)
save_result = True
savefig = True
seeds = [1103, 1111, 1214]
for Adam in Adam_ls:
    opt_type = '_Adam' if Adam else ''
    for splus in splus_ls:
        beta = 5  # This is the best value for SGD
        for loss_type in loss_type_ls:
            for H in H_ls:
                for batch_norm, freeze_BN in bnorm_ls:
                    for batch_size in batch_size_ls:
                        names = ['SGD_train', 'SGD_test', 'VI_train', 'VI_test']
                        result_dict = {name: {} for name in names}
                        for seed in seeds:
                            # Train model
                            ipb.reload(sys.modules['utils_gnn_VI'])
                            compute_para_err = False
                            plot_para_recovery = False
                            # Generate Data
                            model_get_data.eval()
                            print(f'True model: {list(model_get_data.parameters())}')
                            X_train, Y_train = utils_layer.get_simulation_data(
                                model_get_data, N, edge_index, n, C, torch_seed=seed)
                            len(X_train)
                            X_test, Y_test = utils_layer.get_simulation_data(
                                model_get_data, N1, edge_index, n, C, train=False, torch_seed=seed)
                            len(X_test)
                            train_loader, test_loader = utils_layer.get_train_test_loader(
                                X_train, X_test, Y_train, Y_test, edge_index, batch_size)
                            # Estimation
                            # SGD first
                            torch.manual_seed(seed)  # For reproducibility
                            model_SGD1 = GCN_SGD(C, H=H, splus=splus, beta=beta,
                                                 batch_norm=batch_norm).to(device)
                            if more_layers:
                                torch.manual_seed(seed)  # For reproducibility
                                model_SGD1 = GCN_more_layer_SGD(C, H1=H, H2=H, splus=splus, beta=beta,
                                                                batch_norm=batch_norm).to(device)
                            SGD_dict_ref = copy.deepcopy(model_SGD1.state_dict())
                            para_error_vanilla_train = []
                            pred_l2error_vanilla_train = []
                            pred_linferror_vanilla_train = []
                            pred_loss_vanilla_train = []
                            para_error_vanilla = []
                            pred_l2error_vanilla = []
                            pred_linferror_vanilla = []
                            pred_loss_vanilla = []
                            # NOW VI
                            torch.manual_seed(seed)  # For reproducibility
                            model_VI1 = GCN_VI(C, H=H, splus=splus, beta=beta,
                                               batch_norm=batch_norm).to(device)
                            if more_layers:
                                torch.manual_seed(seed)  # For reproducibility
                                model_VI1 = GCN_more_layer_VI(C, H1=H, H2=H, splus=splus, beta=beta,
                                                              batch_norm=batch_norm).to(device)
                            VI_dict_ref = copy.deepcopy(model_VI1.state_dict())
                            model_to_feature_ls = [0]  # Place holder, never used
                            para_error_VI_train = []
                            pred_l2error_VI_train = []
                            pred_linferror_VI_train = []
                            pred_loss_VI_train = []
                            para_error_VI = []
                            pred_l2error_VI = []
                            pred_linferror_VI = []
                            pred_loss_VI = []
                            num_divide = 40
                            freq = int(num_epochs//num_divide)
                            for epoch in range(num_epochs):
                                if freeze_BN and batch_norm:
                                    if epoch >= int(num_epochs/2):
                                        model_SGD1.freeze_BN = True
                                        model_VI1.freeze_BN = True
                                print(f'SGD epoch {epoch}')
                                train_loss = utils_layer.train_revised_all_layer(train_loader,
                                                                                 model_to_train=model_SGD1, output_dim=1, more_layers=more_layers, loss_type=loss_type, Adam=Adam)
                                if np.mod(epoch+1, freq) == 0:
                                    para_err, l2_err, linf_err, loss_true = utils_layer.evaluation_simulation(
                                        train_loader, model_get_data, model_SGD1, W2=[], b2=[], data_loader_true=train_loader, loss_type=loss_type)
                                    para_error_vanilla_train.append(para_err)
                                    pred_l2error_vanilla_train.append(l2_err)
                                    pred_linferror_vanilla_train.append(linf_err)
                                    pred_loss_vanilla_train.append(loss_true)
                                    print(
                                        f'[Train: rel Para err, rel l2 err, rel linf err, loss] at {epoch} is \n {[para_err, l2_err, linf_err, loss_true]}')
                                    para_err, l2_err, linf_err, loss_true = utils_layer.evaluation_simulation(
                                        test_loader, model_get_data, model_SGD1, W2=[], b2=[], data_loader_true=test_loader, loss_type=loss_type)
                                    para_error_vanilla.append(para_err)
                                    pred_l2error_vanilla.append(l2_err)
                                    pred_linferror_vanilla.append(linf_err)
                                    pred_loss_vanilla.append(loss_true)
                                    print(
                                        f'[Test: rel Para err, rel l2 err, rel linf err, loss] at {epoch} is \n {[para_err, l2_err, linf_err, loss_true]}')
                                    if model_SGD1.batch_norm:
                                        if more_layers:
                                            print(
                                                f'SGD layer 1 rvar: {model_SGD1.BN1.running_var[-1]}')
                                            print(
                                                f'SGD layer 2 rvar: {model_SGD1.BN2.running_var[-1]}')
                                        else:
                                            print(
                                                f'SGD rvar: {model_SGD1.BN.running_var[-1]}')
                                print(f'VI epoch {epoch}')
                                train_loss = utils_layer.train_revised_all_layer(train_loader,
                                                                                 model_to_train=model_VI1, output_dim=1, more_layers=more_layers, model_to_feature_ls=model_to_feature_ls, loss_type=loss_type, Adam=Adam)

                                if np.mod(epoch+1, freq) == 0:
                                    para_err, l2_err, linf_err, loss_true = utils_layer.evaluation_simulation(
                                        train_loader, model_get_data, model_VI1, W2=[], b2=[], data_loader_true=train_loader, loss_type=loss_type)
                                    para_error_VI_train.append(para_err)
                                    pred_l2error_VI_train.append(l2_err)
                                    pred_linferror_VI_train.append(linf_err)
                                    pred_loss_VI_train.append(loss_true)
                                    print(
                                        f'[Train: rel Para err, rel l2 err, rel linf err, loss] at {epoch} is \n {[para_err, l2_err, linf_err, loss_true]}')
                                    para_err, l2_err, linf_err, loss_true = utils_layer.evaluation_simulation(
                                        test_loader, model_get_data, model_VI1, W2=[], b2=[], data_loader_true=test_loader, loss_type=loss_type)
                                    para_error_VI.append(para_err)
                                    pred_l2error_VI.append(l2_err)
                                    pred_linferror_VI.append(linf_err)
                                    pred_loss_VI.append(loss_true)
                                    print(
                                        f'[Test: rel Para err, rel l2 err, rel linf err, loss] at {epoch} is \n {[para_err, l2_err, linf_err, loss_true]}')
                                    if model_SGD1.batch_norm:
                                        if more_layers:
                                            print(
                                                f'VI layer 1 (rvar,weight): {(model_VI1.rvar[-1],model_VI1.weight[-1])}')
                                            print(
                                                f'VI layer 2 (rvar,weight): {(model_VI1.rvar1[-1],model_VI1.weight1[-1])}')
                                        else:
                                            print(
                                                f'VI (rvar,weight): {(model_VI1.rvar[-1],model_VI1.weight[-1])}')
                            result_dict['SGD_train'][seed] = [para_error_vanilla_train,
                                                              pred_l2error_vanilla_train,
                                                              pred_linferror_vanilla_train, pred_loss_vanilla_train]
                            result_dict['SGD_test'][seed] = [para_error_vanilla,
                                                             pred_l2error_vanilla,
                                                             pred_linferror_vanilla, pred_loss_vanilla]
                            SGD_dict_final = copy.deepcopy(model_SGD1.state_dict())
                            result_dict['VI_train'][seed] = [para_error_VI_train, pred_l2error_VI_train,
                                                             pred_linferror_VI_train, pred_loss_VI_train]
                            result_dict['VI_test'][seed] = [para_error_VI,
                                                            pred_l2error_VI, pred_linferror_VI, pred_loss_VI]
                            VI_dict_final = copy.deepcopy(model_VI1.state_dict())
                        if save_result:
                            # Just save the parametplot_quickers for teh last seed is enough
                            splus_suff = '_splus' if splus else ''
                            bnorm = '_bnorm' if batch_norm else ''
                            bsize = f'_bsize={batch_size}'
                            more_layer = '_more_layers' if more_layers else ''
                            half = '_half' if freeze_BN else ''
                            name1 = f'SGD_VI_dynamics_H={H}_{loss_type}{opt_type}_ref{splus_suff}{bnorm}{half}{more_layer}{bsize}.pth'
                            name2 = f'SGD_VI_dynamics_H={H}_{loss_type}{opt_type}_SGD{splus_suff}{bnorm}{half}{more_layer}{bsize}.pth'
                            name3 = f'SGD_VI_dynamics_H={H}_{loss_type}{opt_type}_VI{splus_suff}{bnorm}{half}{more_layer}{bsize}.pth'
                            torch.save(SGD_dict_ref, name1)
                            torch.save(SGD_dict_final, name2)
                            torch.save(VI_dict_final, name3)
                            # Save after all seeds
                            json_linf = json.dumps(str(result_dict))
                            name = f'SGD_VI_dynamics_H={H}_{loss_type}{opt_type}{splus_suff}{bnorm}{half}{more_layer}{bsize}'
                            f = open(f"{name}.json", "w")
                            # write json object to file
                            f.write(json_linf)
                            # close file
                            f.close()
                        # Get parameters and plot. Still just visualize last layer dynamics
                        key = 'conv2.lin.weight' if more_layers else 'conv1.lin.weight'
                        w_ref = SGD_dict_ref[key].cpu().detach().numpy()
                        w_SGD = SGD_dict_final[key].cpu().detach().numpy()
                        w_VI = VI_dict_final[key].cpu().detach().numpy()
                        w_ref_inner = np.sum(w_ref*w_ref, axis=1)
                        w_SGD_final = np.sum(w_ref*w_SGD, axis=1)
                        w_VI_final = np.sum(w_ref*w_VI, axis=1)
                        key2 = 'conv3.lin.weight' if more_layers else 'conv2.lin.weight'
                        a_SGD_ref = SGD_dict_ref[key2].cpu().detach().numpy().flatten()
                        a_VI_ref = VI_dict_ref[key2].cpu().detach().numpy().flatten()
                        a_SGD_final = SGD_dict_final[key2].cpu().detach().numpy().flatten()
                        a_VI_final = VI_dict_final[key2].cpu().detach().numpy().flatten()
                        a_SGD = [a_SGD_ref, a_SGD_final]
                        a_VI = [a_VI_ref, a_VI_final]
                        w_SGD = [w_ref_inner, w_SGD_final]
                        w_VI = [w_ref_inner, w_VI_final]
                        # Get average results with std for plot
                        para_error_vanilla_train, para_error_vanilla_trainSE, pred_l2error_vanilla_train, pred_l2error_vanilla_trainSE, pred_linferror_vanilla_train, pred_linferror_vanilla_trainSE, pred_loss_vanilla_train, pred_loss_vanilla_trainSE = utils_layer.get_all(
                            result_dict['SGD_train'])
                        para_error_vanilla, para_error_vanillaSE, pred_l2error_vanilla, pred_l2error_vanillaSE, pred_linferror_vanilla, pred_linferror_vanillaSE, pred_loss_vanilla, pred_loss_vanillaSE = utils_layer.get_all(
                            result_dict['SGD_test'])
                        para_error_VI_train, para_error_VI_trainSE, pred_l2error_VI_train, pred_l2error_VI_trainSE, pred_linferror_VI_train, pred_linferror_VI_trainSE, pred_loss_VI_train, pred_loss_VI_trainSE = utils_layer.get_all(
                            result_dict['VI_train'])
                        para_error_VI, para_error_VISE, pred_l2error_VI, pred_l2error_VISE, pred_linferror_VI, pred_linferror_VISE, pred_loss_VI, pred_loss_VISE = utils_layer.get_all(
                            result_dict['VI_test'])
                        linferror_ls = [pred_linferror_vanilla_train, pred_linferror_vanilla_trainSE, pred_linferror_vanilla,
                                        pred_linferror_vanillaSE, pred_linferror_VI_train, pred_linferror_VI_trainSE, pred_linferror_VI, pred_linferror_VISE]
                        loss_ls = [pred_loss_vanilla_train, pred_loss_vanilla_trainSE, pred_loss_vanilla,
                                   pred_loss_vanillaSE, pred_loss_VI_train, pred_loss_VI_trainSE, pred_loss_VI, pred_loss_VISE]
                        # plt.plot(pred_linferror_vanilla_train)
                        # fig = plot_dynamics(
                        #     a_SGD, w_SGD, a_VI, w_VI, linferror_ls, loss_ls, freq, more_layers, batch_norm, plot_neuron_dynamics=False)
                        # if savefig:
                        #     fig.savefig(f'SGD_VI_dynamics_H={H}_{loss_type}{opt_type}{splus_suff}{bnorm}{half}{more_layer}{bsize}_ndynamics.pdf',
                        #                 dpi=300, bbox_inches='tight', pad_inches=0)


# Load json filed to remake plots
def plot_dynamics(a_SGD, w_SGD, a_VI, w_VI, linferror_ls, loss_ls, freq, more_layers=False, batch_norm=False, freeze_BN=False, plot_neuron_dynamics=True):
    # Plot parameters distribution (Fig. 2a)
    # Parameters
    plt.rcParams['xtick.labelsize'] = 18
    plt.rcParams['ytick.labelsize'] = 18
    pred_linferror_vanilla_train, pred_linferror_vanilla_trainSE, pred_linferror_vanilla, pred_linferror_vanillaSE, pred_linferror_VI_train, pred_linferror_VI_trainSE, pred_linferror_VI, pred_linferror_VISE = linferror_ls
    pred_loss_vanilla_train, pred_loss_vanilla_trainSE, pred_loss_vanilla, pred_loss_vanillaSE, pred_loss_VI_train, pred_loss_VI_trainSE, pred_loss_VI, pred_loss_VISE = loss_ls
    if plot_neuron_dynamics:
        fig = plt.figure(tight_layout=True, figsize=(14, 5))
        gs = gridspec.GridSpec(2, 5)
        cutoff = 2
        ax = fig.add_subplot(gs[:, :cutoff])
        ax.set_xlabel("Epoch", fontsize=16)
        ax.set_ylabel(r"$l_{\infty}$ Error", fontsize=20)
        SGD_label = 'Adam' if Adam else 'SGD'
        xaxis = range(len(pred_linferror_vanilla_train))
        ax.plot(pred_linferror_vanilla_train, linestyle='dashed',
                label=f'{SGD_label} training', color='black')
        ax.fill_between(xaxis, pred_linferror_vanilla_train-pred_linferror_vanilla_trainSE, pred_linferror_vanilla_train+pred_linferror_vanilla_trainSE,
                        color='black', alpha=0.1)
        ax.plot(pred_linferror_vanilla, label=f'{SGD_label} test', color='black')
        ax.fill_between(xaxis, pred_linferror_vanilla-pred_linferror_vanillaSE, pred_linferror_vanilla+pred_linferror_vanillaSE,
                        color='black', alpha=0.1)
        ax.plot(pred_linferror_VI_train, linestyle='dashed', label=f'SVI training', color='orange')
        ax.fill_between(xaxis, pred_linferror_VI_train-pred_linferror_VI_trainSE, pred_linferror_VI_train+pred_linferror_VI_trainSE,
                        color='orange', alpha=0.1)
        ax.plot(pred_linferror_VI, label=f'SVI test', color='orange')
        ax.fill_between(xaxis, pred_linferror_VI-pred_linferror_VISE, pred_linferror_VI+pred_linferror_VISE,
                        color='orange', alpha=0.1)
        ax.legend(loc='upper right', fontsize=12.5, ncol=2)
        # ax.tick_params(labelsize=14)
        labels = [item*freq for item in ax.get_xticks()]
        labels = np.array(labels, dtype=int)
        ax.set_xticklabels(labels)
        ax.grid(which='both')
        for i in range(2):
            prefix = SGD_label if i == 0 else f'VI-{SGD_label}'
            if i == 0:
                a = a_SGD
                w = w_SGD
            else:
                a = a_VI
                w = w_VI
            ax2 = fig.add_subplot(gs[i, cutoff:])
            if i == 0:
                temp_fig = ax2
            # ax2.set_ylim(YLIM_l, YLIM_u)
            # ax2.set_xlim(XLIM_l, XLIM_u)
            ax2.set_xlabel(r"$a_i$", fontsize=16)
            ax2.set_ylabel(r"$w^\parallel_i$", fontsize=16)
            # ax2.tick_params(labelsize=14)
            slist = range(len(a[0]))
            for j in slist:
                ax2.plot([a[0][j], a[1][j]], [w[0][j], w[1][j]],
                         '-', color='black', linewidth=0.5)
            ax2.plot(a[0], w[0], 'o', color='grey', mfc='none', mew=1.5, label="Epoch = 0")
            ax2.plot(a[-1], w[-1], 'o', color='purple', mfc='none', mew=1.5,
                     label=f"Epoch = {num_epochs}")
            mult = 10
            ax2.quiver(a[0], w[0], mult*w[0], mult*a[0], color='black',
                       scale=60.0, width=0.003, headwidth=3)
            # ax2.set_title(f'{prefix} Dynamics')
            if i == 0:
                ax2.legend(ncol=2, fontsize=14, loc='upper center', bbox_to_anchor=(0.5, 1.4))
            if i == 1:
                ax2.get_shared_x_axes().join(temp_fig, ax2)
                ax2.get_shared_y_axes().join(temp_fig, ax2)
        plt.tight_layout()
        fig1 = 0
    else:
        # Loss
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.set_xlabel("Epoch", fontsize=20)
        ax.set_ylabel('Loss', fontsize=20)
        if Adam == False:
            if loss_type == 'MSE':
                if more_layers:
                    if batch_norm:
                        end = 0.15  # Same end because initially all use BN
                        if freeze_BN:
                            start = 0.13
                        else:
                            start = 0.13
                    else:
                        start = 0.135
                        end = 0.195
                else:
                    if batch_norm:
                        end = 0.25  # Same end because initially all use BN
                        if freeze_BN:
                            start = 0.2465
                        else:
                            start = 0.2465
                    else:
                        start = 0.247
                        end = 0.251
            else:
                if more_layers:
                    start = 0.42
                    end = 0.53
                else:
                    start = 0.685
                    end = 0.7
        else:
            if more_layers:
                start = 0.419
                end = 0.44
            else:
                start = 0.68
                end = 0.71
        ax.set_ylim([start, end])
        ax.yaxis.set_ticks(np.arange(start, end, (end-start)/10))
        SGD_label = 'Adam' if Adam else 'SGD'
        xaxis = range(len(pred_loss_vanilla_train))
        ax.plot(pred_loss_vanilla_train, linestyle='dashed',
                label=f'{SGD_label} training', color='black')
        ax.fill_between(xaxis, pred_loss_vanilla_train-pred_loss_vanilla_trainSE, pred_loss_vanilla_train+pred_loss_vanilla_trainSE,
                        color='black', alpha=0.1)
        ax.plot(pred_loss_vanilla, label=f'{SGD_label} test', color='black')
        ax.fill_between(xaxis, pred_loss_vanilla-pred_loss_vanillaSE, pred_loss_vanilla+pred_loss_vanillaSE,
                        color='black', alpha=0.1)
        ax.plot(pred_loss_VI_train, linestyle='dashed', label=f'SVI training', color='orange')
        ax.fill_between(xaxis, pred_loss_VI_train-pred_loss_VI_trainSE, pred_loss_VI_train+pred_loss_VI_trainSE,
                        color='orange', alpha=0.1)
        ax.plot(pred_loss_VI, label=f'SVI test', color='orange')
        ax.fill_between(xaxis, pred_loss_VI-pred_loss_VISE, pred_loss_VI+pred_loss_VISE,
                        color='orange', alpha=0.1)
        ax.tick_params(labelsize=18)
        labels = [item*freq for item in ax.get_xticks()]
        labels = np.array(labels, dtype=int)
        ax.set_xticklabels(labels)
        ax.grid(which='both')
        # ax[0].set_yscale('log')
        # Linferror
        fig1, ax = plt.subplots(figsize=(7, 4))
        ax.set_xlabel("Epoch", fontsize=20)
        ax.set_ylabel(r"$l_{\infty}$ Error", fontsize=20)
        if Adam == False:
            if loss_type == 'MSE':
                if more_layers:
                    if batch_norm:
                        end = 0.18  # Same end because initially all use BN
                        if freeze_BN:
                            start = 0.04
                        else:
                            start = 0.06
                    else:
                        start = 0.02
                        end = 0.42
                else:
                    if batch_norm:
                        end = 0.1  # Same end because initially all use BN
                        if freeze_BN:
                            start = 0.024
                        else:
                            start = 0.024
                    else:
                        start = 0.052
                        end = 0.13
            else:
                if more_layers:
                    start = 0.05
                    end = 0.3
                else:
                    start = 0.025
                    end = 0.15
        else:
            if more_layers:
                start = 0.03
                end = 0.26
            else:
                start = 0.02
                end = 0.15
        ax.set_ylim([start, end])
        ax.yaxis.set_ticks(np.arange(start, end, (end-start)/10))
        SGD_label = 'Adam' if Adam else 'SGD'
        xaxis = range(len(pred_linferror_vanilla_train))
        ax.plot(pred_linferror_vanilla_train, linestyle='dashed',
                label=f'{SGD_label} training', color='black')
        ax.fill_between(xaxis, pred_linferror_vanilla_train-pred_linferror_vanilla_trainSE, pred_linferror_vanilla_train+pred_linferror_vanilla_trainSE,
                        color='black', alpha=0.1)
        ax.plot(pred_linferror_vanilla, label=f'{SGD_label} test', color='black')
        ax.fill_between(xaxis, pred_linferror_vanilla-pred_linferror_vanillaSE, pred_linferror_vanilla+pred_linferror_vanillaSE,
                        color='black', alpha=0.1)
        ax.plot(pred_linferror_VI_train, linestyle='dashed',
                label=f'SVI training', color='orange')
        ax.fill_between(xaxis, pred_linferror_VI_train-pred_linferror_VI_trainSE, pred_linferror_VI_train+pred_linferror_VI_trainSE,
                        color='orange', alpha=0.1)
        ax.plot(pred_linferror_VI, label=f'SVI test', color='orange')
        ax.fill_between(xaxis, pred_linferror_VI-pred_linferror_VISE, pred_linferror_VI+pred_linferror_VISE,
                        color='orange', alpha=0.1)
        ax.tick_params(labelsize=18)
        labels = [item*freq for item in ax.get_xticks()]
        labels = np.array(labels, dtype=int)
        ax.set_xticklabels(labels)
        ax.grid(which='both')
        # fig.tight_layout()
        # ax.legend(ncol=4, loc='lower right', bbox_to_anchor=(1, -0.48))
        # ax.set_yscale('log')
    return [fig, fig1]


batch_vs_H = False  # If True, then comprehensive results with Adam=False, splus=False, loss=MSE. If False, then illustrative examples with Adam, splus, cross-entropy loss, and fixed H, B
for _ in [1]:
    N = 2000
    Adam_ls = [True, False] if batch_vs_H else [True, False]
    splus_ls = [False] if batch_vs_H else [True]
    loss_type_ls = ['MSE'] if batch_vs_H else ['Cross-Entropy']
    bnorm_ls = [[True, False], [True, True], [False, False]]
    H_ls = [2, 4, 8, 16, 32] if batch_vs_H else [16]
    batch_size_ls = [int(N/40), int(N/20), int(N/10)] if batch_vs_H else [int(N/20)]
    more_layers_ls = [False, True]
    N = 2000
    num_epochs = 200
    num_divide = 40
    C = 2
    freq = int(num_epochs//num_divide)
plot_neuron_dynamics = False
if plot_neuron_dynamics:
    # Then all ls above are reduced in size, since I do not plot so many neuron dynamics.
    Adam_ls, splus_ls, loss_type_ls, bnorm_ls, H_ls, batch_size_ls = [
        False], [False], ['MSE'], [[False, False]], [16], [int(N/20)]
make_plot = True  # If just to get table, then need not make plot
Table_dict = {}  # So it is more_layer+batch_norm per H_ls and batch_size_ls
for more_layers in more_layers_ls:
    for Adam in Adam_ls:
        opt_type = '_Adam' if Adam else ''
        for splus in splus_ls:
            beta = 5  # This is the best value for SGD
            for loss_type in loss_type_ls:
                for batch_norm, freeze_BN in bnorm_ls:
                    Table = np.zeros((len(H_ls), 2*24))
                    columns = np.tile(['SGD training mean', 'SGD training SE', 'SGD test mean', 'SGD test SE',
                                       'SVI training mean', 'SVI training SE', 'SVI test mean', 'SVI test SE'], 6)
                    type = np.repeat([f'{loss_type} loss', '$l_{\infty}$ error'], 24)
                    tuples = list(zip(*[type, columns]))
                    index = pd.MultiIndex.from_tuples(tuples)
                    for k, H in enumerate(H_ls):
                        long_ls_loss = []
                        long_ls_linferror = []
                        for batch_size in batch_size_ls:
                            half = '_half' if freeze_BN else ''
                            splus_suff = '_splus' if splus else ''
                            bnorm = '_bnorm' if batch_norm else ''
                            bsize = f'_bsize={batch_size}'
                            more_layer = '_more_layers' if more_layers else ''
                            name1 = f'SGD_VI_dynamics_H={H}_{loss_type}{opt_type}_ref{splus_suff}{bnorm}{half}{more_layer}{bsize}.pth'
                            name2 = f'SGD_VI_dynamics_H={H}_{loss_type}{opt_type}_SGD{splus_suff}{bnorm}{half}{more_layer}{bsize}.pth'
                            name3 = f'SGD_VI_dynamics_H={H}_{loss_type}{opt_type}_VI{splus_suff}{bnorm}{half}{more_layer}{bsize}.pth'
                            model = GCN_SGD(C, H=H, splus=splus, beta=beta,
                                            batch_norm=batch_norm).to(device)
                            if more_layers:
                                model = GCN_more_layer_SGD(C, H1=H, H2=H, splus=splus, beta=beta,
                                                           batch_norm=batch_norm).to(device)
                            model.load_state_dict(torch.load(
                                name1, map_location=torch.device('cpu')))
                            SGD_dict_ref = copy.deepcopy(model.state_dict())
                            VI_dict_ref = copy.deepcopy(SGD_dict_ref)
                            model.load_state_dict(torch.load(
                                name2, map_location=torch.device('cpu')))
                            SGD_dict_final = copy.deepcopy(model.state_dict())
                            model = GCN_VI(C, H=H, splus=splus, beta=beta,
                                           batch_norm=batch_norm).to(device)
                            if more_layers:
                                model = GCN_more_layer_VI(C, H1=H, H2=H, splus=splus, beta=beta,
                                                          batch_norm=batch_norm).to(device)
                            model.load_state_dict(torch.load(
                                name3, map_location=torch.device('cpu')))
                            VI_dict_final = copy.deepcopy(model.state_dict())
                            name = f'SGD_VI_dynamics_H={H}_{loss_type}{opt_type}{splus_suff}{bnorm}{half}{more_layer}{bsize}.json'
                            with open(name, 'r') as j:
                                result_dict = json.loads(j.read())
                                result_dict = ast.literal_eval(result_dict)
                            key = 'conv2.lin.weight' if more_layers else 'conv1.lin.weight'
                            w_ref = SGD_dict_ref[key].cpu().detach().numpy()
                            w_SGD = SGD_dict_final[key].cpu().detach().numpy()
                            w_VI = VI_dict_final[key].cpu().detach().numpy()
                            w_ref_inner = np.sum(w_ref*w_ref, axis=1)
                            w_SGD_final = np.sum(w_ref*w_SGD, axis=1)
                            w_VI_final = np.sum(w_ref*w_VI, axis=1)
                            key2 = 'conv3.lin.weight' if more_layers else 'conv2.lin.weight'
                            a_SGD_ref = SGD_dict_ref[key2].cpu().detach().numpy().flatten()
                            a_VI_ref = VI_dict_ref[key2].cpu().detach().numpy().flatten()
                            a_SGD_final = SGD_dict_final[key2].cpu().detach().numpy().flatten()
                            a_VI_final = VI_dict_final[key2].cpu().detach().numpy().flatten()
                            a_SGD = [a_SGD_ref, a_SGD_final]
                            a_VI = [a_VI_ref, a_VI_final]
                            w_SGD = [w_ref_inner, w_SGD_final]
                            w_VI = [w_ref_inner, w_VI_final]
                            para_error_vanilla_train, para_error_vanilla_trainSE, pred_l2error_vanilla_train, pred_l2error_vanilla_trainSE, pred_linferror_vanilla_train, pred_linferror_vanilla_trainSE, pred_loss_vanilla_train, pred_loss_vanilla_trainSE = utils_layer.get_all(
                                result_dict['SGD_train'])
                            para_error_vanilla, para_error_vanillaSE, pred_l2error_vanilla, pred_l2error_vanillaSE, pred_linferror_vanilla, pred_linferror_vanillaSE, pred_loss_vanilla, pred_loss_vanillaSE = utils_layer.get_all(
                                result_dict['SGD_test'])
                            para_error_VI_train, para_error_VI_trainSE, pred_l2error_VI_train, pred_l2error_VI_trainSE, pred_linferror_VI_train, pred_linferror_VI_trainSE, pred_loss_VI_train, pred_loss_VI_trainSE = utils_layer.get_all(
                                result_dict['VI_train'])
                            para_error_VI, para_error_VISE, pred_l2error_VI, pred_l2error_VISE, pred_linferror_VI, pred_linferror_VISE, pred_loss_VI, pred_loss_VISE = utils_layer.get_all(
                                result_dict['VI_test'])
                            linferror_ls = [pred_linferror_vanilla_train, pred_linferror_vanilla_trainSE, pred_linferror_vanilla,
                                            pred_linferror_vanillaSE, pred_linferror_VI_train, pred_linferror_VI_trainSE, pred_linferror_VI, pred_linferror_VISE]
                            loss_ls = [pred_loss_vanilla_train, pred_loss_vanilla_trainSE, pred_loss_vanilla,
                                       pred_loss_vanillaSE, pred_loss_VI_train, pred_loss_VI_trainSE, pred_loss_VI, pred_loss_VISE]
                            long_ls_loss += [a[-1] for a in loss_ls]
                            long_ls_linferror += [a[-1] for a in linferror_ls]
                            if make_plot:
                                fig, fig1 = plot_dynamics(
                                    a_SGD, w_SGD, a_VI, w_VI,  linferror_ls, loss_ls, freq, more_layers, batch_norm, freeze_BN, plot_neuron_dynamics)
                                dynamics = '_ndynamics' if plot_neuron_dynamics else ''
                                if fig1 != 0:
                                    fig.savefig(f'SGD_VI_dynamics_H={H}_{loss_type}{opt_type}{splus_suff}{bnorm}{half}{more_layer}{bsize}_loss.pdf',
                                                dpi=300, bbox_inches='tight', pad_inches=0)
                                    fig1.savefig(f'SGD_VI_dynamics_H={H}_{loss_type}{opt_type}{splus_suff}{bnorm}{half}{more_layer}{bsize}_linferror.pdf',
                                                 dpi=300, bbox_inches='tight', pad_inches=0)
                                else:
                                    fig.savefig(f'SGD_VI_dynamics_H={H}_{loss_type}{opt_type}{splus_suff}{bnorm}{half}{more_layer}{bsize}{dynamics}.pdf',
                                                dpi=300, bbox_inches='tight', pad_inches=0)
                        if batch_vs_H and plot_neuron_dynamics == False:
                            # This is because otherwise, only have one batch size
                            Table[k] = long_ls_loss+long_ls_linferror
                    Table = pd.DataFrame(Table, index=H_ls, columns=index)
                    Table.index.name = 'H'
                    layername = 'more_layer' if more_layers else 'one_layer'
                    normname = 'BN' if batch_norm else 'noBN'
                    print(f'{layername}+{normname}{half}')
                    Table_dict[f'{layername}+{normname}{half}'] = Table


# Get the summary table
Table_dict.keys()
# Find the best result entry-wise
for type in ['one', 'more']:
    in_keys = []
    for key in Table_dict.keys():
        if type in key:
            in_keys.append(key)
    shape = Table_dict[in_keys[0]].shape
    Table_merged = np.zeros(shape)
    import itertools
    for a, b in itertools.product(*[np.arange(shape[0]), np.arange(int(shape[1]/2))]):
        small_mean = [Table_dict[k].iloc[a, 2*b] for k in in_keys]
        small_var = [Table_dict[k].iloc[a, 2*b+1] for k in in_keys]
        p = np.argmin(small_mean)
        Table_merged[a, 2*b] = small_mean[p]
        Table_merged[a, 2*b+1] = small_var[p]
    Table_merged = pd.DataFrame(
        Table_merged, index=Table_dict[in_keys[0]].index, columns=Table_dict[in_keys[0]].columns)
    Table_dict[f'{type}_layer+best'] = Table_merged

columns = np.tile(['SGD training', 'SGD test', 'SVI training', 'SVI test'], 6)
type = np.repeat([f'{loss_type} loss', '$l_{\infty}$ error'], 12)
tuples = list(zip(*[type, columns]))
new_colindex = pd.MultiIndex.from_tuples(tuples)
round_more = True
ipb.reload(sys.modules['utils_gnn_VI'])
Table_new_one_layer_noBN = utils_layer.concatenat_to_one(
    Table_dict['one_layer+noBN'], new_colindex, round_more)
Table_new_one_layer_best = utils_layer.concatenat_to_one(
    Table_dict['one_layer+best'], new_colindex, round_more)
Table_new_more_layer_noBN = utils_layer.concatenat_to_one(
    Table_dict['more_layer+noBN'], new_colindex, round_more)
Table_new_more_layer_best = utils_layer.concatenat_to_one(
    Table_dict['more_layer+best'], new_colindex, round_more)
# Print them to latex one by one
print(Table_new_more_layer_noBN.to_latex(escape=False))


# Sanity check: No hidden layer model


F_out_ls = [1]
C_ls = [5, 10, 15]
batch_size_ls = [50, 100, 200]
seeds = [1103, 1111, 1214]
N, N1 = 2000, 2000
loss_type, Adam, num_epochs, graph_type = 'MSE', False, 20, 'small'
for F_out in F_out_ls:
    for C in C_ls:
        for batch_size in batch_size_ls:
            names = ['VI_train', 'VI_test']
            result_dict = {name: {} for name in names}
            for seed in seeds:
                ipb.reload(sys.modules['utils_gnn_VI'])
                # Generate true data
                n = 15
                mu = 0.5
                sigma = 1
                np.random.seed(2)
                W1 = np.random.normal(mu, sigma, F_out*C).reshape((F_out, C)).astype(np.float32)
                b1 = np.random.normal(mu, sigma, F_out).astype(np.float32)  # F-by-1
                G = nx.fast_gnp_random_graph(n=n, p=0.15, seed=1103)
                edge_index = torch.tensor(list(G.edges)).T.type(torch.long)
                pertub = 0.2 if graph_type == 'small' else 0.05
                G_est = utils.G_reformat(G, percent_perturb=pertub, return_G=True)
                # utils_layer.draw_graph(edge_index, edge_index_est, graph_type)
                model_get_data = GCN_VI_simple(C, F_out).to(device)
                # NOTE: another way to change parameters, which FORCES me to make sure parameters match the shape I want
                old_dict = model_get_data.state_dict()
                old_dict['conv1.bias'] = torch.from_numpy(b1)
                old_dict['conv1.lin.weight'] = torch.from_numpy(W1)
                model_get_data.load_state_dict(old_dict)
                model_get_data.eval()
                X_train, Y_train = utils_layer.get_simulation_data(
                    model_get_data, N, edge_index, n, C, torch_seed=seed)
                len(X_train)
                X_test, Y_test = utils_layer.get_simulation_data(
                    model_get_data, N1, edge_index, n, C, train=False, torch_seed=seed)
                len(X_test)
                train_loader, test_loader = utils_layer.get_train_test_loader(
                    X_train, X_test, Y_train, Y_test, edge_index, batch_size)
                # Estimation
                torch.manual_seed(seed)  # For reproducibility
                model_VI1 = GCN_VI_simple(C, F_out).to(device)
                model_to_feature_ls = [0]  # Place holder, never used
                para_error_VI_train = []
                pred_l2error_VI_train = []
                pred_linferror_VI_train = []
                pred_loss_VI_train = []
                para_error_VI = []
                pred_l2error_VI = []
                pred_linferror_VI = []
                pred_loss_VI = []
                num_divide = 40
                freq = int(num_epochs//num_divide)
                for epoch in range(num_epochs):
                    print(f'VI epoch {epoch}')
                    train_loss = utils_layer.train_revised_all_layer(train_loader,
                                                                     model_to_train=model_VI1, output_dim=F_out, model_to_feature_ls=model_to_feature_ls, loss_type=loss_type, Adam=Adam)
                    if np.mod(epoch+1, freq) == 0:
                        para_err, l2_err, linf_err, loss_true = utils_layer.evaluation_simulation(
                            train_loader, model_get_data, model_VI1, W2=[], b2=[], data_loader_true=train_loader, loss_type=loss_type)
                        para_error_VI_train.append(para_err)
                        pred_l2error_VI_train.append(l2_err)
                        pred_linferror_VI_train.append(linf_err)
                        pred_loss_VI_train.append(loss_true)
                        print(
                            f'[Train: rel Para err, rel l2 err, rel linf err, loss] at {epoch} is \n {[para_err, l2_err, linf_err, loss_true]}')
                        para_err, l2_err, linf_err, loss_true = utils_layer.evaluation_simulation(
                            test_loader, model_get_data, model_VI1, W2=[], b2=[], data_loader_true=test_loader, loss_type=loss_type)
                        para_error_VI.append(para_err)
                        pred_l2error_VI.append(l2_err)
                        pred_linferror_VI.append(linf_err)
                        pred_loss_VI.append(loss_true)
                        print(
                            f'[Test: rel Para err, rel l2 err, rel linf err, loss] at {epoch} is \n {[para_err, l2_err, linf_err, loss_true]}')
                result_dict['VI_train'][seed] = [para_error_VI_train, pred_l2error_VI_train,
                                                 pred_linferror_VI_train, pred_loss_VI_train]
                result_dict['VI_test'][seed] = [para_error_VI,
                                                pred_l2error_VI, pred_linferror_VI, pred_loss_VI]
                VI_dict_final = copy.deepcopy(model_VI1.state_dict())
                opt_type = '_Adam' if Adam else ''
                bsize = f'_bsize={batch_size}'
                name3 = f'SGD_VI_dynamics_C={C}_F={F_out}_{loss_type}{opt_type}_VI{bsize}.pth'
                torch.save(VI_dict_final, name3)
                # Save after all seeds
                json_linf = json.dumps(str(result_dict))
                name = f'SGD_VI_dynamics_C={C}_F={F_out}_{loss_type}{opt_type}{bsize}'
                f = open(f"{name}.json", "w")
                # write json object to file
                f.write(json_linf)
                # close file
                f.close()
                para_error_VI_train, para_error_VI_trainSE, pred_l2error_VI_train, pred_l2error_VI_trainSE, pred_linferror_VI_train, pred_linferror_VI_trainSE, pred_loss_VI_train, pred_loss_VI_trainSE = utils_layer.get_all(
                    result_dict['VI_train'])
                para_error_VI, para_error_VISE, pred_l2error_VI, pred_l2error_VISE, pred_linferror_VI, pred_linferror_VISE, pred_loss_VI, pred_loss_VISE = utils_layer.get_all(
                    result_dict['VI_test'])
                linferror_ls = [pred_linferror_VI_train, pred_linferror_VI_trainSE,
                                pred_linferror_VI, pred_linferror_VISE]
                loss_ls = [pred_loss_VI_train, pred_loss_VI_trainSE, pred_loss_VI, pred_loss_VISE]
                # fig = plot_quick(linferror_ls, loss_ls, freq, C)
                # fig.savefig(f'SGD_VI_dynamics_C={C}_F={F_out}_{loss_type}{opt_type}{bsize}.pdf',
                #             dpi=300, bbox_inches='tight', pad_inches=0)

# Load json files and plot


def plot_quick(linferror_ls, loss_ls, freq, C):
    pred_linferror_VI_train, pred_linferror_VI_trainSE, pred_linferror_VI, pred_linferror_VISE = linferror_ls
    pred_loss_VI_train, pred_loss_VI_trainSE, pred_loss_VI, pred_loss_VISE = loss_ls
    # Parameters
    plt.rcParams['xtick.labelsize'] = 18
    plt.rcParams['ytick.labelsize'] = 18
    plt.rcParams['legend.fontsize'] = 18
    fig, ax = plt.subplots(figsize=(7, 4))
    if C == 15:
        start = 0.12
        end = 0.15
    if C == 10:
        start = 0.2
        end = 0.25
    if C == 5:
        start = 0.2
        end = 0.23
    ax.set_ylim([start, end])
    ax.yaxis.set_ticks(np.arange(start, end, (end-start)/10))
    ax.set_xlabel("Epoch", fontsize=20)
    ax.set_ylabel('Loss', fontsize=20)
    xaxis = range(len(pred_loss_VI_train))
    ax.plot(pred_loss_VI_train, linestyle='dashed', label=f'SVI training', color='orange')
    ax.fill_between(xaxis, pred_loss_VI_train-pred_loss_VI_trainSE, pred_loss_VI_train+pred_loss_VI_trainSE,
                    color='orange', alpha=0.1)
    ax.plot(pred_loss_VI, label=f'SVI test', color='orange')
    ax.fill_between(xaxis, pred_loss_VI-pred_loss_VISE, pred_loss_VI+pred_loss_VISE,
                    color='orange', alpha=0.1)
    ax.tick_params(labelsize=18)
    labels = [item*freq for item in ax.get_xticks()]
    labels = np.array(labels, dtype=int)
    ax.set_xticklabels(labels)
    ax.grid(which='both')
    fig1, ax = plt.subplots(figsize=(7, 4))
    if C == 15:
        start = 0.02
        end = 0.5
    if C == 10:
        start = 0.018
        end = 0.36
    if C == 5:
        start = 0.013
        end = 0.15
    ax.set_ylim([start, end])
    ax.yaxis.set_ticks(np.arange(start, end, (end-start)/10))
    ax.set_xlabel("Epoch", fontsize=20)
    ax.set_ylabel(r"$l_{\infty}$ Error", fontsize=20)
    xaxis = range(len(pred_linferror_VI_train))
    ax.plot(pred_linferror_VI_train, linestyle='dashed', label=f'SVI training', color='orange')
    ax.fill_between(xaxis, pred_linferror_VI_train-pred_linferror_VI_trainSE, pred_linferror_VI_train+pred_linferror_VI_trainSE,
                    color='orange', alpha=0.1)
    ax.plot(pred_linferror_VI, label=f'SVI test', color='orange')
    ax.fill_between(xaxis, pred_linferror_VI-pred_linferror_VISE, pred_linferror_VI+pred_linferror_VISE,
                    color='orange', alpha=0.1)
    ax.tick_params(labelsize=18)
    labels = [item*freq for item in ax.get_xticks()]
    labels = np.array(labels, dtype=int)
    ax.set_xticklabels(labels)
    ax.grid(which='both')
    # ax.legend(ncol=2, loc='lower right', bbox_to_anchor=(1, -0.5))
    return [fig, fig1]


F_out_ls = [1]
C_ls = [5, 10, 15]
batch_size_ls = [50, 100, 200]
num_epochs = 200
num_divide = 40
freq = int(num_epochs//num_divide)
loss_type, opt_type = 'MSE', ''
for F_out in F_out_ls:
    for C in C_ls:
        for batch_size in batch_size_ls:
            bsize = f'_bsize={batch_size}'
            name = f'SGD_VI_dynamics_C={C}_F={F_out}_{loss_type}{opt_type}{bsize}.json'
            with open(name, 'r') as j:
                result_dict = json.loads(j.read())
                result_dict = ast.literal_eval(result_dict)
            para_error_VI_train, para_error_VI_trainSE, pred_l2error_VI_train, pred_l2error_VI_trainSE, pred_linferror_VI_train, pred_linferror_VI_trainSE, pred_loss_VI_train, pred_loss_VI_trainSE = utils_layer.get_all(
                result_dict['VI_train'])
            para_error_VI, para_error_VISE, pred_l2error_VI, pred_l2error_VISE, pred_linferror_VI, pred_linferror_VISE, pred_loss_VI, pred_loss_VISE = utils_layer.get_all(
                result_dict['VI_test'])
            linferror_ls = [pred_linferror_VI_train, pred_linferror_VI_trainSE,
                            pred_linferror_VI, pred_linferror_VISE]
            loss_ls = [pred_loss_VI_train, pred_loss_VI_trainSE, pred_loss_VI, pred_loss_VISE]
            print(
                f'F_out={F_out}, C={C}, bsize={batch_size}\n loss, linf error = {pred_loss_VI_train[-1]},{pred_linferror_VI_train[-1]}')
            fig, fig1 = plot_quick(linferror_ls, loss_ls, freq, C)
            fig.savefig(f'SGD_VI_dynamics_C={C}_F={F_out}_{loss_type}{opt_type}{bsize}_loss.pdf',
                        dpi=300, bbox_inches='tight', pad_inches=0)
            fig1.savefig(f'SGD_VI_dynamics_C={C}_F={F_out}_{loss_type}{opt_type}{bsize}_linferror.pdf',
                         dpi=300, bbox_inches='tight', pad_inches=0)
