import pandas as pd
import json
import ast
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score
import utils_gnn_VI as utils_layer
from torch.autograd import Variable
from matplotlib.ticker import MaxNLocator
from scipy.linalg import block_diag
from scipy.linalg import sqrtm
import networkx as nx
import matplotlib.pyplot as plt
import sys
import importlib as ipb
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import torch
import scipy.io
import os
import numpy as np
import matplotlib
import matplotlib as mpl
mpl.use('pgf')
mpl.rc('text', usetex=True)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams['font.size'] = 24
plt.rcParams['axes.titlesize'] = 24
plt.rcParams['figure.titlesize'] = 30

'''Code Layout:
    The exact line numbers can be a little off
    Run both the solar binary classification and the traffic multi-class classification
        Change the dataset argument before data_ls as either 'solar' or 'traffic'
'''

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GCN_more_layer_SGD(torch.nn.Module):
    def __init__(self, C, F_out=1, H1=128, H2=128, batch_norm=False):
        super().__init__()
        print(f'{[H1,H2]} hidden nodes')
        self.conv1 = GCNConv(C, H1)
        self.conv2 = GCNConv(H1, H2)
        self.conv3 = GCNConv(H2, F_out)
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
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        if self.BN2 != None:
            x = self.BN2(x)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        if F_out == 1:
            return torch.sigmoid(x)
        else:
            return F.log_softmax(x, dim=1)


class GCN_more_layer_VI(torch.nn.Module):
    def __init__(self, C, F_out=1, H1=128, H2=128, batch_norm=False):
        super().__init__()
        print(f'{[H1,H2]} hidden nodes')
        self.conv1 = GCNConv(C, H1)
        self.conv2 = GCNConv(H1, H2)
        self.conv3 = GCNConv(H2, F_out)
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
            self.rmean = self.BN1.running_mean
            self.rvar = self.BN1.running_var
            self.weight = torch.ones(H).to(device)
            self.bias = torch.zeros(H).to(device)
            if self.affine:
                self.weight = self.BN1.weight.clone()
                self.bias = self.BN1.bias.clone()
        layer1_x = F.relu(layer1_x)
        if feature2:
            return self.conv2_feature(layer1_x, edge_index)
        self.layer1_x = Variable(layer1_x, requires_grad=True)
        layer2_x = self.conv2(self.layer1_x, edge_index)
        # Normalize after convolution & before activation
        if self.BN2 != None:
            layer1_x = self.BN2(layer1_x)
            H = layer1_x.shape[1]
            self.rmean1 = self.BN2.running_mean
            self.rvar1 = self.BN2.running_var
            self.weight1 = torch.ones(H).to(device)
            self.bias1 = torch.zeros(H).to(device)
            if self.affine:
                self.weight1 = self.BN2.weight.clone()
                self.bias1 = self.BN2.bias.clone()
        layer2_x = F.relu(layer2_x)
        self.layer2_x = layer2_x
        self.layer2_x.retain_grad()
        if feature3:
            return self.conv3_feature(layer2_x, edge_index)
        x = self.conv3(self.layer2_x, edge_index)
        if F_out == 1:
            return torch.sigmoid(x)
        else:
            return F.log_softmax(x, dim=1)


# On 'ATL', adding layer or not does not really matter a lot. Ignore this example. For 'CA': run sparse graph. For 'LA': run fully connected
dataset = 'solar'  # 'solar' or 'traffic'
data_ls = ['CA', 'LA'] if dataset == 'solar' else ['Traffic']
# data_ls = ['CA'] if dataset == 'solar' else ['Traffic']
graph_connect = {'CA': False, 'LA': True}
H_ls = [8, 16, 32, 64]
# H_ls = [32]
para_dict = {'CA': [10, 5, 30, 100, 365, 1], 'LA': [
    9, 5, 30, 100, 365, 1], 'Traffic': [20, 5, 4, 600, 100, 6138, 3]}
seed_ls = [1103, 1111, 1214]
# seed_ls = [1103]
savefig = True  # True for solar, because it is very efficient
loss_ls = ['Cross-Entropy', 'MSE']
# loss_ls = ['Cross-Entropy']
Adam = False
opt_type = '_Adam' if Adam else ''
bnorm_ls = [[True, False], [True, True], [False, False]]
bnorm_ls = [[True, True]]
for loss_type in loss_ls:
    for (H_vanilla, H_VI) in zip(H_ls, H_ls):
        for city in data_ls:
            for batch_norm, freeze_BN in bnorm_ls:
                result_SGD_no_layer_dict = {}
                result_VI_no_layer_dict = {}
                result_SGD1_dict = {}
                result_VI1_dict = {}
                for seed in seed_ls:
                    ipb.reload(sys.modules['VI_class'])
                    ipb.reload(sys.modules['utils_gnn_VI_no_layer'])
                    ipb.reload(sys.modules['utils_gnn_VI'])
                    # Process data in memory
                    if city == 'Traffic':
                        use_subdata = False
                        n, num_neighbor, d, batch_size, num_epochs, N, F_out = para_dict[city]
                        X_train, X_test, Y_train, Y_test, edge_index = utils_layer.get_traffic_train_test(
                            num_neighbor, d, use_subdata)
                        fully_connected = False
                    else:
                        n, d, batch_size, num_epochs, N, F_out = para_dict[city]
                        test_data = scipy.io.loadmat(f'Data/{city}_2018_single.mat')
                        train_data = scipy.io.loadmat(f'Data/{city}_2017_single.mat')
                        train_data = train_data['obs'].reshape((N, n))
                        test_data = test_data['obs'].reshape((N, n))
                        X_train, X_test, Y_train, Y_test = utils_layer.get_solar_train_test(
                            train_data, test_data, N, d=d)
                        count_train = np.unique(Y_train, return_counts=True)[1]
                        count_test = np.unique(Y_test, return_counts=True)[1]
                        print(
                            f'#1/#0 in training data is {count_train[1]/count_train[0]}')
                        print(f'#1/#0 in test data is {count_test[1]/count_test[0]}')
                        fully_connected = graph_connect[city]
                        edge_index = utils_layer.get_edge_list(
                            Y_train, n, fully_connected)
                    freq = int(num_epochs/25)
                    tot = N - d
                    indices = []
                    for i in range(tot // batch_size):
                        indices.append(np.arange(i * batch_size, (i + 1) * batch_size))
                    k = np.mod(tot, batch_size)
                    if k > 0:
                        indices.append(np.arange(tot - k, tot))
                    # Prepare the data for PyTorch Geometric, which is solved using inherent gradient descent technique
                    # From FAQ https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html#frequently-asked-questions,
                    train_loader, test_loader = utils_layer.get_train_test_loader(
                        X_train, X_test, Y_train, Y_test, edge_index, batch_size)
                    # Multi-layer SGD
                    print('Running SGD multi layer')
                    torch.manual_seed(seed)  # For reproducibility
                    model_vanilla = GCN_more_layer_SGD(
                        d, H1=H_vanilla, H2=H_vanilla, F_out=F_out, batch_norm=batch_norm).to(device)
                    mod_vanilla = utils_layer.GCN_train(
                        model_vanilla, train_loader, test_loader, model_get_data=None, more_layers=True)
                    result_SGD1_dict[f'Seed {seed}'] = mod_vanilla.training_and_eval(
                        num_epochs, compute_para_err=False, output_dim=F_out, loss_type=loss_type, Adam=Adam, freq=freq, freeze_BN=freeze_BN)
                    # Multi-layer VI
                    print('Running VI multi layer')
                    torch.manual_seed(seed)  # For reproducibility
                    model_VI = GCN_more_layer_VI(d, H1=H_VI, H2=H_VI, F_out=F_out, batch_norm=batch_norm).to(
                        device)  # Initialize a model
                    mod_VI = utils_layer.GCN_train(
                        model_VI, train_loader, test_loader, model_get_data=None, more_layers=True)
                    model_to_feature_ls = [0]  # Place holder, never used
                    result_VI1_dict[f'Seed {seed}'] = mod_VI.training_and_eval(
                        num_epochs, compute_para_err=False, output_dim=F_out, model_to_feature_ls=model_to_feature_ls, loss_type=loss_type, Adam=Adam, freq=freq, freeze_BN=freeze_BN)
                    sub_fix = '_layer'
                    bnorm = '_bnorm' if batch_norm else ''
                    half = '_half' if freeze_BN else ''
                    json_SGD = json.dumps(str(result_SGD1_dict))
                    json_VI = json.dumps(str(result_VI1_dict))
                    name = f'SGD_{city}_H={H_vanilla}{sub_fix}_{loss_type}{opt_type}{bnorm}{half}'
                    name1 = f'VI_{city}_H={H_VI}{sub_fix}_{loss_type}{opt_type}{bnorm}{half}'
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

# Load json and replot
dataset = 'traffic'  # solar or 'traffic'
data_ls = ['CA', 'LA'] if dataset == 'solar' else ['Traffic']
# data_ls = ['CA'] if dataset == 'solar' else ['Traffic']
H_ls = [8, 16, 32, 64]  # 64, 256
# H_ls = [8]  # 64, 256
bnorm_ls = [[True, False], [True, True], [False, False]] if dataset == 'solar' else [[False, False]]
loss_ls = ['Cross-Entropy'] if dataset == 'traffic' else ['MSE']
Adam = False
opt_type = '_Adam' if Adam else ''
graph_connect = {'CA': False, 'LA': True}
num_epochs = 100
Table_dict = {}
replot = True  # Set to true if want to remake ALL plots
savefig = True  # Always set true

for batch_norm, freeze_BN in bnorm_ls:
    for loss_type in loss_ls:
        for city in data_ls:
            Table = np.zeros((len(H_ls), 2*12))
            for idx, (H_vanilla, H_VI) in enumerate(zip(H_ls, H_ls)):
                normname = 'BN' if batch_norm else 'noBN'
                half = 'half' if freeze_BN else ''
                print(f'At {city}, H={H_VI}, {normname}{half}')
                ipb.reload(sys.modules['utils_gnn_VI'])
                fully_connected = False
                if dataset == 'solar':
                    fully_connected = graph_connect[city]
                columns = np.tile(['SGD Training mean', 'SGD Training SE', 'SGD Test mean', 'SGD Test SE',
                                   'VI-SGD Training mean', 'VI-SGD Training SE', 'VI-SGD Test mean', 'VI-SGD Test SE'], 3)
                type = np.repeat([f'{loss_type} loss', 'Classification error',
                                  f'Weighted $F_1$ score'], 8)
                tuples = list(zip(*[type, columns]))
                freq = int(num_epochs/25)
                index = pd.MultiIndex.from_tuples(tuples)
                sub_fix = '_layer'
                bnorm = '_bnorm' if batch_norm else ''
                half = '_half' if freeze_BN else ''
                name = f'SGD_{city}_H={H_vanilla}{sub_fix}_{loss_type}{opt_type}{bnorm}{half}.json'
                name1 = f'VI_{city}_H={H_VI}{sub_fix}_{loss_type}{opt_type}{bnorm}{half}.json'
                with open(name, 'r') as j:
                    result_SGD1_dict = json.loads(j.read())
                    result_SGD1_dict = ast.literal_eval(result_SGD1_dict)
                with open(name1, 'r') as j:
                    result_VI1_dict = json.loads(j.read())
                    result_VI1_dict = ast.literal_eval(result_VI1_dict)
                losses_vanilla = utils_layer.get_all_real(
                    result_SGD1_dict)
                losses_VI = utils_layer.get_all_real(
                    result_VI1_dict)
                long_ls = []
                for i in range(0, 3, 2):
                    long_ls += [losses_vanilla[i][-1], losses_vanilla[i+1][-1]]
                for i in range(0, 3, 2):
                    long_ls += [losses_VI[i][-1], losses_VI[i+1][-1]]
                for i in range(4, 7, 2):
                    long_ls += [losses_vanilla[i][-1], losses_vanilla[i+1][-1]]
                for i in range(4, 7, 2):
                    long_ls += [losses_VI[i][-1], losses_VI[i+1][-1]]
                for i in range(8, 11, 2):
                    long_ls += [losses_vanilla[i][-1], losses_vanilla[i+1][-1]]
                for i in range(8, 11, 2):
                    long_ls += [losses_VI[i][-1], losses_VI[i+1][-1]]
                Table[idx] = long_ls
                city_name = 'solar_' + city if city != 'Traffic' else city
                if replot:
                    utils_layer.GNN_VI_layer_plt_real_data(num_epochs, losses_vanilla,
                                                           losses_VI, fully_connected, city_name, H_vanilla, H_VI, savefig=savefig, loss_type=loss_type, Adam=Adam, batch_norm=batch_norm, freq=freq, freeze_BN=freeze_BN)
                if (loss_type == 'Cross-Entropy' and city == 'Traffic' and H_VI == 64) or (loss_type == 'MSE' and city == 'CA' and H_VI == 64):
                    utils_layer.GNN_VI_layer_plt_real_data(num_epochs, losses_vanilla,
                                                           losses_VI, fully_connected, city_name, H_vanilla, H_VI, savefig=savefig, loss_type=loss_type, Adam=Adam, single_plot=True, batch_norm=batch_norm, freq=freq, freeze_BN=freeze_BN)
            Table = pd.DataFrame(Table, index=H_ls, columns=index)
            Table.index.name = '# Hidden nodes'
            Table_dict[loss_type+city] = Table

# NOTE: we either run batch norm or not, so need to be careful with the table :)
columns = np.tile(['SGD Training', 'SGD Test', 'VI-SGD Training', 'VI-SGD Test'], 3)
type = np.repeat([f'{loss_type} loss', 'Classification error',
                  f'Weighted $F_1$ score'], 4)
tuples = list(zip(*[type, columns]))
new_colindex = pd.MultiIndex.from_tuples(tuples)

list(Table_dict.keys())

round_more = False
if dataset == 'solar':
    Table_new_CA = utils_layer.concatenat_to_one(Table_dict['MSECA'], new_colindex, round_more)
    Table_new_LA = utils_layer.concatenat_to_one(Table_dict['MSELA'], new_colindex, round_more)
    print(Table_new_CA.to_latex(escape=False))
    print(Table_new_LA.to_latex(escape=False))
if dataset != 'solar':
    Table_new_traffic = utils_layer.concatenat_to_one(
        Table_dict['Cross-EntropyTraffic'], new_colindex, round_more)
    print(Table_new_traffic.to_latex(escape=False))
