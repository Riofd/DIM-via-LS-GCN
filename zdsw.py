import torch
from torch_geometric.data import Data, DataLoader, InMemoryDataset
import argparse
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv, NNConv, GATConv
from torch_geometric.nn import SAGPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch import nn
from zdsw_Dataset import zdswDataset
import os
import matplotlib.pyplot as plt
from smoothing import LabelSmoothingCrossEntropy, smooth_one_hot, LabelSmoothingLoss, ECELoss
from sklearn.metrics import f1_score
import scipy.io as sio
import time
from torch_scatter import scatter_mean, scatter_max
from torchsummary import summary

torch.cuda.set_device(2)
os.environ['CUDA_VISIBLE_DEVICES'] = '2'


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64,
                    help='batch size')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='momentum')
parser.add_argument('--weight_decay', type=float, default=0.0005,
                    help='weight decay')
parser.add_argument('--nhid', type=int, default=128,
                    help='hidden size')
parser.add_argument('--pooling_ratio', type=float, default=0.5,
                    help='pooling ratio')
parser.add_argument('--dropout_ratio', type=float, default=0.4,
                    help='dropout ratio')
parser.add_argument('--epochs', type=int, default=500,
                    help='maximum number of epochs')
parser.add_argument('--patience', type=int, default=50,
                    help='patience for early stop')
parser.add_argument('--pooling_layer_type', type=str, default='GCNConv',
                    help='convolution type for pooling')
parser.add_argument('--label_smoothing', type=float, default=0.1, metavar='S', help='label smoothing')
args = parser.parse_args()

dataset = zdswDataset(root='./dataset', dataname='bus_vol_mag_phase')
dataset = dataset.shuffle()
batch_size = args.batch_size
num_samples = 6800
train_dataset = dataset[:int(0.7*num_samples)]
trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = dataset[int(0.7*num_samples):int(0.9*num_samples)]
valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_dataset = dataset[int(0.9*num_samples):]
# test_dataset = dataset[:]
testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
train_size = int(0.7*num_samples)
d = train_dataset


class Net(torch.nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.args = args
        self.pooling_ratio = args.pooling_ratio
        self.nhid = args.nhid
        self.dropout_ratio = args.dropout_ratio
        self.conv1 = GCNConv(dataset.num_features, self.nhid)
        self.pool1 = SAGPooling(self.nhid, ratio=self.pooling_ratio)
        self.bn1 = nn.BatchNorm1d(self.nhid, eps=1e-05, momentum=0.1, affine=True)
        self.conv2 = GCNConv(self.nhid, self.nhid)
        self.pool2 = SAGPooling(self.nhid, ratio=self.pooling_ratio)
        self.conv3 = GCNConv(self.nhid, self.nhid)
        self.pool3 = SAGPooling(self.nhid, ratio=self.pooling_ratio)

        self.lin1 = nn.Linear(self.nhid, self.nhid)
        self.lin2 = nn.Linear(self.nhid, self.nhid//2)
        self.lin3 = nn.Linear(self.nhid//2, dataset.num_classes)
        # self.conv1 = GATConv(dataset.num_features, 128, heads=5, concat=True, dropout=0.5)
        # self.conv2 = GATConv(5 * 128, 64, heads=4, concat=False, dropout=0.5)
        # self.fc = nn.Sequential(
        #     nn.Linear(64, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, dataset.num_classes)
        # )

    def forward(self, data):
        # x, edge_index, batch, edge_weight = data.x, data.edge_index, data.batch, data.edge_weight
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # x = F.relu(self.conv1(x, edge_index))
        # x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        # x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        #
        # x = F.relu(self.conv2(x, edge_index))
        # x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        # x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        #
        # x = F.relu(self.conv3(x, edge_index))
        # x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        # x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        #
        # x = x1 + x2 + x3
        # x = scatter_mean(x, data.batch, dim=0)
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn1(x)
        x, _ = scatter_max(x, data.batch, dim=0)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = self.lin3(x)
        # x = F.log_softmax(self.lin3(x), dim=-1)
        return x


torch.set_num_threads(1)
# criterion = nn.CrossEntropyLoss()
criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
ECE = ECELoss(n_bins=10)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = nn.DataParallel(Net, device_ids=[0, 2])
# model = model.cuda()
model = Net(args).to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                            momentum=args.momentum)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95, last_epoch=-1)


def train():
    if epoch == 10:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0005
    if epoch == 100:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0002
    if epoch == 200:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0001
    if epoch == 300:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 1e-5
    model.train()
    loss_all = 0
    best_loss = 100
    print('Epoch:{}'.format(epoch))
    for data in trainloader:
        # data = nn.DataParallel(data, device_ids=[0, 2])
        data = data.to(device)

        optimizer.zero_grad()
        out = model(data)
        # smooth_label = smooth_one_hot(data.y, classes=3, smoothing=0.1)
        # print(smooth_label)
        # train_loss = F.nll_loss(out, data.y)
        train_loss = criterion(out, data.y)
        loss_all += train_loss.item()
        train_loss.backward()
        optimizer.step()
        if train_loss.item() < best_loss:
            best_loss = train_loss.item()
            # best_label = data.y
    train_avg_loss = loss_all / len(trainloader)
    print("Training average loss: {:.4f}, Training best loss: {:.4f}".format(train_avg_loss, best_loss))
    # print(best_label)
    return train_avg_loss, best_loss


def test(loader):
    model.eval()
    correct = 0.
    loss = 0.
    topk = 0.
    # eces = 0.
    # avg_acc = 0.
    # avg_conf = 0.
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            test_labels = data.y
            out = model(data)
            # ece, conf, acc = ECE(out, data.y)
            # eces += ece
            # avg_conf += conf
            # avg_acc += acc
            # target = smooth_one_hot(data.y, classes=3, smoothing=0.1)
            pred = out.max(1)[1]
            _, maxk = torch.topk(out, k=2, dim=-1)
            test_labels = test_labels.view(-1, 1)
            topk += (test_labels == maxk).sum().item()
            correct += pred.eq(data.y).sum().item()

            # loss += F.nll_loss(out, data.y, reduction='sum').item()
            loss += criterion(out, data.y).item()
    # print(avg_conf / len(loader), avg_acc / len(loader))
    return correct / len(loader.dataset), topk / len(loader.dataset), loss / len(loader)


loss_list = []
acc_list = []
top2_acc_list = []
train_loss = []
best_loss = []
for epoch in range(1, args.epochs+1):
    start_time = time.time()
    training_loss, b_loss = train()
    train_loss.append(training_loss)
    best_loss.append(b_loss)
    val_acc, top2_acc, val_loss = test(valloader)
    lr = optimizer.param_groups[0]['lr']
    # lr = scheduler.optimizer.param_groups[0]['lr']
    # scheduler.step()
    loss_list.append(val_loss)
    acc_list.append(100. * val_acc)
    top2_acc_list.append(100. * top2_acc)
    end_time = time.time()
    print("Validation loss:{:.4f}\taccuracy:{:.4f}\ttop2_acc:{:.4f}\tlr:{:.3e}\tTime:{:.2f}".format(val_loss, val_acc, top2_acc,
                                                                                   lr, end_time-start_time))

sio.savemat('val_loss.mat', mdict={'val_loss': loss_list})
sio.savemat('val_acc.mat', mdict={'val_acc': acc_list})
sio.savemat('top2_acc.mat', mdict={'top2_acc': top2_acc_list})
sio.savemat('train_loss.mat', mdict={'train_loss': train_loss})
sio.savemat('best_loss.mat', mdict={'best_loss': best_loss})
x1 = range(1, args.epochs+1)
x2 = range(1, args.epochs+1)
y1 = acc_list
y2 = loss_list
plt.subplot(2, 1, 1)
plt.plot(x1, y1, '.-')
# plt.title('Test accuracy vs. epoches')
plt.xlabel('Epochs')
plt.ylabel('Test accuracy')
plt.subplot(2, 1, 2)
plt.plot(x2, y2, '.-')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig("accuracy_loss.jpg")
plt.show()

test_acc, top2_acc, test_loss = test(testloader)
print('Epoch: {:02d}, Test: {:.4f}, Top2_Acc: {:.4f}'.format(epoch, test_acc, top2_acc))
torch.save(model.state_dict(), 'zdsw_model.pkl')


# 加载保存的模型进行测试
# activation = {}
#
#
# def get_activation(name):
#     def hook(model, input, output):
#         # 如果你想feature的梯度能反向传播，那么去掉 detach（）
#         activation[name] = output.detach()
#     return hook
#
#
# model.load_state_dict(torch.load('./result/LS alpha=0.05/zdsw_model.pkl'))
# model.lin2.register_forward_hook(get_activation('lin2'))
# val_acc, top2_acc, val_loss = test(testloader)
# print("Validation loss:{:.4f}\taccuracy:{:.4f}\ttop2_acc:{:.4f}".format(val_loss, val_acc, top2_acc))
# print(activation['lin2'].size())
