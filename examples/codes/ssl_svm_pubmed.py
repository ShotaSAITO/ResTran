import argparse
import os.path as osp

import torch
import torch.nn.functional as F
from copy import copy

import datasets

import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='pubmed')
parser.add_argument('--dataset_path', type=str, default='./data')
parser.add_argument('--hidden_channels', type=int, default=16)
parser.add_argument('--train_mask', type=float, default=0.05)
parser.add_argument('--val_mask', type=float, default=0.25)
parser.add_argument('--dropout_rate', type=float, default=0.6)
parser.add_argument('--heads', type=int, default=8)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--wandb', action='store_true', help='Track experiment')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
init_wandb(name=f'GAT-{args.dataset}', heads=args.heads, epochs=args.epochs,
           hidden_channels=args.hidden_channels, lr=args.lr, device=device)

gammas = [10 ** -i for i in range(10)]


test_accs = []

test_accs = []

def generate_masks_v2(indices, y, train_mask=0.05, val_mask=0.05, test_mask=0.9):
    from sklearn.model_selection import train_test_split
    n_labels = int(max(y) + 1)
    indices_full = copy(indices)
    indices_full = np.array(indices_full)
    indices_train = []
    indices_test = []
    indices_val = []

    indices_train_tmp = []

    while np.unique(y[indices_train_tmp]).size != n_labels:
        #pLsq_train, pLsq_unlabelled, Y_train, Y_unlabelled = train_test_split(pLsq, Y, train_size=n_train)
        indices_train_tmp, indices_test_valid_tmp = train_test_split(indices, train_size=train_mask)

    indices_test_tmp, indices_val_tmp = train_test_split(indices_test_valid_tmp,
                                                         train_size=test_mask / (val_mask + test_mask))

    indices_train += list(indices_train_tmp)
    indices_test += list(indices_test_tmp)
    indices_val += list(indices_val_tmp)

    train_mask = torch.Tensor([False for i in range(len(indices_full))])
    train_mask[indices_train] = True
    train_mask = train_mask.to(torch.bool)

    test_mask = torch.Tensor([False for i in range(len(indices_full))])
    test_mask[indices_test] = True
    test_mask = test_mask.to(torch.bool)

    val_mask = torch.Tensor([False for i in range(len(indices_full))])
    val_mask[indices_val] = True
    val_mask = val_mask.to(torch.bool)

    # return indices_train, indices_test, indices_val

    return train_mask, test_mask, val_mask


for _ in range(10):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
    LX, X, Y = datasets.load_data_krylov(args.dataset, args.dataset_path)
    indices = [i for i in range(Y.shape[0])]
    train_mask, test_mask, val_mask = generate_masks_v2(indices, Y, train_mask=args.train_mask,
                                                        val_mask=args.val_mask, test_mask=1-(args.train_mask + args.val_mask))


    @torch.no_grad()
    def test():
        pred = model.predict(X)

        accs = []
        #for mask in [data.train_mask, data.val_mask, data.test_mask]:
        for mask in [train_mask, val_mask, test_mask]:
            accs.append(int((pred[mask] == Y[mask]).sum()) / int(mask.sum()))
        return accs

    #model = make_pipeline(StandardScaler(), OneVsRestClassifier(SVC(gamma='scale')))
    best_val_acc = final_test_acc = 0
    for gamma in gammas:
        model = make_pipeline(OneVsRestClassifier(SVC(gamma=gamma)))
        model.fit(LX[train_mask, :], Y[train_mask])

        (train_acc, val_acc, test_acc) = test()
        log(Gamma=gamma, Train=train_acc, Val=val_acc, Test=test_acc)
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            final_test_acc = test_acc

    test_accs.append(final_test_acc)
    '''
    for epoch in range(1, args.epochs + 1):
        loss = train()
        train_acc, val_acc, tmp_test_acc = test()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
        log(Epoch=epoch, Loss=loss, Train=train_acc, Val=val_acc, Test=test_acc)

    test_accs.append(test_acc)
    '''
print('{0}:{1:2f}=-{2:2f}'.format(args.dataset, sum(test_accs) / 10, np.std(test_accs)))
for test_acc in test_accs:
    print(test_acc)