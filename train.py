import os
import os.path as osp
import math
import tqdm

import numpy as np
import torch
import torch.nn.functional as F

import znn
from time import strftime
from sklearn.metrics import confusion_matrix


def main():
    ckpt_dir = strftime('ckpts_znn_%b%d_%H%M%S')

    # n_epochs = 20
    # n_epochs = 20
    n_epochs = 400
    # n_epochs = 3

    batch_size = 16
    train_dataset, test_dataset, train_loader, test_loader = znn.get_loaders()

    epoch_size = len(train_dataset)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device', device)
    model = znn.DynamicReductionNetwork(hidden_dim=64, k=16).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=0.9, weight_decay=1e-3, nesterov=True)

    scheduler = znn.CyclicLRWithRestarts(optimizer, batch_size, epoch_size, restart_period=400, t_mult=1.1, policy="cosine")

    znn.print_model_summary(model)

    nsig = 0
    for data in tqdm.tqdm(train_loader, total=len(train_loader)):
        nsig += data.y.sum()
    s_over_n = float(nsig/len(train_dataset))
    print('sig/total=', s_over_n)
    loss_weights = torch.tensor([s_over_n, 1.-s_over_n]).to(device)
    # loss_weights = torch.tensor([1., 4.]).to(device)

    def write_checkpoint(checkpoint_number=None, best=False):
        ckpt = 'ckpt_best.pth.tar' if best else 'ckpt_{0}.pth.tar'.format(checkpoint_number)
        ckpt = osp.join(ckpt_dir, ckpt)
        os.makedirs(ckpt_dir, exist_ok=True)
        if best: print('Saving epoch {0} as new best'.format(checkpoint_number))
        torch.save(dict(model=model.state_dict()), ckpt)

    def train(epoch):
        print('Training epoch', epoch)
        model.train()
        scheduler.step()

        for data in tqdm.tqdm(train_loader, total=len(train_loader)):
            data = data.to(device)
            optimizer.zero_grad()
            result = model(data)
            log_probabilities = torch.nn.functional.log_softmax(result, dim=1)
            # pred = probabilities.argmax(1)
            # print('probabilities=', probabilities)
            # print('pred=', pred)
            # print('data.y=', data.y)
            # raise Exception

            # print('y =', data.y)
            # print('result =', result)
            # print('Sizes of y and result:', data.y.size(), result.size())

            loss = F.nll_loss(log_probabilities, data.y, weight=loss_weights)
            loss.backward()

            #print(torch.unique(torch.argmax(result, dim=-1)))
            #print(torch.unique(data.y))

            optimizer.step()
            scheduler.batch_step()

    def test():
        with torch.no_grad():
            model.eval()

            correct = 0
            pred = np.zeros(len(test_dataset), dtype=np.int8)
            truth = np.zeros(len(test_dataset), dtype=np.int8)

            for i, data in enumerate(test_loader):
                data = data.to(device)
                result = model(data)
                probabilities = torch.exp(torch.nn.functional.log_softmax(result, dim=1))
                predictions = torch.argmax(probabilities, dim=-1)
                # print('probabilities=', probabilities)
                # print('pred=', pred)
                # print('data.y=', data.y)

                correct += predictions.eq(data.y).sum().item()
                pred[i*batch_size:(i+1)*batch_size] = predictions.cpu()
                truth[i*batch_size:(i+1)*batch_size] = data.y.cpu()

            print(confusion_matrix(truth, pred, labels=[0,1]))
            acc = correct / len(test_dataset)
            print(
                'Epoch: {:02d}, Test acc: {:.4f}'
                .format(epoch, acc)
                )
            return acc


    test_accs = []

    best_test_acc = 0.0
    for epoch in range(1, 1+n_epochs):
        train(epoch)
        test_acc = test()
        write_checkpoint(epoch)
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            write_checkpoint(epoch, best=True)
        test_accs.append(test_acc)

    np.savez(osp.join(ckpt_dir, 'testaccs.npz'), testaccs=test_accs)

if __name__ == '__main__':
    main()

