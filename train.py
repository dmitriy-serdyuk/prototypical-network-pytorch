#!/usr/bin/env python
import argparse
import os.path as osp

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import Omniglot
from torchvision import transforms

from mini_imagenet import MiniImageNet
from samplers import CategoriesSampler, CategoriesSamplerMult
from convnet import Convnet
from utils import pprint, ensure_path, Averager, Timer, count_acc, euclidean_metric


class Data:
    def __init__(self, name, n_batches, train_ways, test_ways, shots, queries):
        self.name = name
        self.n_batches = n_batches
        self.train_ways = train_ways
        self.test_ways = test_ways
        self.valid_ways = test_ways
        self.shots = shots
        self.queries = queries

    @property
    def train(self):
        return self.klass('train')

    @property
    def valid(self):
        return self.klass('val')

    def get_loader(self, subset):
        if self.name == "miniimagenet":
            dataset = MiniImageNet(subset)
            labels = dataset.labels
        elif self.name == "omniglot":
            dataset = Omniglot(root="data/", download=True,
                               transform=transforms.ToTensor(),
                               background=subset == 'train')
            labels = list(map(lambda x: x[1], dataset._flat_character_images))
        else:
            raise ValueError
        sampler = CategoriesSamplerMult(
                labels,
                n_batches=self.n_batches if subset == 'train' else 400,
                ways=dict(train=self.train_ways, valid=self.valid_ways)[subset],
                n_images=self.shots + self.queries,
                n_combinations=2)
        return DataLoader(dataset=dataset, batch_sampler=sampler,
                          num_workers=8, pin_memory=True)

    @property
    def train_loader(self):
        return self.get_loader('train')

    @property
    def valid_loader(self):
        return self.get_loader('valid')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epoch', type=int, default=200)
    parser.add_argument('--save-epoch', type=int, default=20)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--train-way', type=int, default=30)
    parser.add_argument('--test-way', type=int, default=5)
    parser.add_argument('--save-path', default='./save/proto-1')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--n-batches', type=int, default=100)
    parser.add_argument('--no-permute', action='store_false', dest='permute')
    parser.add_argument('--dataset', choices=['miniimagenet', 'omniglot'], default='omniglot')
    #parser.add_argument('--sampler', choices=[CategoriesSampler, CategoriesSamplerMult],
    #        default=CategoriesSamplerMult,
    #        type=dict(base=CategoriesSampler, mult=CategoriesSamplerMult).__getitem__)
    args = parser.parse_args()
    pprint(vars(args))
    return args


def main(args):
    device = torch.device(args.device)
    ensure_path(args.save_path)

    data = Data(args.dataset, args.n_batches, args.train_way, args.test_way, args.shot, args.query)
    train_loader = data.train_loader
    val_loader = data.valid_loader

    model = Convnet(x_dim=2).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    def save_model(name):
        torch.save(model.state_dict(), osp.join(args.save_path, name + '.pth'))
    
    trlog = dict(
        args=vars(args),
        train_loss=[],
        val_loss=[],
        train_acc=[],
        val_acc=[],
        max_acc=0.0,
    )

    timer = Timer()

    for epoch in range(1, args.max_epoch + 1):
        lr_scheduler.step()

        model.train()

        tl = Averager()
        ta = Averager()

        for i, batch in enumerate(train_loader, 1):
            data, _ = [_.to(device) for _ in batch]
            data = data.reshape(-1, 2, 105, 105)
            p = args.shot * args.train_way
            embedded = model(data)
            embedded_shot, embedded_query = embedded[:p], embedded[p:]

            proto = embedded_shot.reshape(args.shot, args.train_way, -1).mean(dim=0)

            label = torch.arange(args.train_way).repeat(args.query).to(device)

            logits = euclidean_metric(embedded_query, proto)
            loss = F.cross_entropy(logits, label)
            acc = count_acc(logits, label)
            print('epoch {}, train {}/{}, loss={:.4f} acc={:.4f}'
                  .format(epoch, i, len(train_loader), loss.item(), acc))

            tl.add(loss.item())
            ta.add(acc)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        tl = tl.item()
        ta = ta.item()

        model.eval()

        vl = Averager()
        va = Averager()

        for i, batch in enumerate(val_loader, 1):
            data, _ = [_.cuda() for _ in batch]
            data = data.reshape(-1, 2, 105, 105)
            p = args.shot * args.test_way
            data_shot, data_query = data[:p], data[p:]

            proto = model(data_shot)
            proto = proto.reshape(args.shot, args.test_way, -1).mean(dim=0)

            label = torch.arange(args.test_way).repeat(args.query).to(device)

            logits = euclidean_metric(model(data_query), proto)
            loss = F.cross_entropy(logits, label)
            acc = count_acc(logits, label)

            vl.add(loss.item())
            va.add(acc)

        vl = vl.item()
        va = va.item()
        print('epoch {}, val, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))

        if va > trlog['max_acc']:
            trlog['max_acc'] = va
            save_model('max-acc')

        trlog['train_loss'].append(tl)
        trlog['train_acc'].append(ta)
        trlog['val_loss'].append(vl)
        trlog['val_acc'].append(va)

        torch.save(trlog, osp.join(args.save_path, 'trlog'))

        save_model('epoch-last')

        if epoch % args.save_epoch == 0:
            save_model('epoch-{}'.format(epoch))

        print('ETA:{}/{}'.format(timer.measure(), timer.measure(epoch / args.max_epoch)))


if __name__ == '__main__':
    args = parse_args()
    main(args)

