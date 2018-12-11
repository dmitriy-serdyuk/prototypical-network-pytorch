#!/usr/bin/env python
import argparse
import os.path as osp

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mini_imagenet import MiniImageNet
from samplers import CategoriesSampler
from convnet import Convnet
from utils import pprint, ensure_path, Averager, Timer, count_acc, euclidean_metric


if __name__ == '__main__':
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
    args = parser.parse_args()
    pprint(vars(args))

    device = torch.device(args.device)
    ensure_path(args.save_path)

    trainset = MiniImageNet('train')
    train_sampler = CategoriesSampler(trainset.label,
                                      n_batches=args.n_batches,
                                      ways=args.train_way,
                                      n_images=args.shot + args.query,
                                      permute=args.permute)
    train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler,
                              num_workers=8, pin_memory=True)

    valset = MiniImageNet('val')
    val_sampler = CategoriesSampler(valset.label, 400,
                                    args.test_way, args.shot + args.query,
                                    permute=args.permute)
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler,
                            num_workers=8, pin_memory=True)

    model = Convnet().to(device)
    
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
            p = args.shot * args.test_way
            data_shot, data_query = data[:p], data[p:]

            proto = model(data_shot)
            proto = proto.reshape(args.shot, args.test_way, -1).mean(dim=0)

            label = torch.arange(args.test_way).repeat(args.query)
            label = label.type(torch.cuda.LongTensor)

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
