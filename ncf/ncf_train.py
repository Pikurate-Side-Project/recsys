import argparse
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torch_optimizer as custom_optim
from modules.FeedForwardNN import FeedForwardNN

from utils.trainer import NCFTrainer
from utils.dataset import NCFDataset, NCFCollator
from utils.utils import read_text


def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True)
    p.add_argument('--train_fn', required=True)
    
    p.add_argument('--gpu_id', type=int, default=-1)
    p.add_argument('--verbose', type=int, default=2)

    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--n_epochs', type=int, default=5)

    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--adam_epsilon', type=float, default=1e-8)
    # If you want to use RAdam, I recommend to use LR=1e-4.
    # Also, you can set warmup_ratio=0.
    p.add_argument('--use_radam', action='store_true')
    p.add_argument('--valid_ratio', type=float, default=.2)

    config = p.parse_args()

    return config


def get_loaders(fn, valid_ratio=.2):
    # Get list of labels and list of texts.
    users, urls, labels = read_text(fn)

    # Generate label to index map.
    unique_users = list(set(users))
    unique_urls = list(set(urls))

    # Shuffle before split into train and validation set.
    shuffled = list(zip(users, urls, labels))
    random.shuffle(shuffled)
    users = [e[0] for e in shuffled]
    urls = [e[1] for e in shuffled]
    labels = [e[2] for e in shuffled]
    idx = int(len(labels) * (1 - valid_ratio))

    # Get dataloaders using given tokenizer as collate_fn.
    train_loader = DataLoader(
        NCFDataset(users[:idx], urls[:idx], labels[:idx]),
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=NCFCollator(),
    )
    valid_loader = DataLoader(
        NCFDataset(users[idx:], urls[idx:], labels[idx:]),
        batch_size=config.batch_size,
        collate_fn=NCFCollator(),
    )

    return train_loader, valid_loader, len(unique_users), len(unique_urls)


def get_optimizer(model, config):
    if config.use_radam:
        optimizer = custom_optim.RAdam(model.parameters(), lr=config.lr)
    else:
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': 0.01
            },
            {
                'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]

        optimizer = optim.AdamW(
            optimizer_grouped_parameters,
            lr=config.lr,
            eps=config.adam_epsilon
        )

    return optimizer


def main(config):
    # Get pretrained tokenizer.
    train_loader, valid_loader, index_to_label, num_users, num_urls = get_loaders(
        config.train_fn,
        valid_ratio=config.valid_ratio
    )

    print(
        '|train| =', len(train_loader) * config.batch_size,
        '|valid| =', len(valid_loader) * config.batch_size,
    )

    n_total_iterations = len(train_loader) * config.n_epochs
    print(
        '#total_iters =', n_total_iterations
    )

    # Get pretrained model with specified softmax layer.
    model = FeedForwardNN(
        n_users=num_users, n_urls=num_urls, 
        n_factors=16, hidden=[64, 32, 16], 
        embedding_dropout=0.05, dropouts=[0.3, 0.3, 0.3]
    )
    optimizer = get_optimizer(model, config)
    crit = nn.BCELoss()

    if config.gpu_id >= 0:
        model.cuda(config.gpu_id)
        crit.cuda(config.gpu_id)

    trainer = NCFTrainer(config)
    model = trainer.train(
        model,
        crit,
        optimizer,
        train_loader,
        valid_loader
    )

    torch.save({
        'config': config,
        'classes': index_to_label,
    }, config.model_fn)

if __name__ == '__main__':
    config = define_argparser()
    main(config)