# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import errno
import torch
import sys
import logging
import json
from pathlib import Path
import torch.distributed as dist
import csv
import os.path as osp
import time
import re
from transformers import AdamW

logger = logging.getLogger(__name__)


def init_logger(is_main=True, is_distributed=False, filename=None):
    if is_distributed:
        torch.distributed.barrier()
    handlers = [logging.StreamHandler(sys.stdout)]
    if filename is not None:
        handlers.append(logging.FileHandler(filename=filename))
    logging.basicConfig(
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main else logging.WARN,
        format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
        handlers=handlers,
    )

    # get running command
    command = ["python", sys.argv[0]]
    for x in sys.argv[1:]:
        if x.startswith('--'):
            assert '"' not in x and "'" not in x
            command.append(x)
        else:
            assert "'" not in x
            if re.match('^[a-zA-Z0-9_]+$', x):
                command.append("%s" % x)
            else:
                command.append("'%s'" % x)
    command = ' '.join(command)

    logging.getLogger('transformers.tokenization_utils').setLevel(logging.ERROR)
    logging.getLogger('transformers.tokenization_utils_base').setLevel(logging.ERROR)
    logger.info("Running command: %s" % command)
    return logger

# def get_checkpoint_path(opt):
#     checkpoint_path = Path(opt.checkpoint_dir) / opt.name
#     checkpoint_exists = checkpoint_path.exists()
#     if opt.is_distributed:
#         torch.distributed.barrier()
#     checkpoint_path.mkdir(parents=True, exist_ok=True)
#     return checkpoint_path, checkpoint_exists


def symlink_force(target, link_name):
    try:
        os.symlink(target, link_name)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(link_name)
            os.symlink(target, link_name)
        else:
            raise e


def my_load_model(opt, model, dataroot, model_path):
    # TODO: path

    save_path = osp.join(dataroot, 'model', model_path)
    state_dict = torch.load(save_path, map_location='cpu')

    model.load_state_dict(state_dict)
    print(f"loading model from {model_path},  done!")
    return model


def my_save_model(opt, model, dataroot, model_path=None):
    # now_time = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time()))
    now_time = time.strftime("%m-%d-%H", time.localtime(time.time()))

    model_root = osp.join(dataroot, 'model')
    if model_path is None:
        model_path = osp.join(model_root, f"{opt.dataset}_{opt.model_size}.pth")
    if not osp.exists(model_root):
        os.mkdir(model_root)

    # model_name = type(model).__name__
    print(f'save model to: {model_path} ...')
    save_path = model_path
    torch.save(model.state_dict(), save_path)
    # print(f"saving model done!")
    return save_path


def save(model, optimizer, scheduler, step, best_eval_metric, opt, dir_path, name):
    model_to_save = model.module if hasattr(model, "module") else model
    path = os.path.join(dir_path, "checkpoint")
    epoch_path = os.path.join(path, name)  # "step-%s" % step)
    os.makedirs(epoch_path, exist_ok=True)
    model_to_save.save_pretrained(epoch_path)
    cp = os.path.join(path, "latest")
    fp = os.path.join(epoch_path, "optimizer.pth.tar")
    checkpoint = {
        "step": step,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "opt": opt,
        "best_eval_metric": best_eval_metric,
    }
    torch.save(checkpoint, fp)
    symlink_force(epoch_path, cp)


def load(model_class, dir_path, opt, reset_params=False):
    epoch_path = os.path.realpath(dir_path)
    optimizer_path = os.path.join(epoch_path, "optimizer.pth.tar")
    logger.info("Loading %s" % epoch_path)
    model = model_class.from_pretrained(epoch_path)
    # model = model.to(opt.device)
    model = model.cuda()
    logger.info("loading checkpoint %s" % optimizer_path)
    checkpoint = torch.load(optimizer_path, map_location=f"cuda:{str(opt.gpu)}")
    opt_checkpoint = checkpoint["opt"]
    step = checkpoint["step"]
    if "best_eval_metric" in checkpoint:
        best_eval_metric = checkpoint["best_eval_metric"]
    else:
        best_eval_metric = checkpoint["best_dev_em"]
    if not reset_params:
        optimizer, scheduler = set_optim(opt_checkpoint, model)
        scheduler.load_state_dict(checkpoint["scheduler"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        optimizer, scheduler = set_optim(opt, model)

    return model, optimizer, scheduler, opt_checkpoint, step, best_eval_metric


class WarmupLinearScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup_steps, scheduler_steps, min_ratio, fixed_lr, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.scheduler_steps = scheduler_steps
        self.min_ratio = min_ratio
        self.fixed_lr = fixed_lr
        super(WarmupLinearScheduler, self).__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch
        )

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return (1 - self.min_ratio) * step / float(max(1, self.warmup_steps)) + self.min_ratio

        if self.fixed_lr:
            return 1.0

        return max(0.0,
                   1.0 + (self.min_ratio - 1) * (step - self.warmup_steps) / float(max(1.0, self.scheduler_steps - self.warmup_steps)),
                   )


class FixedScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, last_epoch=-1):
        super(FixedScheduler, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        return 1.0


def set_dropout(model, dropout_rate):
    for mod in model.modules():
        if isinstance(mod, torch.nn.Dropout):
            mod.p = dropout_rate


def layerwise_decay_optimizer(model, lr, wd, layerwise_decay=None):

    # optimizer and lr scheduler
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': wd},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': wd}
    ]

    if layerwise_decay is not None:
        optimizer_grouped_parameters = []
        for i in range(12):
            tmp = [{'params': [p for n, p in model.named_parameters()
                               if 'bert.encoder.layer.' + str(i) + '.' in n
                               and not any(nd in n for nd in no_decay)],
                    'lr': lr * (layerwise_decay ** (7 - i)),
                    'weight_decay': wd},

                   {'params': [p for n, p in model.named_parameters()
                               if 'bert.encoder.layer.' + str(i) + '.' in n
                               and any(nd in n for nd in no_decay)],
                    'lr': lr * (layerwise_decay ** (7 - i)),
                    'weight_decay': 0}
                   ]
            optimizer_grouped_parameters += tmp

        tmp = [{'params': [p for n, p in model.named_parameters()
                           if 'bert.encoder.layer.' not in n
                           and not any(nd in n for nd in no_decay)],
                'weight_decay': wd},

               {'params': [p for n, p in model.named_parameters()
                           if 'bert.encoder.layer.' not in n
                           and any(nd in n for nd in no_decay)],
                'weight_decay': 0}
               ]
        optimizer_grouped_parameters += tmp

    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, correct_bias=False)

    return optimizer


def set_optim(opt, model):
    if opt.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    elif opt.optim == 'adamw':
        # optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
        optimizer = layerwise_decay_optimizer(model, lr=opt.lr, wd=opt.weight_decay)
    if opt.scheduler == 'fixed':
        scheduler = FixedScheduler(optimizer)
    elif opt.scheduler == 'linear':
        if opt.scheduler_steps is None:
            scheduler_steps = opt.total_steps
            print("use total step...")
        else:
            scheduler_steps = opt.scheduler_steps
        scheduler = WarmupLinearScheduler(optimizer, warmup_steps=opt.warmup_steps, scheduler_steps=scheduler_steps, min_ratio=0., fixed_lr=opt.fixed_lr)
    return optimizer, scheduler


def average_main(x, opt):
    if not opt.is_distributed:
        return x
    if opt.world_size > 1:
        dist.reduce(x, 0, op=dist.ReduceOp.SUM)
        if opt.is_main:
            x = x / opt.world_size
    return x


def sum_main(x, opt):
    if not opt.is_distributed:
        return x
    if opt.world_size > 1:
        dist.reduce(x, 0, op=dist.ReduceOp.SUM)
    return x


def weighted_average(x, count, opt):
    if not opt.is_distributed:
        return x, count
    t_loss = torch.tensor([x * count], device=opt.device)
    t_total = torch.tensor([count], device=opt.device)
    t_loss = sum_main(t_loss, opt)
    t_total = sum_main(t_total, opt)
    return (t_loss / t_total).item(), t_total.item()


def write_output(glob_path, output_path):
    files = list(glob_path.glob('*.txt'))
    files.sort()
    with open(output_path, 'w') as outfile:
        for path in files:
            with open(path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    outfile.write(line)
            path.unlink()
    glob_path.rmdir()


# def save_distributed_dataset(data, opt):
#     dir_path = Path(opt.checkpoint_dir) / opt.name
#     write_path = dir_path / 'tmp_dir'
#     write_path.mkdir(exist_ok=True)
#     tmp_path = write_path / f'{opt.global_rank}.json'
#     with open(tmp_path, 'w') as fw:
#         json.dump(data, fw)
#     if opt.is_distributed:
#         torch.distributed.barrier()
#     if opt.is_main:
#         final_path = dir_path / 'dataset_wscores.json'
#         logger.info(f'Writing dataset with scores at {final_path}')
#         glob_path = write_path / '*'
#         results_path = write_path.glob('*.json')
#         alldata = []
#         for path in results_path:
#             with open(path, 'r') as f:
#                 data = json.load(f)
#             alldata.extend(data)
#             path.unlink()
#         with open(final_path, 'w') as fout:
#             json.dump(alldata, fout, indent=4)
#         write_path.rmdir()

def load_passages(path):
    if not os.path.exists(path):
        logger.info(f'{path} does not exist')
        return
    logger.info(f'Loading passages from: {path}')
    passages = []
    with open(path) as fin:
        reader = csv.reader(fin, delimiter='\t')
        for k, row in enumerate(reader):
            if not row[0] == 'id':
                try:
                    passages.append((row[0], row[1], row[2]))
                except:
                    logger.warning(f'The following input line has not been correctly loaded: {row}')
    return passages
