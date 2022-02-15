import os
import copy
import pytorch_lightning as pl

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.models as models
import colossalai
import colossalai.nn as col_nn
from colossalai.core import global_context as gpc
from colossalai.logging import get_dist_logger
import argparse
from colossalai.trainer import Trainer, hooks
from colossalai.nn.lr_scheduler import LinearWarmupLR
from torch.nn.modules.loss import *
from torch.nn.modules.loss import _Loss

from colossalai.nn.layer.utils import get_tensor_parallel_mode 

from utils.config import ex
from models.vilt import ViLT
from utils.dataloader import MTDataModule
from schedule import viltSchedule
from torchvision.transforms.functional import resize

class MLM_loss(_Loss):
    def __init__(self, reduction: bool = True, *args, **kwargs):
        super().__init__()
    def itm_mlm_loss(self,output):
        total_loss = sum([v for k, v in output.items() if "loss" in k])
        return total_loss
    def forward(self, *args):
        return self.itm_mlm_loss(*args)

NUM_CHUNKS = 2

@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    config = './configs.py'
    gpc.load_config(config)
    print('config loaded')
    if 'OMPI_COMM_WORLD_RANK' in os.environ:
        colossalai.launch_from_openmpi(config=config,
        host='localhost',
        port='11455',
        backend='nccl')
    elif 'SLURM_PROCID' in os.environ:
        colossalai.launch_from_slurm(config=config,
        host='localhost',
        port='11455',
        backend='nccl')
    elif 'WORLD_SIZE' in os.environ:
        colossalai.launch_from_torch(config=config,
        host='localhost',
        port='11455',
        backend='nccl')
    else:
        colossalai.launch(
            config=config,
            host='localhost',
            port='11455',
            rank=0,
            world_size=1,
            backend='nccl')
    logger = get_dist_logger('root')
    logger.info('launched')
    print('launched')
    dm = MTDataModule(_config, dist=False)
    dm.prepare_data()
    dm.setup(0)
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    model = ViLT(_config)


    #Initialize engine and trainer

    optim = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        betas=(0.9, 0.99)
    )

   
    def batch_data_process_func(sample):
        data = dict()
        img = sample['image'][0]
        img = resize(img, (384, 384))
        data['image'] = img
        data['text_ids_mlm'] = sample['text_ids_mlm']
        data['text_labels_mlm'] = sample['text_labels_mlm']
        return data,dict()
    schedule=viltSchedule(batch_data_process_func)
    lr_scheduler = LinearWarmupLR(optim, warmup_steps=50, total_steps=gpc.config.NUM_EPOCHS)
    criterion=MLM_loss()

    engine, train_dataloader, test_dataloader, lr_scheduler = colossalai.initialize(model=model,
    optimizer=optim,
    criterion=criterion,
    train_dataloader=train_loader,
    test_dataloader=val_loader,
    verbose=True,)

    logger.info("engine is built", ranks=[0])

    trainer = Trainer(engine=engine,
            schedule=schedule, logger=logger)
    
    hook_list = [
        hooks.LossHook(),
        hooks.LogMetricByStepHook(),
        hooks.SaveCheckpointHook(checkpoint_dir='./ckpt'),
        hooks.TensorboardHook(log_dir='./logs/')
    ]
    
    # hook_list = [
    #     hooks.LossHook(),
    #     hooks.AccuracyHook(),
    #     hooks.LogMetricByEpochHook(logger),
    #     hooks.LRSchedulerHook(lr_scheduler, by_epoch=True),

    #     # comment if you do not need to use the hooks below
    #     hooks.SaveCheckpointHook(interval=1, checkpoint_dir='./ckpt'),
    #     hooks.TensorboardHook(log_dir='./tb_logs', ranks=[0]),
    # ]

    logger.info("trainer is built", ranks=[0])



    colossalai.context.config.Config.from_file(config)

    logger.info("start training", ranks=[0])
    trainer.fit(
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        hooks=hook_list,
        epochs=gpc.config.num_epochs,
        display_progress=True,
        test_interval=2
    )

class MLMLossHook(hooks.LossHook()):
    def itm_mlm_loss(self,output):
        total_loss = sum([v for k, v in output.items() if "loss" in k])
        return total_loss
    def after_train_iter(self, trainer, logits, label, loss):
        if self._is_stage_to_compute:
            self.train_loss.update(self.itm_mlm_loss(logits))
    def after_test_iter(self, trainer, logits, label, loss):
        if self._is_stage_to_compute:
            self.test_loss.update(self.itm_mlm_loss(logits))


