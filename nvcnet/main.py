# Copyright 2021 Sony Group Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
from pathlib import Path

import nnabla as nn
import numpy as np
from nnabla.ext_utils import get_extension_context
from nnabla.logger import logger
from nnabla.utils.data_iterator import data_iterator
from nnabla_ext.cuda import StreamEventHandler

from dataset import VCTKDataSource
from hparams import hparams as hp
from model.model import Discriminator, NVCNet
from train import Trainer
from utils.misc import CommunicatorWrapper
from utils.optim import Optimizer


def run(args):
    # create output path
    Path(hp.output_path).mkdir(parents=True, exist_ok=True)

    # setup nnabla context
    ctx = get_extension_context(args.context, device_id='0')
    nn.set_default_context(ctx)

    hp.comm = CommunicatorWrapper(ctx)
    hp.event = StreamEventHandler(int(hp.comm.ctx.device_id))

    with open(hp.speaker_dir) as f:
        hp.n_speakers = len(f.read().split('\n'))
        logger.info(f'Training data with {hp.n_speakers} speakers.')

    if hp.comm.n_procs > 1 and hp.comm.rank == 0:
        n_procs = hp.comm.n_procs
        logger.info(f'Distributed training with {n_procs} processes.')
    rng = np.random.RandomState(hp.seed)
    train_loader = data_iterator(
        VCTKDataSource('metadata_train.csv', hp, shuffle=True, rng=rng),
        batch_size=hp.batch_size, with_memory_cache=False, rng=rng
    )
    dataloader = dict(train=train_loader, valid=None)
    gen = NVCNet(hp)
    gen_optim = Optimizer(
        weight_decay=hp.weight_decay,
        name='Adam', alpha=hp.g_lr,
        beta1=hp.beta1, beta2=hp.beta2
    )
    dis = Discriminator(hp)
    dis_optim = Optimizer(
        weight_decay=hp.weight_decay,
        name='Adam', alpha=hp.d_lr,
        beta1=hp.beta1, beta2=hp.beta2
    )
    Trainer(gen, gen_optim, dis, dis_optim, dataloader, rng, hp).run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--context', '-c', type=str, default='cudnn',
        help="Extension module. 'cudnn' is highly recommended.")
    parser.add_argument("--device-id", "-d", type=str, default='-1',
                        help='A list of device ids to use.\
                        This is only valid if you specify `-c cudnn`.\
                        Defaults to use all available GPUs.')

    for key, value in hp.__dict__.items():
        name = "--" + key
        if type(value) == list:
            nargs, t = '+', type(value[0])
        else:
            nargs, t = None, type(value)
        parser.add_argument(name, type=t, nargs=nargs, default=value)

    args = parser.parse_args()
    for k, v in vars(args).items():
        hp.__dict__[k] = v

    # setup context for nnabla
    if args.device_id != '-1':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    run(args)
