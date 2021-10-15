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

import json
import os
from pathlib import Path

import nnabla as nn
import soundfile as sf

from utils.audio import random_flip, random_jitter, random_scaling
from utils.logger import ProgressMeter
from utils.misc import set_persistent_all


class Trainer:
    r"""Trainer is a basic class for training a model."""

    def __init__(self, gen, gen_optim, dis, dis_optim, dataloader, rng, hp):
        self.gen = gen
        self.gen_optim = gen_optim
        self.dis = dis
        self.dis_optim = dis_optim

        self.dataloader = dataloader
        self.rng = rng
        self.hp = hp
        self.one_epoch_train = dataloader['train'].size // hp.batch_size

        self.placeholder = dict()
        self.monitor = ProgressMeter(
            self.one_epoch_train, hp.output_path,
            quiet=hp.comm.rank > 0
        )
        hp.save(os.path.join(hp.output_path, 'settings.json'))

    def update_graph(self, key='train'):
        r"""Builds the graph and update the placeholder.

        Args:
            training (bool, optional): Type of the graph. Defaults to `train`.
        """
        assert key in ('train', 'valid')

        self.gen.training = key == 'train'
        self.dis.training = key == 'train'
        hp = self.hp

        def data_aug(v):
            v = random_flip(v)
            v = random_scaling(v, hp.scale_low, hp.scale_high)
            return v

        # define input variables
        input_x = nn.Variable((hp.batch_size, 1, hp.segment_length))
        input_y = nn.Variable((hp.batch_size, 1, hp.segment_length))
        label_x = nn.Variable((hp.batch_size, 1))
        label_y = nn.Variable((hp.batch_size, 1))

        x_aug = data_aug(input_x)
        r_jitter_x = random_jitter(x_aug, hp.max_jitter_steps)

        x_real_con = self.gen.encode(x_aug)
        s_real, s_mu, s_logvar = self.gen.embed(data_aug(input_x))
        x_real = self.gen.decode(x_real_con, s_real)

        r_fake = self.gen.embed(data_aug(input_y))[0]
        x_fake = self.gen.decode(x_real_con, r_fake)
        x_fake_con = self.gen.encode(random_flip(x_fake))

        dis_real_x = self.dis(data_aug(input_x), label_x)
        dis_fake_x = self.dis(data_aug(x_fake), label_y)

        # ------------------------------ Discriminator -----------------------
        d_loss = (self.dis.adversarial_loss(dis_real_x, 1.0)
                  + self.dis.adversarial_loss(dis_fake_x, 0.0))
        # --------------------------------------------------------------------

        # -------------------------------- Generator -------------------------
        g_loss_avd = self.dis.adversarial_loss(self.dis(x_fake, label_y), 1.0)
        g_loss_con = self.dis.preservation_loss(x_fake_con, x_real_con)
        g_loss_kld = self.gen.kl_loss(s_mu, s_logvar)
        g_loss_rec = (
            self.dis.perceptual_loss(x_real, r_jitter_x)
            + self.dis.spectral_loss(x_real, r_jitter_x)
        )
        g_loss = (
            g_loss_avd
            + hp.lambda_con * g_loss_con
            + hp.lambda_rec * g_loss_rec
            + hp.lambda_kld * g_loss_kld
        )

        # -------------------------------------------------------------------
        set_persistent_all(
            g_loss_con, g_loss_avd, g_loss,
            d_loss, x_fake, g_loss_kld, g_loss_rec
        )

        self.placeholder[key] = dict(
            input_x=input_x, label_x=label_x,
            input_y=input_y, label_y=label_y,
            x_fake=x_fake, d_loss=d_loss,
            g_loss_avd=g_loss_avd, g_loss_con=g_loss_con,
            g_loss_rec=g_loss_rec, g_loss_kld=g_loss_kld,
            g_loss=g_loss,
        )

    def callback_on_start(self):
        self.cur_epoch = 0
        checkpoint = Path(self.hp.output_path) / 'checkpoint.json'
        if checkpoint.is_file():
            self.load_checkpoint_model(str(checkpoint))

        self.update_graph('train')
        params = self.gen.get_parameters(grad_only=True)
        self.gen_optim.set_parameters(params)

        dis_params = self.dis.get_parameters(grad_only=True)
        self.dis_optim.set_parameters(dis_params)

        if checkpoint.is_file():
            self.load_checkpoint_optim(str(checkpoint))

        self._grads = [x.grad for x in params.values()]
        self._discs = [x.grad for x in dis_params.values()]

        self.log_variables = [
            'g_loss_avd', 'g_loss_con',
            'g_loss_rec', 'g_loss_kld',
            'd_loss',
        ]

    def run(self):
        r"""Run the training process."""
        self.callback_on_start()

        for cur_epoch in range(self.cur_epoch + 1, self.hp.epoch + 1):
            self.monitor.reset()
            lr = self.gen_optim.get_learning_rate()
            self.monitor.info(f'Running epoch={cur_epoch}\tlr={lr:.5f}\n')
            self.cur_epoch = cur_epoch

            for i in range(self.one_epoch_train):
                self.train_on_batch(i)
                if i % (self.hp.print_frequency) == 0:
                    self.monitor.display(i, self.log_variables)

            self.callback_on_epoch_end()

        self.callback_on_finish()
        self.monitor.close()

    def _zero_grads(self):
        self.gen_optim.zero_grad()
        self.dis_optim.zero_grad()

    def getdata(self, key='train'):
        data, label = self.dataloader[key].next()
        idx = self.rng.permutation(self.hp.batch_size)
        return data, label, data[idx], label[idx]

    def train_on_batch(self, i):
        r"""Updates the model parameters."""
        hp = self.hp
        bs, p = hp.batch_size, self.placeholder['train']
        p['input_x'].d, p['label_x'].d, p['input_y'].d, p['label_y'].d = \
            self.getdata('train')

        # ----------------------------- train discriminator ------------------
        if i % hp.n_D_updates == 0:
            self._zero_grads()
            p['x_fake'].need_grad = False
            p['d_loss'].forward()
            p['d_loss'].backward(clear_buffer=True)
            self.monitor.update('d_loss', p['d_loss'].d.copy(), bs)
            hp.comm.all_reduce(self._discs, division=True, inplace=False)
            self.dis_optim.update()
            p['x_fake'].need_grad = True
        # ---------------------------------------------------------------------

        # ------------------------------ train generator ----------------------
        self._zero_grads()
        p['g_loss'].forward()
        p['g_loss'].backward(clear_buffer=True)
        self.monitor.update('g_loss', p['g_loss'].d.copy(), bs)
        self.monitor.update('g_loss_avd', p['g_loss_avd'].d.copy(), bs)
        self.monitor.update('g_loss_con', p['g_loss_con'].d.copy(), bs)
        self.monitor.update('g_loss_rec', p['g_loss_rec'].d.copy(), bs)
        self.monitor.update('g_loss_kld', p['g_loss_kld'].d.copy(), bs)
        hp.comm.all_reduce(self._grads, division=True, inplace=False)
        self.gen_optim.update()
        # -------------------------------------------------------------------------

    def callback_on_epoch_end(self):
        hp = self.hp
        if (hp.comm.rank == 0):
            path = Path(hp.output_path) / 'artifacts'
            path.joinpath('states').mkdir(parents=True, exist_ok=True)
            path.joinpath('samples').mkdir(parents=True, exist_ok=True)
            self.save_checkpoint(path / 'states')
            self.write_samples(path / 'samples')
            if self.cur_epoch % hp.epochs_per_checkpoint == 0:
                path = path / f"epoch_{self.cur_epoch}"
                path.mkdir(parents=True, exist_ok=True)
                self.gen.save_parameters(str(path / 'model.h5'))
                self.dis.save_parameters(str(path / 'cls.h5'))

    def callback_on_finish(self):
        if self.hp.comm.rank == 0:
            path = Path(self.hp.output_path)
            self.gen.save_parameters(str(path / 'model.h5'))
            self.dis.save_parameters(str(path / 'cls.h5'))

    def save_checkpoint(self, path):
        r"""Save the current states of the trainer."""
        if self.hp.comm.rank == 0:
            path = Path(path)
            self.gen.save_parameters(str(path / 'model.h5'))
            self.dis.save_parameters(str(path / 'cls.h5'))
            self.gen_optim.save_states(str(path / 'gen_optim.h5'))
            self.dis_optim.save_states(str(path / 'dis_optim.h5'))
            with open(Path(self.hp.output_path) / 'checkpoint.json', 'w') as f:
                json.dump(
                    dict(cur_epoch=self.cur_epoch,
                         params_path=str(path),
                         gen_optim_n_iters=self.gen_optim._iter,
                         dis_optim_n_iters=self.dis_optim._iter,),
                    f
                )
            self.monitor.info(f"Checkpoint saved: {str(path)}\n")

    def load_checkpoint_model(self, checkpoint):
        r"""Load the last states of the trainer."""
        with open(checkpoint, 'r') as file:
            info = json.load(file)
            path = Path(info['params_path'])
        self.gen.load_parameters(str(path / 'model.h5'), raise_if_missing=True)
        self.dis.load_parameters(str(path / 'cls.h5'), raise_if_missing=True)

    def load_checkpoint_optim(self, checkpoint):
        r"""Load the last states of the trainer."""
        with open(checkpoint, 'r') as file:
            info = json.load(file)
            path = Path(info['params_path'])
            self.gen_optim._iter = info['gen_optim_n_iters']
            self.dis_optim._iter = info['dis_optim_n_iters']
            self.cur_epoch = info['cur_epoch']
        self.gen_optim.load_states(str(path / 'gen_optim.h5'))
        self.dis_optim.load_states(str(path / 'dis_optim.h5'))

    def write_samples(self, path):
        r"""write a few samples."""
        hp = self.hp
        p = self.placeholder['train']
        X, Z = p['input_x'].d.copy(), p['x_fake'].d.copy()
        for i, (x, z) in enumerate(zip(X, Z)):
            sf.write(str(path / f'input_{i}.wav'), x[0], hp.sr)
            sf.write(str(path / f'convert_{i}.wav'), z[0], hp.sr)
