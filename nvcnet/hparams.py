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

from utils.hparams import HParams

hparams = HParams(
    # directory to the data
    save_data_dir="precomputed-vctk/",  # path to precomputed features
    speaker_dir="data/list_of_speakers.txt",

    # weighing hyper-parameters
    lambda_rec=10.0,  # reconstruction term
    lambda_con=10.0,  # content preservation term
    lambda_kld=0.02,  # kl divergence term

    # optimization parameters
    batch_size=8,                     # batch size
    epoch=500,                        # number of epochs
    print_frequency=50,               # number of iterations before printing
    epochs_per_checkpoint=50,         # number of epochs for each checkpoint
    output_path="./log/output/",      # directory to save results
    seed=123456,                      # random seed
    g_lr=1e-4,                        # learning rate for generator
    d_lr=1e-4,                        # learning rate for discriminator
    beta1=0.5,
    beta2=0.9,
    weight_decay=0,

    sr=22050,                         # sampling rate
    segment_length=32768,             # sample length
    n_speaker_embedding=128,          # dimension of speaker embedding


    # Discriminator network
    ndf=16,
    n_layers_D=4,                     # number of layers in discriminator
    num_D=3,
    downsamp_factor=4,
    n_D_updates=2,

    # Generator network
    ngf=32,
    n_residual_layers=4,
    ratios=[8, 8, 2, 2],
    bottleneck_dim=4,

    # Speaker network
    n_spk_layers=5,                  # number of layers in speaker encoder

    # multi-scale spectral loss
    window_sizes=[2048, 1024, 512],

    # data augmentation
    scale_low=0.25,                  # lower bound used in random scaling
    scale_high=1.0,                  # upper bound used in random scaling
    split_low=30,                    # lower bound used in random shuffle
    split_hight=45,                  # upper bound used in random shuffle
    max_jitter_steps=30,             # random jitter
)
