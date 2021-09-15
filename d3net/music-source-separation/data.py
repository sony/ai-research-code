# Copyright 2021 Sony Group Corporation.
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

'''
MUSDB18 data-iterator code for MSS.
'''

import random
import numpy as np
import musdb
from nnabla.utils.data_source import DataSource


class Compose():
    """Composes several augmentation transforms.
    Args:
        augmentations: list of augmentations to compose.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, audio):
        for t in self.transforms:
            audio = t(audio)
        return audio


def _augment_gain(audio, low=0.75, high=1.25):
    """Applies a random gain between `low` and `high`"""
    g = random.uniform(low, high)
    return audio * g


def _augment_channelswap(audio):
    """Swap channels of stereo signals with a probability of p=0.5"""
    if audio.shape[0] == 2 and random.random() < 0.5:
        return np.flip(audio, 0)
    else:
        return audio


def load_datasources(parser, args):
    """Loads the specified dataset from commandline arguments
    Returns:
        train_dataset, validation_dataset
    """

    parser.add_argument('--is-wav', action='store_true', default=True,
                        help='loads wav instead of STEMS')
    parser.add_argument('--samples-per-track', type=int, default=64)
    parser.add_argument(
        '--source-augmentations', type=str, nargs='+',
        default=['gain', 'channelswap']
    )

    args = parser.parse_args()

    source_augmentations = Compose(
        [globals()['_augment_' + aug] for aug in args.source_augmentations]
    )

    train_dataset = MUSDBDataSource(
        source_augmentations=source_augmentations, random_track_mix=True, args=args)

    return train_dataset, args


class MUSDBDataSource(DataSource):
    def __init__(
        self,
        args,
        download=False,
        samples_per_track=64,
        source_augmentations=lambda audio: audio,
        random_track_mix=False,
        dtype=np.float32,
        seed=42,
        rng=None
    ):
        """
        MUSDB18 nnabla.utils.data_source that samples from the MUSDB tracks
        using track and excerpts with replacement.

        Parameters
        ----------
        args : additional arguments used to add further control for 
            the musdb dataset initialization function.
        download : boolean
            automatically download 7s preview version of MUS
        samples_per_track : int
            sets the number of samples, yielded from each track per epoch.
            Defaults to 64
        source_augmentations : list[callables]
            provide list of augmentation function that take a multi-channel
            audio file of shape (src, samples) as input and output. Defaults to
            no-augmentations (input = output)
        random_track_mix : boolean
            randomly mixes sources from different tracks to assemble a
            custom mix. This augmenation is only applied for the train subset.
        seed : int
            control randomness of dataset iterations
        dtype : numeric type
            data type of torch output tuple x and y
        """

        super(MUSDBDataSource, self).__init__(shuffle=True)
        if rng is None:
            rng = np.random.RandomState(seed)
        self.rng = rng

        random.seed(seed)
        self.args = args
        self.download = args.root is None
        self.samples_per_track = samples_per_track
        self.source_augmentations = source_augmentations
        self.random_track_mix = random_track_mix
        self.mus = musdb.DB(
            root=args.root,
            is_wav=args.is_wav,
            split=None,
            subsets='train',
            download=download
        )

        print(f"Finished loading dataset with {len(self.mus.tracks)} tracks.")

        self.sample_rate = 44100  # musdb has fixed sample rate
        self.dtype = dtype

        self._size = len(self.mus.tracks) * self.samples_per_track
        self._variables = ('mixture', 'target')
        self.reset()

    def _get_data(self, position):
        index = self._indexes[position]
        audio_sources = []
        target_ind = None

        # select track
        track = self.mus.tracks[index // self.samples_per_track]

        # at training time we assemble a custom mix
        if self.args.seq_dur:
            for k, source in enumerate(self.mus.setup['sources']):

                # memorize index of target source
                if source == self.args.target:
                    target_ind = k

                # select a random track
                if self.random_track_mix:
                    track = random.choice(self.mus.tracks)

                # set the excerpt duration
                track.chunk_duration = self.args.seq_dur

                # set random start index
                track.chunk_start = random.uniform(
                    0, track.duration - self.args.seq_dur
                )

                # load source audio and apply time domain source_augmentations
                audio = track.sources[source].audio.T
                audio = self.source_augmentations(audio)
                audio_sources.append(audio)

            # create stem tensor of shape (source, channel, samples)
            stems = np.stack(audio_sources, axis=0)
            # # apply linear mix over source index=0
            x = np.sum(stems, axis=0)

            # get the target stem
            if target_ind is not None:
                y = stems[target_ind]
            # assuming vocal/accompaniment scenario if target!=source
            else:
                vocind = list(self.mus.setup['sources'].keys()).index('vocals')
                # apply time domain subtraction
                y = x - stems[vocind]

        # for validation and test, we deterministically yield the full musdb track
        else:
            # get the non-linear source mix straight from musdb
            x = track.audio.T
            y = track.targets[self.args.target].audio.T
        return x, y

    def reset(self):
        if self._shuffle:
            self._indexes = self.rng.permutation(self._size)
        else:
            self._indexes = np.arange(self._size)
        super(MUSDBDataSource, self).reset()
