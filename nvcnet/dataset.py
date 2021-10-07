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

from pathlib import Path

import numpy as np
from nnabla.utils.data_source import DataSource


class VCTKDataSource(DataSource):
    r""" Data source for the VCTK dataset."""

    def __init__(self, metadata, hp, shuffle=False, rng=None):
        if rng is None:
            rng = np.random.RandomState(hp.seed)
        super().__init__(shuffle=shuffle, rng=rng)
        self._path = Path(hp.save_data_dir)
        waves = list()
        with open(self._path / metadata) as reader:
            for line in reader:
                data = line.strip()
                waves.append(data)

        # split data
        n = len(waves)
        index = self._rng.permutation(n) if shuffle else np.arange(n)

        if hasattr(hp, 'comm'):  # distributed learning
            num = n // hp.comm.n_procs
            index = index[num * hp.comm.rank:num * (hp.comm.rank + 1)]

        self._waves = [waves[i] for i in index]
        self._size = len(self._waves)
        self._variables = ["wave", "speaker_id"]
        self.hp = hp

        self.reset()

    def reset(self):
        if self._shuffle:
            self._indexes = self._rng.permutation(self._size)
        else:
            self._indexes = np.arange(self._size)
        super().reset()

    def _get_data(self, position):
        r"""Return a tuple of data."""
        index = self._indexes[position]
        name = self._waves[index]
        hp = self.hp
        data = np.load(self._path / "data" / name)
        w, label = data["wave"], data["speaker_id"]
        w *= 0.99 / (np.max(np.abs(w)) + 1e-7)
        if len(w) > hp.segment_length:
            idx = self._rng.randint(0, len(w) - hp.segment_length)
            w = w[idx:idx + hp.segment_length]
        else:
            w = np.pad(w, (0, hp.segment_length - len(w)), mode='constant')
        return w[None, ...], [label]
