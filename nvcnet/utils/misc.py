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

import os

import matplotlib.pyplot as plt
import nnabla as nn
import nnabla.communicators as C
import nnabla.functions as F
import numpy as np
import seaborn as sns
from nnabla.ext_utils import get_extension_context
from nnabla.function import PythonFunction
from nnabla.initializer import UniformInitializer
from nnabla.parameter import get_parameter_or_create
from nnabla.parametric_functions import parametric_function_api
from sklearn.manifold import TSNE


@parametric_function_api("embed", [
    ('W', 'Embedding matrix', '(n_inputs, n_features)', True),
])
def embed(inp, n_inputs, n_features, initializer=None,
          fix_parameters=False, apply_w=None):
    """ Embed.

    Embed slices a matrix/tensor with indexing array/tensor. Weights are
    initialized with :obj:`nnabla.initializer.UniformInitializer` within
    the range of :math:`-\\sqrt{3}` and :math:`\\sqrt{3}`.

    Args:
        x(~nnabla.Variable): [Integer] Indices with shape
            :math:`(I_0, ..., I_N)`
        n_inputs : number of possible inputs, words or vocabraries
        n_features : number of embedding features
        fix_parameters (bool): When set to `True`, the embedding weight matrix
            will not be updated.
        apply_w (function): Lambda, function, or callable object applied to
            the weights.

    Returns:
        ~nnabla.Variable: Output with shape
            :math:`(I_0, ..., I_N, W_1, ..., W_M)`
    """
    if initializer is None:
        initializer = UniformInitializer((-np.sqrt(3.), np.sqrt(3)))
    w = get_parameter_or_create("W", [n_inputs, n_features],
                                initializer, True, not fix_parameters)
    if apply_w is not None:
        w = apply_w(w)
    return F.embed(inp, w)


def set_persistent_all(*variables):
    for var in variables:
        if var is None:
            continue

        if not isinstance(var, nn.Variable):
            raise ValueError("all variables must be nn.Variable")

        var.persistent = True


def tsne_map(X, labels, fpath):
    n_labels = len(set(labels))
    X = X.reshape((X.shape[0], -1))
    X_embedded = TSNE(n_components=2).fit_transform(X)
    plt.figure(figsize=(5.5, 5.5))
    sns.scatterplot(
        X_embedded[:, 0],
        X_embedded[:, 1],
        hue=labels, legend='full', palette=sns.color_palette(
            "bright", n_labels),
        style=labels, s=50)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(fpath, bbox_inches='tight')
    plt.close()


def immediate_dir(path):
    r"""Returns inmediate directories from path."""
    return [f.name for f in os.scandir(path) if f.is_dir()]


def create_float_context(ctx):
    ctx_float = get_extension_context(ctx.backend[0].split(':')[
                                      0], device_id=ctx.device_id)
    return ctx_float


class CommunicatorWrapper(object):
    def __init__(self, ctx):
        try:
            comm = C.MultiProcessDataParallelCommunicator(ctx)
        except Exception as e:
            print(e)
            print(('No communicator found. Running with a single process. '
                   'If you run this with MPI processes, all processes will '
                   'perform totally same.'))
            self.n_procs = 1
            self.rank = 0
            self.ctx = ctx
            self.ctx_float = create_float_context(ctx)
            self.comm = None
            return

        comm.init()
        self.n_procs = comm.size
        self.rank = comm.rank
        self.ctx = ctx
        self.ctx.device_id = str(self.rank)
        self.ctx_float = create_float_context(self.ctx)
        self.comm = comm

    def all_reduce(self, params, division, inplace):
        if self.n_procs == 1:
            # skip all reduce since no processes have to be all-reduced
            return
        self.comm.all_reduce(params, division=division, inplace=inplace)


class RandomSplit(PythonFunction):
    def __init__(self, lo, hi, axis, rng, ctx):
        super(RandomSplit, self).__init__(ctx)
        self.lo = lo
        self.hi = hi
        self.rng = rng
        self.axis = axis

    def _mask_gen(self, b, n):
        idx_0, idx_1 = [], []
        for i in range(b):
            # random split
            idx = np.cumsum(
                [self.rng.randint(self.lo, self.hi)
                 for _ in range(n // self.lo)])
            idx = idx[idx < n]
            partition = np.split(np.arange(n), idx)
            self.rng.shuffle(partition)
            # build index for gather_nd
            idx_0.append(np.repeat(i, n))
            idx_1.append(np.hstack(partition))

        idx_0 = np.vstack(idx_0)
        idx_1 = np.vstack(idx_1)

        return np.concatenate(([idx_0], [idx_1]))

    def name(self):
        return "RandomSplit"

    def min_outputs(self):
        return 1

    def setup_impl(self, inputs, outputs):
        i0 = inputs[0]
        o0 = outputs[0]
        o0.reset_shape(i0.shape, True)
        assert len(i0.shape) == 3
        assert self.axis == 1 or self.axis == 2

    def forward_impl(self, inputs, outputs):
        x0 = inputs[0].data
        if self.axis == 2:
            x0 = F.transpose(x0, (0, 2, 1))

        b, n, *_ = x0.shape
        self.mask = self._mask_gen(b, n)

        mask = nn.NdArray.from_numpy_array(self.mask)
        ans = F.gather_nd(x0, mask)

        if self.axis == 2:
            ans = F.transpose(ans, (0, 2, 1))

        y = outputs[0].data
        y.copy_from(ans)

    def backward_impl(self, inputs, outputs, propagate_down, accum):
        # Grads of inputs and outputs
        dx0 = inputs[0].grad
        dy = outputs[0].grad
        grad = dy

        if self.axis == 2:
            grad = F.transpose(grad, (0, 2, 1))

        mask = nn.NdArray.from_numpy_array(self.mask)
        grad = F.gather_nd(grad, mask)

        if self.axis == 2:
            grad = F.transpose(grad, (0, 2, 1))

        # backward w.r.t. x0
        if propagate_down[0]:
            if accum[0]:
                dx0 += grad
            else:
                dx0.copy_from(grad)
