""" Parametric Quantization """
import numpy as np
import nnabla as nn
import nnabla.functions as F
from nnabla.parametric_functions import parametric_function_api
from nnabla.parameter import get_parameter_or_create
from nnabla.initializer import (
    calc_uniform_lim_glorot, ConstantInitializer, UniformInitializer)
from nnabla.function import PythonFunction


@parametric_function_api("parametric_fp", [
    ('n', 'bitwidth (float)', '()', True),
    ('m', 'dynamic range (float)', '()', True),
])
def parametric_fixed_point_quantize(x, sign=True,
                                    n_init=8, n_min=2, n_max=16,
                                    m_init=1, m_min=-8, m_max=8,
                                    fix_parameters=False):
    """Parametric version of `fixed_point_quantize` where the
    bitwidth `n` and dynamic range `m` are learnable parameters.

    Args:
        x(~nnabla.Variable): N-D array as input
        sign (bool): keep sign information during quantization.
        n_init (:obj:`nnabla.initializer.BaseInitializer` or :obj:`numpy.ndarray`): Initializer for bitwidth parameter.
        n_min (int): lower bound for bitwidth.
        n_max (int): upper bound for bitwidth.
        m_init (:obj:`nnabla.initializer.BaseInitializer` or :obj:`numpy.ndarray`): Initializer for dynamic range.
        m_min (float): lower bound for dynamic range.
        m_max (float): upper bound for range.
        fix_parameters (bool): When set to `True`, the negative slope values
            will not be updated.

    Returns:
        ~nnabla.Variable: N-D array.
    """
    def clip_scalar(v, min_value, max_value):
        return F.minimum_scalar(F.maximum_scalar(v, min_value), max_value)

    def broadcast_scalar(v, shape):
        return F.broadcast(F.reshape(v, (1,) * len(shape), inplace=False), shape=shape)

    def quantize_pow2(v):
        return 2 ** F.round(F.log(v) / np.log(2.))

    n = get_parameter_or_create("n", (),
                                ConstantInitializer(n_init),
                                need_grad=True,
                                as_need_grad=not fix_parameters)
    m = get_parameter_or_create("m", (),
                                ConstantInitializer(m_init),
                                need_grad=True,
                                as_need_grad=not fix_parameters)

    # ensure that bitwidth is in specified range and an integer
    n_q = F.round(clip_scalar(n, n_min, n_max))
    if sign:
        n_q = n_q - 1

    # ensure that dynamic range is in specified range
    m_q = clip_scalar(m, m_min, m_max)

    # compute step size from dynamic range and make sure that it is a pow2
    d_q = quantize_pow2((2 ** m_q) / (2 ** n_q - 1))

    # compute min/max value that we can represent
    x_max = d_q * (2 ** n_q - 1)
    if sign:
        x_min = -x_max
    else:
        x_min = nn.Variable((1,), need_grad=False)
        x_min.d = 0.

    # broadcast variables to correct size
    d_q = broadcast_scalar(d_q, shape=x.shape)
    x_min = broadcast_scalar(x_min, shape=x.shape)
    x_max = broadcast_scalar(x_max, shape=x.shape)

    # apply fixed-point quantization
    return d_q * F.round(F.clip_by_value(x, x_min, x_max) / d_q)


# CASE A: PARAMETRIZATION BY B AND XMAX
@parametric_function_api("parametric_fp_b_xmax", [
    ('n', 'bitwidth (float)', '()', True),
    ('xmax', 'dynamic range (float)', '()', True),
])
def parametric_fixed_point_quantize_b_xmax(x, sign=True,
                                           n_init=8, n_min=2, n_max=16,
                                           xmax_init=1, xmax_min=0.001, xmax_max=10,
                                           fix_parameters=False):
    """Parametric version of `fixed_point_quantize` where the
    bitwidth `b` and dynamic range `xmax` are learnable parameters.

    Returns:
        ~nnabla.Variable: N-D array.
    """
    def clip_scalar(v, min_value, max_value):
        return F.minimum_scalar(F.maximum_scalar(v, min_value), max_value)

    def broadcast_scalar(v, shape):
        return F.broadcast(F.reshape(v, (1,) * len(shape), inplace=False), shape=shape)

    def quantize_pow2(v):
        return 2 ** F.round(F.log(v) / np.log(2.))

    n = get_parameter_or_create("n", (),
                                ConstantInitializer(n_init),
                                need_grad=True,
                                as_need_grad=not fix_parameters)
    xmax = get_parameter_or_create("xmax", (),
                                   ConstantInitializer(xmax_init),
                                   need_grad=True,
                                   as_need_grad=not fix_parameters)

    # ensure that bitwidth is in specified range and an integer
    n = F.round(clip_scalar(n, n_min, n_max))
    if sign:
        n = n - 1

    # ensure that dynamic range is in specified range
    xmax = clip_scalar(xmax, xmax_min, xmax_max)

    # compute step size from dynamic range and make sure that it is a pow2
    d = quantize_pow2(xmax / (2 ** n - 1))

    # compute min/max value that we can represent
    if sign:
        xmin = -xmax
    else:
        xmin = nn.Variable((1,), need_grad=False)
        xmin.d = 0.

    # broadcast variables to correct size
    d = broadcast_scalar(d, shape=x.shape)
    xmin = broadcast_scalar(xmin, shape=x.shape)
    xmax = broadcast_scalar(xmax, shape=x.shape)

    # apply fixed-point quantization
    return d * F.round(F.clip_by_value(x, xmin, xmax) / d)


# CASE B: PARAMETRIZATION BY D AND XMAX
@parametric_function_api("parametric_fp_d_xmax", [
    ('d', 'step size (float)', '()', True),
    ('xmax', 'dynamic range (float)', '()', True),
])
def parametric_fixed_point_quantize_d_xmax(x, sign=True,
                                           d_init=2**-4, d_min=2**-8, d_max=2**8,
                                           xmax_init=1, xmax_min=0.001, xmax_max=10,
                                           fix_parameters=False):
    """Parametric version of `fixed_point_quantize` where the
    stepsize `d` and dynamic range `xmax` are learnable parameters.

    Returns:
        ~nnabla.Variable: N-D array.
    """
    def clip_scalar(v, min_value, max_value):
        return F.minimum_scalar(F.maximum_scalar(v, min_value), max_value)

    def broadcast_scalar(v, shape):
        return F.broadcast(F.reshape(v, (1,) * len(shape), inplace=False), shape=shape)

    def quantize_pow2(v):
        return 2 ** F.round(F.log(v) / np.log(2.))

    d = get_parameter_or_create("d", (),
                                ConstantInitializer(d_init),
                                need_grad=True,
                                as_need_grad=not fix_parameters)
    xmax = get_parameter_or_create("xmax", (),
                                   ConstantInitializer(xmax_init),
                                   need_grad=True,
                                   as_need_grad=not fix_parameters)

    # ensure that stepsize is in specified range and a power of two
    d = quantize_pow2(clip_scalar(d, d_min, d_max))

    # ensure that dynamic range is in specified range
    xmax = clip_scalar(xmax, xmax_min, xmax_max)

    # compute min/max value that we can represent
    if sign:
        xmin = -xmax
    else:
        xmin = nn.Variable((1,), need_grad=False)
        xmin.d = 0.

    # broadcast variables to correct size
    d = broadcast_scalar(d, shape=x.shape)
    xmin = broadcast_scalar(xmin, shape=x.shape)
    xmax = broadcast_scalar(xmax, shape=x.shape)

    # apply fixed-point quantization
    return d * F.round(F.clip_by_value(x, xmin, xmax) / d)


# CASE C: PARAMETRIZATION BY B AND D
@parametric_function_api("parametric_fp_d_b", [
     ('n', 'bitwidth (float)', '()', True),
     ('d', 'step size (float)', '()', True),
])
def parametric_fixed_point_quantize_d_b(x, sign=True,
                                        n_init=8, n_min=2, n_max=16,
                                        d_init=2**-4, d_min=2**-8, d_max=2**8,
                                        fix_parameters=False):
    """Parametric version of `fixed_point_quantize` where the
    bitwidth `b` and stepsize `d` are learnable parameters.

    Returns:
        ~nnabla.Variable: N-D array.
    """
    def clip_scalar(v, min_value, max_value):
        return F.minimum_scalar(F.maximum_scalar(v, min_value), max_value)

    def broadcast_scalar(v, shape):
        return F.broadcast(F.reshape(v, (1,) * len(shape), inplace=False), shape=shape)

    def quantize_pow2(v):
        return 2 ** F.round(F.log(v) / np.log(2.))

    n = get_parameter_or_create("n", (),
                                ConstantInitializer(n_init),
                                need_grad=True,
                                as_need_grad=not fix_parameters)
    d = get_parameter_or_create("d", (),
                                ConstantInitializer(d_init),
                                need_grad=True,
                                as_need_grad=not fix_parameters)

    # ensure that bitwidth is in specified range and an integer
    n = F.round(clip_scalar(n, n_min, n_max))
    if sign:
        n = n - 1

    # ensure that stepsize is in specified range and a power of two
    d = quantize_pow2(clip_scalar(d, d_min, d_max))

    # ensure that dynamic range is in specified range
    xmax = d * (2 ** n - 1)

    # compute min/max value that we can represent
    if sign:
        xmin = -xmax
    else:
        xmin = nn.Variable((1,), need_grad=False)
        xmin.d = 0.

    # broadcast variables to correct size
    d = broadcast_scalar(d, shape=x.shape)
    xmin = broadcast_scalar(xmin, shape=x.shape)
    xmax = broadcast_scalar(xmax, shape=x.shape)

    # apply fixed-point quantization
    return d * F.round(F.clip_by_value(x, xmin, xmax) / d)


@parametric_function_api("parametric_pow2", [
    ('n', 'bitwidth (float)', '()', True),
    ('m', 'dynamic range (float)', '()', True),
])
def parametric_pow2_quantize(x, sign=True, with_zero=True,
                             n_init=8, n_min=1, n_max=16,
                             m_init=1, m_min=-8, m_max=8,
                             fix_parameters=False):
    """Parametric version of `pow2_quantize` where the
    bitwidth `n` and dynamic range `m` are learnable parameters.

    Args:
        x(~nnabla.Variable): N-D array as input
        sign (bool): keep sign information during quantization.
        with_zero (bool): quantize small weights to zero.
        n_init (:obj:`nnabla.initializer.BaseInitializer` or :obj:`numpy.ndarray`): Initializer for bitwidth parameter.
        n_min (int): lower bound for bitwidth.
        n_max (int): upper bound for bitwidth.
        m_init (:obj:`nnabla.initializer.BaseInitializer` or :obj:`numpy.ndarray`): Initializer for dynamic range.
        m_min (float): lower bound for dynamic range.
        m_max (float): upper bound for dynamic range.
        fix_parameters (bool): When set to `True`, the negative slope values
            will not be updated.

    Returns:
        ~nnabla.Variable: N-D array.
    """
    def clip_scalar(v, min_value, max_value):
        return F.minimum_scalar(F.maximum_scalar(v, min_value), max_value)

    def broadcast_scalar(v, shape):
        return F.broadcast(F.reshape(v, (1,) * len(shape), inplace=False), shape=shape)

    def quantize_pow2(v):
        return 2 ** F.round(F.log(F.abs(v)) / np.log(2.))

    n = get_parameter_or_create("n", (),
                                ConstantInitializer(n_init),
                                need_grad=True,
                                as_need_grad=not fix_parameters)
    m = get_parameter_or_create("m", (),
                                ConstantInitializer(m_init),
                                need_grad=True,
                                as_need_grad=not fix_parameters)

    # ensure that bitwidth is in specified range and an integer
    n_q = F.round(clip_scalar(n, n_min, n_max))
    if sign:
        n_q = n_q - 1
    if with_zero:
        n_q = n_q - 1

    # ensure that dynamic range is in specified range and an integer
    m_q = F.round(clip_scalar(m, m_min, m_max))

    # compute min/max value that we can represent
    x_max = 2 ** m_q
    x_min = 2 ** (m_q - (2 ** n_q) + 1)

    # broadcast variables to correct size
    x_min = broadcast_scalar(x_min, shape=x.shape)
    x_max = broadcast_scalar(x_max, shape=x.shape)

    # if unsigned, then quantize all negative values to zero
    if not sign:
        x = F.relu(x)

    # compute absolute value/sign of input
    ax = F.abs(x)
    sx = F.sign(x)

    if with_zero:
        # prune smallest elements (in magnitude) to zero if they are smaller
        # than `x_min / \sqrt(2)`
        x_threshold = x_min / np.sqrt(2)

        idx1 = F.greater_equal(ax, x_threshold) * F.less(ax, x_min)
        idx2 = F.greater_equal(ax, x_min) * F.less(ax, x_max)
        idx3 = F.greater_equal(ax, x_max)
    else:
        idx1 = F.less(ax, x_min)
        idx2 = F.greater_equal(ax, x_min) * F.less(ax, x_max)
        idx3 = F.greater_equal(ax, x_max)

    # do not backpropagate gradient through indices
    idx1.need_grad = False
    idx2.need_grad = False
    idx3.need_grad = False

    # do not backpropagate gradient through sign
    sx.need_grad = False

    # take care of values outside of dynamic range
    return sx * (x_min * idx1 + quantize_pow2(ax) * idx2 + x_max * idx3)


# CASE A: PARAMETRIZATION BY B AND XMAX
@parametric_function_api("parametric_pow2_b_xmax", [
    ('n', 'bitwidth (float)', '()', True),
    ('xmax', 'dynamic range (float)', '()', True),
])
def parametric_pow2_quantize_b_xmax(x, sign=True, with_zero=True,
                                    n_init=8, n_min=1, n_max=8,
                                    xmax_init=2**0, xmax_min=2**-8, xmax_max=256,
                                    fix_parameters=False):
    """Parametric version of `pow2_quantize` where the
    bitwidth `n` and range `xmax` are learnable parameters.

    Returns:
        ~nnabla.Variable: N-D array.
    """
    def clip_scalar(v, min_value, max_value):
        return F.minimum_scalar(F.maximum_scalar(v, min_value), max_value)

    def broadcast_scalar(v, shape):
        return F.broadcast(F.reshape(v, (1,) * len(shape), inplace=False), shape=shape)

    def quantize_pow2(v):
        return 2 ** F.round(F.log(F.abs(v)) / np.log(2.))

    n = get_parameter_or_create("n", (),
                                ConstantInitializer(n_init),
                                need_grad=True,
                                as_need_grad=not fix_parameters)
    xmax = get_parameter_or_create("xmax", (),
                                   ConstantInitializer(xmax_init),
                                   need_grad=True,
                                   as_need_grad=not fix_parameters)

    # ensure that bitwidth is in specified range and an integer
    n = F.round(clip_scalar(n, n_min, n_max))
    if sign:
        n = n - 1
    if with_zero:
        n = n - 1

    # ensure that dynamic range is in specified range and an integer
    xmax = quantize_pow2(clip_scalar(xmax, xmax_min, xmax_max))

    # compute min value that we can represent
    xmin = (2 ** (-(2 ** n) + 1)) * xmax

    # broadcast variables to correct size
    xmin = broadcast_scalar(xmin, shape=x.shape)
    xmax = broadcast_scalar(xmax, shape=x.shape)

    # if unsigned, then quantize all negative values to zero
    if not sign:
        x = F.relu(x)

    # compute absolute value/sign of input
    ax = F.abs(x)
    sx = F.sign(x)

    if with_zero:
        # prune smallest elements (in magnitude) to zero if they are smaller
        # than `x_min / \sqrt(2)`
        x_threshold = xmin / np.sqrt(2)

        idx1 = F.greater_equal(ax, x_threshold) * F.less(ax, xmin)
        idx2 = F.greater_equal(ax, xmin) * F.less(ax, xmax)
        idx3 = F.greater_equal(ax, xmax)
    else:
        idx1 = F.less(ax, xmin)
        idx2 = F.greater_equal(ax, xmin) * F.less(ax, xmax)
        idx3 = F.greater_equal(ax, xmax)

    # do not backpropagate gradient through indices
    idx1.need_grad = False
    idx2.need_grad = False
    idx3.need_grad = False

    # do not backpropagate gradient through sign
    sx.need_grad = False

    # take care of values outside of dynamic range
    return sx * (xmin * idx1 + quantize_pow2(ax) * idx2 + xmax * idx3)


# CASE B: PARAMETRIZATION BY B AND XMIN
@parametric_function_api("parametric_pow2_b_xmin", [
    ('n', 'bitwidth (float)', '()', True),
    ('xmin', 'minimum dynamic range (float)', '()', True),
])
def parametric_pow2_quantize_b_xmin(x, sign=True, with_zero=True,
                                    n_init=8, n_min=1, n_max=8,
                                    xmin_init=2**-7, xmin_min=2**-15, xmin_max=256,
                                    fix_parameters=False):
    """Parametric version of `pow2_quantize` where the
    bitwidth `n` and the smallest value `xmin` are learnable parameters.

    Returns:
        ~nnabla.Variable: N-D array.
    """
    def clip_scalar(v, min_value, max_value):
        return F.minimum_scalar(F.maximum_scalar(v, min_value), max_value)

    def broadcast_scalar(v, shape):
        return F.broadcast(F.reshape(v, (1,) * len(shape), inplace=False), shape=shape)

    def quantize_pow2(v):
        return 2 ** F.round(F.log(F.abs(v)) / np.log(2.))

    n = get_parameter_or_create("n", (),
                                ConstantInitializer(n_init),
                                need_grad=True,
                                as_need_grad=not fix_parameters)
    xmin = get_parameter_or_create("xmin", (),
                                   ConstantInitializer(xmin_init),
                                   need_grad=True,
                                   as_need_grad=not fix_parameters)

    # ensure that bitwidth is in specified range and an integer
    n = F.round(clip_scalar(n, n_min, n_max))
    if sign:
        n = n - 1
    if with_zero:
        n = n - 1

    # ensure that minimum dynamic range is in specified range and a power-of-two
    xmin = quantize_pow2(clip_scalar(xmin, xmin_min, xmin_max))

    # compute min/max value that we can represent
    xmax = xmin * (2 ** ((2 ** n) - 1))

    # broadcast variables to correct size
    xmin = broadcast_scalar(xmin, shape=x.shape)
    xmax = broadcast_scalar(xmax, shape=x.shape)

    # if unsigned, then quantize all negative values to zero
    if not sign:
        x = F.relu(x)

    # compute absolute value/sign of input
    ax = F.abs(x)
    sx = F.sign(x)

    if with_zero:
        # prune smallest elements (in magnitude) to zero if they are smaller
        # than `x_min / \sqrt(2)`
        x_threshold = xmin / np.sqrt(2)

        idx1 = F.greater_equal(ax, x_threshold) * F.less(ax, xmin)
        idx2 = F.greater_equal(ax, xmin) * F.less(ax, xmax)
        idx3 = F.greater_equal(ax, xmax)
    else:
        idx1 = F.less(ax, xmin)
        idx2 = F.greater_equal(ax, xmin) * F.less(ax, xmax)
        idx3 = F.greater_equal(ax, xmax)

    # do not backpropagate gradient through indices
    idx1.need_grad = False
    idx2.need_grad = False
    idx3.need_grad = False

    # do not backpropagate gradient through sign
    sx.need_grad = False

    # take care of values outside of dynamic range
    return sx * (xmin * idx1 + quantize_pow2(ax) * idx2 + xmax * idx3)


# CASE C: PARAMETRIZATION BY XMIN AND XMAX
@parametric_function_api("parametric_pow2_xmin_xmax", [
    ('xmin', 'min dynamic range (float)', '()', True),
    ('xmax', 'max dynamic range (float)', '()', True),
])
def parametric_pow2_quantize_xmin_xmax(x, sign=True, with_zero=True,
                                       xmin_init=2**-7, xmin_min=2**-15, xmin_max=256,
                                       xmax_init=2**0, xmax_min=2**-8, xmax_max=256,
                                       fix_parameters=False):
    """Parametric version of `pow2_quantize` where the
    min value `xmin` and max value `xmax` are learnable parameters.

    Returns:
        ~nnabla.Variable: N-D array.
    """
    def clip_scalar(v, min_value, max_value):
        return F.minimum_scalar(F.maximum_scalar(v, min_value), max_value)

    def broadcast_scalar(v, shape):
        return F.broadcast(F.reshape(v, (1,) * len(shape), inplace=False), shape=shape)

    def quantize_pow2(v):
        return 2. ** F.round(F.log(F.abs(v)) / np.log(2.))

    xmin = get_parameter_or_create("xmin", (),
                                   ConstantInitializer(xmin_init),
                                   need_grad=True,
                                   as_need_grad=not fix_parameters)
    xmax = get_parameter_or_create("xmax", (),
                                   ConstantInitializer(xmax_init),
                                   need_grad=True,
                                   as_need_grad=not fix_parameters)

    # ensure that minimum dynamic range is in specified range and a power-of-two
    xmin = quantize_pow2(clip_scalar(xmin, xmin_min, xmin_max))

    # ensure that minimum dynamic range is in specified range and a power-of-two
    xmax = quantize_pow2(clip_scalar(xmax, xmax_min, xmax_max))

    # broadcast variables to correct size
    xmin = broadcast_scalar(xmin, shape=x.shape)
    xmax = broadcast_scalar(xmax, shape=x.shape)

    # if unsigned, then quantize all negative values to zero
    if not sign:
        x = F.relu(x)

    # compute absolute value/sign of input
    ax = F.abs(x)
    sx = F.sign(x)

    if with_zero:
        # prune smallest elements (in magnitude) to zero if they are smaller
        # than `x_min / \sqrt(2)`
        x_threshold = xmin / np.sqrt(2)

        idx1 = F.greater_equal(ax, x_threshold) * F.less(ax, xmin)
        idx2 = F.greater_equal(ax, xmin) * F.less(ax, xmax)
        idx3 = F.greater_equal(ax, xmax)
    else:
        idx1 = F.less(ax, xmin)
        idx2 = F.greater_equal(ax, xmin) * F.less(ax, xmax)
        idx3 = F.greater_equal(ax, xmax)

    # do not backpropagate gradient through indices
    idx1.need_grad = False
    idx2.need_grad = False
    idx3.need_grad = False

    # do not backpropagate gradient through sign
    sx.need_grad = False

    # take care of values outside of dynamic range
    return sx * (xmin * idx1 + quantize_pow2(ax) * idx2 + xmax * idx3)


@parametric_function_api("quantized_affine", [
    ('W', 'Weight matrix in float', '(inmaps, outmaps)', True),
    ('b', 'Bias vector in float', '(outmaps,)', True),
    ('W_q', 'Quantized weights', '(inmaps, outmaps)', False),
    ('b_q', 'Quantized biases', '(outmaps,)', False),
])
def quantized_affine(inp, n_outmaps,
                     base_axis=1,
                     w_init=None, b_init=None,
                     fix_parameters=False, rng=None, with_bias=True,
                     quantization_w=None, quantization_b=None):
    """Quantized Affine.

    Quantized affine with

    .. math::

        y_j = \sum_{i} Q_w(w_{ji}) x_i + Q_b(b_j),

    where :math:`Q_w(.)` is the weight quantization function
    and :math:`Q_b(.)` the bias quantization function, respectively.

    .. note::

        1) if you would like to share weights between some layers, please
        make sure to share the standard, floating value weights (`weight`)
        and not the quantized weights (`quantized weight`)

        2) The weights and the quantized weights become synced only after :func:`~nnabla._variable.Variable.forward` is called,
        and not after a call to :func:`~nnabla._variable.Variable.backward`.
        To access the parameters of the network, remember to call :func:`~nnabla._variable.Variable.forward` once before doing so, otherwise the
        float weights and the quantized weights will not be in sync.

        3) CPU and GPU implementations now use float value for `quantized weight`,
        since this function is only for simulation purposes.

    Args:
        inp (~nnabla.Variable): Input N-D array with shape (:math:`M_0 \\times \ldots \\times M_{B-1} \\times D_B \\times \ldots \\times D_N`). Dimensions before and after base_axis are flattened as if it is a matrix.
        n_outmaps (:obj:`int` or :obj:`tuple` of :obj:`int`): Number of output neurons per data.
        base_axis (int): Dimensions up to `base_axis` are treated as the sample dimensions.
        w_init (:obj:`nnabla.initializer.BaseInitializer` or :obj:`numpy.ndarray`): Initializer for weight.
        b_init (:obj:`nnabla.initializer.BaseInitializer` or :obj:`numpy.ndarray`): Initializer for bias.
        fix_parameters (bool): When set to `True`, the weights and biases will not be updated.
        rng (numpy.random.RandomState): Random generator for Initializer.
        with_bias (bool): Specify whether to include the bias term.
        quantization_w (function): Quantization function that is applied to the the weights.
            Use `None` to not quantize the weights.
        quantization_b (function): Quantization function that is applied to the the bias.
            Use `None` to not quantize the bias.

    Returns:
        :class:`~nnabla.Variable`: :math:`(B + 1)`-D array. (:math:`M_0 \\times \ldots \\times M_{B-1} \\times L`)

    """

    if not hasattr(n_outmaps, '__iter__'):
        n_outmaps = [n_outmaps]
    n_outmaps = list(n_outmaps)
    n_outmap = int(np.prod(n_outmaps))
    if w_init is None:
        inmaps = np.prod(inp.shape[base_axis:])
        w_init = UniformInitializer(
            calc_uniform_lim_glorot(inmaps, n_outmap), rng=rng)
    if with_bias and b_init is None:
        b_init = ConstantInitializer()

    # Floating Weight
    w = get_parameter_or_create(
        "W", [int(np.prod(inp.shape[base_axis:]))] + n_outmaps,
        w_init, True, not fix_parameters)

    # Quantize weights
    if quantization_w is not None:
        w_q = get_parameter_or_create(
            "W_q", [int(np.prod(inp.shape[base_axis:]))] + n_outmaps,
            w_init, False)
        # Link computation graph
        real_w_q = quantization_w(w)
        real_w_q.persistent = True

        w_q.data = real_w_q.data
    else:
        real_w_q = w

    # Bias
    # Floating
    b = None
    b_q = None
    real_b_q = None
    if with_bias:
        b = get_parameter_or_create(
            "b", n_outmaps, b_init, True, not fix_parameters)
        if quantization_b is not None:
            b_q = get_parameter_or_create(
                "b_q", n_outmaps, b_init, False)
            # Link computation graph
            real_b_q = quantization_b(b)
            real_b_q.persistent = True

            b_q.data = real_b_q.data
        else:
            real_b_q = b

    return F.affine(inp, real_w_q, real_b_q, base_axis)


@parametric_function_api("quantized_conv", [
    ('W', 'Filter weights in float', '(outmaps, inmaps // group, *kernel)', True),
    ('b', 'Bias vector in float', '(outmaps,)', True),
    ('W_q', 'Quantized weights', '(outmaps, inmaps // group, *kernel)', False),
    ('b_q', 'Quantized biases', '(outmaps,)', False),
])
def quantized_convolution(inp, outmaps, kernel,
                          pad=None, stride=None, dilation=None, group=1,
                          w_init=None, b_init=None,
                          base_axis=1, fix_parameters=False, rng=None, with_bias=True,
                          quantization_w=None, quantization_b=None):
    """Quantized Convolution.

    Quantized Convolution where the input/output
    relationship is

    .. math::

        y_{n, a, b} = \sum_{m} \sum_{i} \sum_{j} Q_w(w_{n, m, i, j}) x_{m, a + i, b + j} + Q_b(b_n), 

    where :math:`Q_w(w_{n, m, i, j})` is the weight quantization function
    and :math:`Q_w(b_{n})` is the bias quantization function.

    .. note::

        1) if you would like to share weights between some layers, please
        make sure to share the standard, floating value weights (`weight`)
        and not the quantized weights (`quantized weight`)

        2) The weights and the quantized weights become synced only after :func:`~nnabla._variable.Variable.forward` is called,
        and not after a call to :func:`~nnabla._variable.Variable.backward`.
        To access the parameters of the network, remember to call :func:`~nnabla._variable.Variable.forward` once before doing so, otherwise the
        float weights and the quantized weights will not be in sync.

        3) CPU and GPU implementations now use float value for `quantized weight`,
        since this function is only for simulation purposes.

    Args:
        inp (~nnabla.Variable): N-D array.
        outmaps (int): Number of convolution kernels (which is equal to the number of output channels). For example, to apply convolution on an input with 16 types of filters, specify 16.
        kernel (:obj:`tuple` of :obj:`int`): Convolution kernel size. For example, to apply convolution on an image with a 3 (height) by 5 (width) two-dimensional kernel, specify (3,5).
        pad (:obj:`tuple` of :obj:`int`): Padding sizes for dimensions.
        stride (:obj:`tuple` of :obj:`int`): Stride sizes for dimensions.
        dilation (:obj:`tuple` of :obj:`int`): Dilation sizes for dimensions.
        group (int): Number of groups of channels. This makes connections across channels more sparse by grouping connections along map direction.
        w_init (:obj:`nnabla.initializer.BaseInitializer` or :obj:`numpy.ndarray`): Initializer for weight.
        b_init (:obj:`nnabla.initializer.BaseInitializer` or :obj:`numpy.ndarray`): Initializer for bias.
        base_axis (int): Dimensions up to `base_axis` are treated as the sample dimensions.
        fix_parameters (bool): When set to `True`, the weights and biases will not be updated.
        rng (numpy.random.RandomState): Random generator for Initializer.
        with_bias (bool): Specify whether to include the bias term.
        quantization_w (function): Quantization function that is applied to the the weights.
            Use `None` to not quantize the weights.
        quantization_b (function): Quantization function that is applied to the the bias.
            Use `None` to not quantize the bias.

    Returns:
        :class:`~nnabla.Variable`: N-D array.

    """
    if w_init is None:
        w_init = UniformInitializer(
            calc_uniform_lim_glorot(inp.shape[base_axis], outmaps, tuple(kernel)), rng=rng)
    if with_bias and b_init is None:
        b_init = ConstantInitializer()

    # Floating Weight
    w = get_parameter_or_create(
        "W", (outmaps, inp.shape[base_axis] // group) + tuple(kernel),
        w_init, True, not fix_parameters)

    # Quantize weights
    if quantization_w is not None:
        w_q = get_parameter_or_create(
            "W_q", (outmaps, inp.shape[base_axis] // group) + tuple(kernel),
            w_init, False)
        # Link computation graph
        real_w_q = quantization_w(w)
        real_w_q.persistent = True

        w_q.data = real_w_q.data
    else:
        real_w_q = w

    # Bias
    # Floating
    b = None
    b_q = None
    real_b_q = None
    if with_bias:
        b = get_parameter_or_create(
            "b", (outmaps,), b_init, True, not fix_parameters)
        if quantization_b is not None:
            b_q = get_parameter_or_create(
                "b_q", (outmaps,), b_init, False)
            # Link computation graph
            real_b_q = quantization_b(b)
            real_b_q.persistent = True

            b_q.data = real_b_q.data
        else:
            real_b_q = b

    return F.convolution(inp, real_w_q, real_b_q, base_axis, pad, stride, dilation, group)
