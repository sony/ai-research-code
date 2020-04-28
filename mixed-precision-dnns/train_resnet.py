import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solver as S
import numpy as np
from nnabla import logger
import cifar_data
from model_resnet import resnet_cifar10

import argparse
import configparser
import attr
from itertools import product

import sys
import os
import collections


def clip_scalar(v, min_value, max_value):
    return F.minimum_scalar(F.maximum_scalar(v, min_value), max_value)


def quantize_pow2(v):
    return 2 ** F.round(F.log(v) / np.log(2.))


def log2(x):
    return F.log(x) / np.log(2.)


def get_args():
    """
    Get command line arguments.

    Arguments set the default values of command line arguments.
    """
    description = "Training ResNet on CIFAR-10"
    parser = argparse.ArgumentParser(description)
    parser.add_argument('experiment')
    parser.add_argument('--gpu', metavar='NUMBER', type=int,
                        help="use the (n+1)'th GPU, thus counting from 0",
                        default=0)
    parser.add_argument('--cfg', metavar='STRING',
                        help="experiment configuration file",
                        default=f"{os.path.splitext(sys.argv[0])[0]}.cfg")
    args = parser.parse_args()
    return args


def configparser_getboolean(value):
    boolean_states = configparser.ConfigParser().BOOLEAN_STATES
    try:
        return boolean_states[str(value).lower()]
    except KeyError:
        print("boolean values must be one of: {}"
              .format(', '.join(map(repr, boolean_states.keys()))))


@attr.s
class Configuration(object):
    experiment = attr.ib()
    num_layers = attr.ib(converter=int)
    shortcut_type = attr.ib()
    initial_learning_rate = attr.ib(converter=float)
    optimizer = attr.ib(default=None) 
    weightfile = attr.ib(default=None)

    w_quantize = attr.ib(default=None)
    a_quantize = attr.ib(default=None)

    # Uniform quantization (sign=True):
    #   xmax = stepsize * ( 2**(bitwidth-1) - 1 )
    #      -> xmax_min = stepsize_min * ( 2**(bitwidth_min-1) - 1)
    #      -> xmax_max = stepsize_max * ( 2**(bitwdith_max-1) - 1)
    # Pow2 quantization (sign=True, zero=True):
    #   xmax = xmin * 2**(2**(bitwidth-2) - 1)
    w_stepsize = attr.ib(converter=float, default=2**-3)
    w_stepsize_min = attr.ib(converter=float, default=2**-8)
    w_stepsize_max = attr.ib(converter=float, default=1)
    w_xmin_min = attr.ib(converter=float, default=2**-16)
    w_xmin_max = attr.ib(converter=float, default=127)
    w_xmax_min = attr.ib(converter=float, default=2**-8)
    w_xmax_max = attr.ib(converter=float, default=127)
    w_bitwidth = attr.ib(converter=int, default=4)
    w_bitwidth_min = attr.ib(converter=int, default=2)  # one bit for sign
    w_bitwidth_max = attr.ib(converter=int, default=8)

    # Uniform quantization (sign=False):
    #   xmax = stepsize * ( 2**bitwidth - 1 )
    # Pow2 quantization (sign=False, zero=True)
    #   xmax = xmin * 2**(2**(bitwidth-1) - 1)
    a_stepsize = attr.ib(converter=float, default=2**-3)
    a_stepsize_min = attr.ib(converter=float, default=2**-8)
    a_stepsize_max = attr.ib(converter=float, default=1)
    a_xmin_min = attr.ib(converter=float, default=2**-14)
    a_xmin_max = attr.ib(converter=float, default=255)
    a_xmax_min = attr.ib(converter=float, default=2**-8)
    a_xmax_max = attr.ib(converter=float, default=255)
    a_bitwidth = attr.ib(converter=int, default=4)
    a_bitwidth_min = attr.ib(converter=int, default=1)
    a_bitwidth_max = attr.ib(converter=int, default=8)

    target_weight_kbytes = attr.ib(converter=float, default=-1.)
    target_activation_kbytes = attr.ib(converter=float, default=-1.)
    target_activation_type = attr.ib(default='max')

    initial_cost_lambda2 = attr.ib(converter=float, default=0.1)
    initial_cost_lambda3 = attr.ib(converter=float, default=0.1)

    scale_layer = attr.ib(converter=bool, default=False)


# read arguments
args = get_args()
print(args)

cfgs = configparser.ConfigParser()
cfgs.read(args.cfg)


cfg = Configuration(**dict(cfgs[args.experiment].items()),
                    experiment=args.experiment)

logger.info("Configuration:")
for param, value in attr.asdict(cfg).items():
    logger.info(f"  {param} = {value}")

cfg.params_dir = f"{args.experiment}"
if not os.path.exists(cfg.params_dir):
    os.makedirs(cfg.params_dir)


def network_size_weights():
    """
    Return total number of weights and network size (for weights) in KBytes
    """
    kbytes = None
    num_params = None

    # get all parameters
    ps = nn.get_parameters()
    for p in ps:
        if ((p.endswith("quantized_conv/W") or
             p.endswith("quantized_conv/b") or
             p.endswith("quantized_affine/W") or
             p.endswith("quantized_affine/b"))):
            _num_params = np.prod(ps[p].shape)
            print(f"{p}\t{ps[p].shape}\t{_num_params}")

            if cfg.w_quantize is not None:
                if cfg.w_quantize in ['parametric_fp_b_xmax',
                                      'parametric_fp_d_b',
                                      'parametric_pow2_b_xmax',
                                      'parametric_pow2_b_xmin']:
                    # parametric quantization
                    n_p = p + "quant/" + cfg.w_quantize + "/n"
                    n = F.round(clip_scalar(ps[n_p], cfg.w_bitwidth_min, cfg.w_bitwidth_max))
                elif cfg.w_quantize == 'parametric_fp_d_xmax':
                    # this quantization methods do not have n, so we need to compute it
                    d = ps[p + "quant/"+cfg.w_quantize+"/d"]
                    xmax = ps[p + "quant/"+cfg.w_quantize+"/xmax"]

                    # ensure that stepsize is in specified range and a power of two
                    d_q = quantize_pow2(clip_scalar(d, cfg.w_stepsize_min, cfg.w_stepsize_max))

                    # ensure that dynamic range is in specified range
                    xmax = clip_scalar(xmax, cfg.w_xmax_min, cfg.w_xmax_max)

                    # compute real `xmax`
                    xmax = F.round(xmax / d_q) * d_q

                    # we do not clip to `cfg.w_bitwidth_max` as xmax/d_q could correspond to more than 8 bit
                    n = F.maximum_scalar(F.ceil(log2(xmax/d_q + 1.0) + 1.0), cfg.w_bitwidth_min)
                elif cfg.w_quantize == 'parametric_pow2_xmin_xmax':
                    # this quantization methods do not have n, so we need to compute it
                    xmin = ps[p + "quant/"+cfg.w_quantize+"/xmin"]
                    xmax = ps[p + "quant/"+cfg.w_quantize+"/xmax"]

                    # ensure that minimum dynamic range is in specified range and a power-of-two
                    xmin = quantize_pow2(clip_scalar(xmin, cfg.w_xmin_min, cfg.w_xmin_max))

                    # ensure that maximum dynamic range is in specified range and a power-of-two
                    xmax = quantize_pow2(clip_scalar(xmax, cfg.w_xmax_min, cfg.w_xmax_max))

                    # use ceil to determine bitwidth
                    n = F.maximum_scalar(F.ceil(log2(log2(xmax/xmin) + 1.0) + 1.), cfg.w_bitwidth_min)
                elif cfg.w_quantize == 'fp' or cfg.w_quantize == 'pow2':
                    # fixed quantization
                    n = nn.Variable((), need_grad=False)
                    n.d = cfg.w_bitwidth
                else:
                    raise ValueError(f'Unknown quantization method {cfg.w_quantize}')
            else:
                # float precision
                n = nn.Variable((), need_grad=False)
                n.d = 32.

            if kbytes is None:
                kbytes = n * _num_params / 8. / 1024.
                num_params = _num_params
            else:
                kbytes += n * _num_params / 8. / 1024.
                num_params += _num_params
    return num_params, kbytes


def network_size_activations():
    """
    Returns total number of activations
    and size in KBytes (NNabla variable using `max` or `sum` operator)
    """
    kbytes = []
    num_activations = 0

    # get all parameters
    ps = nn.get_parameters(grad_only=False)
    for p in ps:
        if "Asize" in p:
            print(f"{p}\t{ps[p].d}")

            num_activations += ps[p].d

            if cfg.a_quantize is not None:
                if cfg.a_quantize in ['fp_relu', 'pow2_relu']:
                    # fixed quantization
                    n = nn.Variable((), need_grad=False)
                    n.d = cfg.a_bitwidth
                elif cfg.a_quantize in ['parametric_fp_relu',
                                        'parametric_fp_b_xmax_relu',
                                        'parametric_fp_d_b_relu',
                                        'parametric_pow2_b_xmax_relu',
                                        'parametric_pow2_b_xmin_relu']:
                    # parametric quantization
                    s = p.replace("/Asize", "/Aquant/"+cfg.a_quantize.replace("_relu", "")+"/n")
                    n = F.round(clip_scalar(ps[s], cfg.a_bitwidth_min, cfg.a_bitwidth_max))
                elif cfg.a_quantize in ['parametric_fp_d_xmax_relu']:
                    # these quantization methods do not have n, so we need to compute it!
                    # parametric quantization
                    d = ps[p.replace("/Asize", "/Aquant/"+cfg.a_quantize.replace("_relu", "")+"/d")]
                    xmax = ps[p.replace("/Asize", "/Aquant/"+cfg.a_quantize.replace("_relu", "")+"/xmax")]

                    # ensure that stepsize is in specified range and a power of two
                    d_q = quantize_pow2(clip_scalar(d, cfg.a_stepsize_min, cfg.a_stepsize_max))

                    # ensure that dynamic range is in specified range
                    xmax = clip_scalar(xmax, cfg.a_xmax_min, cfg.a_xmax_max)

                    # compute real `xmax`
                    xmax = F.round(xmax / d_q) * d_q

                    n = F.maximum_scalar(F.ceil(log2(xmax/d_q + 1.0)), cfg.a_bitwidth_min)
                elif cfg.a_quantize in ['parametric_pow2_xmin_xmax_relu']:
                    # these quantization methods do not have n, so we need to compute it!
                    # parametric quantization
                    xmin = ps[p.replace("/Asize", "/Aquant/"+cfg.a_quantize.replace("_relu", "")+"/xmin")]
                    xmax = ps[p.replace("/Asize", "/Aquant/"+cfg.a_quantize.replace("_relu", "")+"/xmax")]

                    # ensure that dynamic ranges are in specified range and a power-of-two
                    xmin = quantize_pow2(clip_scalar(xmin, cfg.a_xmin_min, cfg.a_xmin_max))
                    xmax = quantize_pow2(clip_scalar(xmax, cfg.a_xmax_min, cfg.a_xmax_max))

                    # use ceil rounding
                    n = F.maximum_scalar(F.ceil(log2(log2(xmax/xmin) + 1.) + 1.), cfg.a_bitwidth_min)
                else:
                    raise ValueError("Unknown quantization method {}".format(cfg.a_quantize))
            else:
                # float precision
                n = nn.Variable((), need_grad=False)
                n.d = 32.

            kbytes.append(F.reshape(n * ps[p].d / 8. / 1024., (1,), inplace=False))

    if cfg.target_activation_type == 'max':
        _kbytes = F.max(F.concatenate(*kbytes))
    elif cfg.target_activation_type == 'sum':
        _kbytes = F.sum(F.concatenate(*kbytes))
    return num_activations, _kbytes


def categorical_error(pred, label):
    """
    Compute categorical error given score vectors and labels as
    numpy.ndarray.
    """
    pred_label = pred.argmax(1)
    return (pred_label != label.flat).mean()


def clip_quant_grads():
    ps = nn.get_parameters(grad_only=False)
    for p in ps:
        if ((p.endswith("quantized_conv/W") or
             p.endswith("quantized_conv/b") or
             p.endswith("quantized_affine/W") or
             p.endswith("quantized_affine/b"))):

            if cfg.w_quantize == 'parametric_fp_d_xmax':
                d = ps[p + "quant/"+cfg.w_quantize+"/d"]
                xmax = ps[p + "quant/"+cfg.w_quantize+"/xmax"]

                d.grad = F.clip_by_value(d.grad, -d.data, d.data)
                xmax.grad = F.clip_by_value(xmax.grad, -d.data, d.data)

            elif cfg.w_quantize == 'parametric_pow2_xmin_xmax':
                xmin = ps[p + "quant/"+cfg.w_quantize+"/xmin"]
                xmax = ps[p + "quant/"+cfg.w_quantize+"/xmax"]

                xmin.grad = F.clip_by_value(xmin.grad, -xmin.data, xmin.data)
                xmax.grad = F.clip_by_value(xmax.grad, -xmin.data, xmin.data)

        if 'Asize' in p.split('/'):
            if cfg.a_quantize == 'parametric_fp_d_xmax_relu':
                d = ps[p.replace("/Asize", "/Aquant/"+cfg.a_quantize.replace("_relu", "")+"/d")]
                xmax = ps[p.replace("/Asize", "/Aquant/"+cfg.a_quantize.replace("_relu", "")+"/xmax")]

                d.grad = F.clip_by_value(d.grad, -d.data, d.data)
                xmax.grad = F.clip_by_value(xmax.grad, -d.data, d.data)

            elif cfg.a_quantize == 'parametric_pow2_xmin_xmax_relu':
                xmin = ps[p.replace("/Asize", "/Aquant/"+cfg.a_quantize.replace("_relu", "")+"/xmin")]
                xmax = ps[p.replace("/Asize", "/Aquant/"+cfg.a_quantize.replace("_relu", "")+"/xmax")]

                xmin.grad = F.clip_by_value(xmin.grad, -xmin.data, xmin.data)
                xmax.grad = F.clip_by_value(xmax.grad, -xmin.data, xmin.data)


def clip_quant_vals():
    p = nn.get_parameters()
    if cfg.w_quantize in ['parametric_fp_b_xmax',
                          'parametric_fp_d_xmax',
                          'parametric_fp_d_b',
                          'parametric_pow2_b_xmax',
                          'parametric_pow2_b_xmin',
                          'parametric_pow2_xmin_xmax']:
        for k in p:
            if 'Wquant' in k.split('/') or 'bquant' in k.split('/'):
                if k.endswith('/m'):  # range
                    p[k].data = clip_scalar(p[k].data,
                                            cfg.w_dynrange_min + 1e-5,
                                            cfg.w_dynrange_max - 1e-5)
                elif k.endswith('/n'):  # bits
                    p[k].data = clip_scalar(p[k].data,
                                            cfg.w_bitwidth_min + 1e-5,
                                            cfg.w_bitwidth_max - 1e-5)
                elif k.endswith('/d'):  # delta
                    if cfg.w_quantize == 'parametric_fp_d_xmax':
                        g = k.replace('/d', '/xmax')
                        min_value = F.minimum2(p[k].data, p[g].data - 1e-5)
                        max_value = F.maximum2(p[k].data + 1e-5, p[g].data)
                        p[k].data = min_value
                        p[g].data = max_value
                    p[k].data = clip_scalar(p[k].data,
                                            cfg.w_stepsize_min + 1e-5,
                                            cfg.w_stepsize_max - 1e-5)
                elif k.endswith('/xmin'):  # xmin
                    if cfg.w_quantize == 'parametric_pow2_xmin_xmax':
                        g = k.replace('/xmin', '/xmax')
                        min_value = F.minimum2(p[k].data, p[g].data - 1e-5)
                        max_value = F.maximum2(p[k].data + 1e-5, p[g].data)
                        p[k].data = min_value
                        p[g].data = max_value
                    p[k].data = clip_scalar(p[k].data,
                                            cfg.w_xmin_min + 1e-5,
                                            cfg.w_xmin_max - 1e-5)
                elif k.endswith('/xmax'):  # xmax
                    p[k].data = clip_scalar(p[k].data,
                                            cfg.w_xmax_min + 1e-5,
                                            cfg.w_xmax_max - 1e-5)

    if cfg.a_quantize in ['parametric_fp_b_xmax_relu',
                          'parametric_fp_d_xmax_relu',
                          'parametric_fp_d_b_relu',
                          'parametric_pow2_b_xmax_relu',
                          'parametric_pow2_b_xmin_relu',
                          'parametric_pow2_xmin_xmax_relu']:
        for k in p:
            if 'Aquant' in k.split('/'):
                if k.endswith('/m'):  # range
                    p[k].data = clip_scalar(p[k].data,
                                            cfg.a_dynrange_min + 1e-5,
                                            cfg.a_dynrange_max - 1e-5)
                elif k.endswith('/n'):  # bits
                    p[k].data = clip_scalar(p[k].data,
                                            cfg.a_bitwidth_min + 1e-5,
                                            cfg.a_bitwidth_max - 1e-5)
                elif k.endswith('/d'):  # delta
                    if cfg.a_quantize == 'parametric_fp_d_xmax_relu':
                        g = k.replace('/d', '/xmax')
                        min_value = F.minimum2(p[k].data, p[g].data - 1e-5)
                        max_value = F.maximum2(p[k].data + 1e-5, p[g].data)
                        p[k].data = min_value
                        p[g].data = max_value
                    p[k].data = clip_scalar(p[k].data,
                                            cfg.a_stepsize_min + 1e-5,
                                            cfg.a_stepsize_max - 1e-5)
                elif k.endswith('/xmin'):  # xmin
                    if cfg.a_quantize == 'parametric_pow2_xmin_xmax_relu':
                        g = k.replace('/xmin', '/xmax')
                        min_value = F.minimum2(p[k].data, p[g].data - 1e-5)
                        max_value = F.maximum2(p[k].data + 1e-5, p[g].data)
                        p[k].data = min_value
                        p[g].data = max_value
                    p[k].data = clip_scalar(p[k].data,
                                            cfg.a_xmin_min + 1e-5,
                                            cfg.a_xmin_max - 1e-5)
                elif k.endswith('/xmax'):  # xmax
                    p[k].data = clip_scalar(p[k].data,
                                            cfg.a_xmax_min + 1e-5,
                                            cfg.a_xmax_max - 1e-5)


def train():
    """
    Main script.

    Steps:

    * Parse command line arguments.
    * Specify a context for computation.
    * Initialize DataIterator for CIFAR10.
    * Construct a computation graph for training and validation.
    * Initialize a solver and set parameter variables to it.
    * Training loop
      * Computate error rate for validation data (periodically)
      * Get a next minibatch.
      * Execute forwardprop on the training graph.
      * Compute training error
      * Set parameter gradients zero
      * Execute backprop.
      * Solver updates parameters by using gradients computed by backprop.
    """

    # define training parameters
    augmented_shift = True
    augmented_flip = True
    batch_size = 128
    vbatch_size = 100
    num_classes = 10
    weight_decay = 0.0002
    momentum = 0.9
    learning_rates = (cfg.initial_learning_rate,)*80 + \
        (cfg.initial_learning_rate / 10.,)*40 + \
        (cfg.initial_learning_rate / 100.,)*40
    print('lr={}'.format(learning_rates))
    print('weight_decay={}'.format(weight_decay))
    print('momentum={}'.format(momentum))

    # create nabla context
    from nnabla.ext_utils import get_extension_context
    ctx = get_extension_context('cudnn', device_id=args.gpu)
    nn.set_default_context(ctx)

    # Initialize DataIterator for CIFAR10.
    logger.info("Get CIFAR10 Data ...")
    data = cifar_data.DataIterator(batch_size, augmented_shift=augmented_shift,
                                   augmented_flip=augmented_flip)
    vdata = cifar_data.DataIterator(vbatch_size, val=True)

    if cfg.weightfile is not None:
        logger.info(f"Loading weights from {cfg.weightfile}")
        nn.load_parameters(cfg.weightfile)

    # TRAIN
    # Create input variables.
    image = nn.Variable([batch_size, 3, 32, 32])
    label = nn.Variable([batch_size, 1])

    # Create prediction graph.
    pred, hidden = resnet_cifar10(image,
                                  num_classes=num_classes,
                                  cfg=cfg,
                                  test=False)
    pred.persistent = True

    # Compute initial network size
    num_weights, kbytes_weights = network_size_weights()
    kbytes_weights.forward()
    print(f"Initial network size (weights) is {float(kbytes_weights.d):.3f}KB "
          f"(total number of weights: {int(num_weights):d}).")

    num_activations, kbytes_activations = network_size_activations()
    kbytes_activations.forward()
    print(f"Initial network size (activations) is {float(kbytes_activations.d):.3f}KB "
          f"(total number of activations: {int(num_activations):d}).")

    # Create loss function.
    cost_lambda2 = nn.Variable(())
    cost_lambda2.d = cfg.initial_cost_lambda2
    cost_lambda2.persistent = True
    cost_lambda3 = nn.Variable(())
    cost_lambda3.d = cfg.initial_cost_lambda3
    cost_lambda3.persistent = True

    loss1 = F.mean(F.softmax_cross_entropy(pred, label))
    loss1.persistent = True

    if cfg.target_weight_kbytes > 0:
        loss2 = F.relu(kbytes_weights - cfg.target_weight_kbytes) ** 2
        loss2.persistent = True
    else:
        loss2 = nn.Variable(())
        loss2.d = 0
        loss2.persistent = True
    if cfg.target_activation_kbytes > 0:
        loss3 = F.relu(kbytes_activations - cfg.target_activation_kbytes) ** 2
        loss3.persistent = True
    else:
        loss3 = nn.Variable(())
        loss3.d = 0
        loss3.persistent = True

    loss = loss1 + cost_lambda2 * loss2 + cost_lambda3 * loss3

    # VALID
    # Create input variables.
    vimage = nn.Variable([vbatch_size, 3, 32, 32])
    vlabel = nn.Variable([vbatch_size, 1])
    # Create predition graph.
    vpred, vhidden = resnet_cifar10(vimage,
                                    num_classes=num_classes,
                                    cfg=cfg,
                                    test=True)
    vpred.persistent = True

    # Create Solver.
    if cfg.optimizer== "adam":
        solver = S.Adam(alpha=learning_rates[0])
    else:
        solver = S.Momentum(learning_rates[0], momentum)

    solver.set_parameters(nn.get_parameters())

    # Training loop (epochs)
    logger.info("Start Training ...")
    i = 0
    best_v_err = 1.0

    # logs of the results
    iters = []
    res_train_err = []
    res_train_loss = []
    res_val_err = []

    # print all variables that exist
    for k in nn.get_parameters():
        print(k)

    res_n_b = collections.OrderedDict()
    res_n_w = collections.OrderedDict()
    res_n_a = collections.OrderedDict()
    res_d_b = collections.OrderedDict()
    res_d_w = collections.OrderedDict()
    res_d_a = collections.OrderedDict()
    res_xmin_b = collections.OrderedDict()
    res_xmin_w = collections.OrderedDict()
    res_xmin_a = collections.OrderedDict()
    res_xmax_b = collections.OrderedDict()
    res_xmax_w = collections.OrderedDict()
    res_xmax_a = collections.OrderedDict()

    for k in nn.get_parameters():
        if (k.split('/')[-1] == 'n') and (k.split('/')[-3] == 'bquant'):
            res_n_b[k] = []
    for k in nn.get_parameters():
        if (k.split('/')[-1] == 'n') and (k.split('/')[-3] == 'Wquant'):
            res_n_w[k] = []
    for k in nn.get_parameters():
        if (k.split('/')[-1] == 'n') and (k.split('/')[-3] == 'Aquant'):
            res_n_a[k] = []
    for k in nn.get_parameters():
        if (k.split('/')[-1] == 'd') and (k.split('/')[-3] == 'bquant'):
            res_d_b[k] = []
    for k in nn.get_parameters():
        if (k.split('/')[-1] == 'd') and (k.split('/')[-3] == 'Wquant'):
            res_d_w[k] = []
    for k in nn.get_parameters():
        if (k.split('/')[-1] == 'd') and (k.split('/')[-3] == 'Aquant'):
            res_d_a[k] = []
    for k in nn.get_parameters():
        if (k.split('/')[-1] == 'xmin') and (k.split('/')[-3] == 'bquant'):
            res_xmin_b[k] = []
    for k in nn.get_parameters():
        if (k.split('/')[-1] == 'xmin') and (k.split('/')[-3] == 'Wquant'):
            res_xmin_w[k] = []
    for k in nn.get_parameters():
        if (k.split('/')[-1] == 'xmin') and (k.split('/')[-3] == 'Aquant'):
            res_xmin_a[k] = []
    for k in nn.get_parameters():
        if (k.split('/')[-1] == 'xmax') and (k.split('/')[-3] == 'bquant'):
            res_xmax_b[k] = []
    for k in nn.get_parameters():
        if (k.split('/')[-1] == 'xmax') and (k.split('/')[-3] == 'Wquant'):
            res_xmax_w[k] = []
    for k in nn.get_parameters():
        if (k.split('/')[-1] == 'xmax') and (k.split('/')[-3] == 'Aquant'):
            res_xmax_a[k] = []

    for epoch in range(len(learning_rates)):
        train_loss = list()
        train_loss1 = list()
        train_loss2 = list()
        train_loss3 = list()
        train_err = list()

        # check whether we need to adapt the learning rate
        if epoch > 0 and learning_rates[epoch-1] != learning_rates[epoch]:
            solver.set_learning_rate(learning_rates[epoch])

        # Training loop (iterations)
        start_epoch = True
        while data.current != 0 or start_epoch:
            start_epoch = False
            # Next batch
            image.d, label.d = data.next()

            # Training forward/backward
            solver.zero_grad()

            loss.forward()
            loss.backward()

            if weight_decay is not None:
                solver.weight_decay(weight_decay)

            # scale gradients
            if cfg.target_weight_kbytes > 0 or cfg.target_activation_kbytes > 0:
                clip_quant_grads()

            solver.update()
            e = categorical_error(pred.d, label.d)
            train_loss += [loss.d]
            train_loss1 += [loss1.d]
            train_loss2 += [loss2.d]
            train_loss3 += [loss3.d]
            train_err += [e]

            # make sure that parametric values are clipped to correct values (if outside)
            clip_quant_vals()

            # Intermediate Validation (when constraint is set and fulfilled)
            kbytes_weights.forward()
            kbytes_activations.forward()
            if ((cfg.target_weight_kbytes > 0 and
                 (cfg.target_weight_kbytes <= 0 or float(kbytes_weights.d) <= cfg.target_weight_kbytes) and
                 (cfg.target_activation_kbytes <= 0 or float(kbytes_activations.d) <= cfg.target_activation_kbytes))):

                ve = list()
                start_epoch_ = True
                while vdata.current != 0 or start_epoch_:
                    start_epoch_ = False
                    vimage.d, vlabel.d = vdata.next()
                    vpred.forward()
                    ve += [categorical_error(vpred.d, vlabel.d)]

                v_err = np.array(ve).mean()
                if v_err < best_v_err: 
                    best_v_err = v_err
                    nn.save_parameters(os.path.join(cfg.params_dir, 'params_best.h5'))
                    print(f'Best validation error (fulfilling constraints: {best_v_err}')
                    sys.stdout.flush()
                    sys.stderr.flush()

            i += 1

        # Validation
        ve = list()
        start_epoch = True
        while vdata.current != 0 or start_epoch:
            start_epoch = False
            vimage.d, vlabel.d = vdata.next()
            vpred.forward()
            ve += [categorical_error(vpred.d, vlabel.d)]

        v_err = np.array(ve).mean()
        kbytes_weights.forward()
        kbytes_activations.forward()
        if ((v_err < best_v_err and
             (cfg.target_weight_kbytes <= 0 or float(kbytes_weights.d) <= cfg.target_weight_kbytes) and
             (cfg.target_activation_kbytes <= 0 or float(kbytes_activations.d) <= cfg.target_activation_kbytes))):
            best_v_err = v_err
            nn.save_parameters(os.path.join(cfg.params_dir, 'params_best.h5'))
            sys.stdout.flush()
            sys.stderr.flush()

        if cfg.target_weight_kbytes > 0:
            print(f"Current network size (weights) is {float(kbytes_weights.d):.3f}KB "
                  f"(#params: {int(num_weights)}, "
                  f"avg. bitwidth: {8. * 1024. * kbytes_weights.d / num_weights})")
            sys.stdout.flush()
            sys.stderr.flush()
        if cfg.target_activation_kbytes > 0:
            print(f"Current network size (activations) is {float(kbytes_activations.d):.3f}KB")
            sys.stdout.flush()
            sys.stderr.flush()

        for k in nn.get_parameters():
            if k.split('/')[-1] == 'n':
                print(f'{k}',
                      f'{nn.get_parameters()[k].d}',
                      f'{nn.get_parameters()[k].g}')
                sys.stdout.flush()
                sys.stderr.flush()
                if k.split('/')[-3] == 'bquant':
                    res_n_b[k].append(np.asscalar(nn.get_parameters()[k].d))
                elif k.split('/')[-3] == 'Wquant':
                    res_n_w[k].append(np.asscalar(nn.get_parameters()[k].d))
                elif k.split('/')[-3] == 'Aquant':
                    res_n_a[k].append(np.asscalar(nn.get_parameters()[k].d))

            elif k.split('/')[-1] == 'd':
                print(f'{k}',
                      f'{nn.get_parameters()[k].d}',
                      f'{nn.get_parameters()[k].g}')
                sys.stdout.flush()
                sys.stderr.flush()
                if k.split('/')[-3] == 'bquant':
                    res_d_b[k].append(np.asscalar(nn.get_parameters()[k].d))
                elif k.split('/')[-3] == 'Wquant':
                    res_d_w[k].append(np.asscalar(nn.get_parameters()[k].d))
                elif k.split('/')[-3] == 'Aquant':
                    res_d_a[k].append(np.asscalar(nn.get_parameters()[k].d))

            elif k.split('/')[-1] == 'xmin':
                print(f'{k}',
                      f'{nn.get_parameters()[k].d}',
                      f'{nn.get_parameters()[k].g}')
                sys.stdout.flush()
                sys.stderr.flush()
                if k.split('/')[-3] == 'bquant':
                    res_xmin_b[k].append(np.asscalar(nn.get_parameters()[k].d))
                elif k.split('/')[-3] == 'Wquant':
                    res_xmin_w[k].append(np.asscalar(nn.get_parameters()[k].d))
                elif k.split('/')[-3] == 'Aquant':
                    res_xmin_a[k].append(np.asscalar(nn.get_parameters()[k].d))

            elif k.split('/')[-1] == 'xmax':
                print(f'{k}',
                      f'{nn.get_parameters()[k].d}',
                      f'{nn.get_parameters()[k].g}')
                sys.stdout.flush()
                sys.stderr.flush()
                if k.split('/')[-3] == 'bquant':
                    res_xmax_b[k].append(np.asscalar(nn.get_parameters()[k].d))
                elif k.split('/')[-3] == 'Wquant':
                    res_xmax_w[k].append(np.asscalar(nn.get_parameters()[k].d))
                elif k.split('/')[-3] == 'Aquant':
                    res_xmax_a[k].append(np.asscalar(nn.get_parameters()[k].d))

        # Print
        logger.info(f'epoch={epoch}(iter={i}); '
                    f'overall cost={np.array(train_loss).mean()}; '
                    f'cross-entropy cost={np.array(train_loss1).mean()}; '
                    f'weight-size cost={np.array(train_loss2).mean()}; '
                    f'activations-size cost={np.array(train_loss3).mean()}; '
                    f'TrainErr={np.array(train_err).mean()}; '
                    f'ValidErr={v_err}; BestValidErr={best_v_err}')
        sys.stdout.flush()
        sys.stderr.flush()

        # update the logs
        iters.append(i)
        res_train_err.append(np.array(train_err).mean())
        res_train_loss.append([np.array(train_loss).mean(),
                               np.array(train_loss1).mean(),
                               np.array(train_loss2).mean(),
                               np.array(train_loss3).mean()])
        res_val_err.append(np.array(v_err).mean())
        res_ges = np.concatenate([np.array(iters)[:, np.newaxis],
                                  np.array(res_train_err)[:, np.newaxis],
                                  np.array(res_val_err)[:, np.newaxis],
                                  np.array(res_train_loss)], axis=-1)

        # save the results
        np.savetxt(cfg.params_dir + '/results.csv',
                   np.array(res_ges),
                   fmt='%10.8f',
                   header='iter,train_err,val_err,loss,loss1,loss2,loss3',
                   comments='',
                   delimiter=',')

        for rs, res in zip(['res_n_b.csv', 'res_n_w.csv', 'res_n_a.csv',
                            'res_d_b.csv', 'res_d_w.csv', 'res_d_a.csv',
                            'res_min_b.csv', 'res_min_w.csv', 'res_min_a.csv',
                            'res_max_b.csv', 'res_max_w.csv', 'res_max_a.csv'],
                           [res_n_b, res_n_w, res_n_a,
                            res_d_b, res_d_w, res_d_a,
                            res_xmin_b, res_xmin_w, res_xmin_a,
                            res_xmax_b, res_xmax_w, res_xmax_a]):
            res_mat = np.array([res[i] for i in res])
            if res_mat.shape[0] > 1 and res_mat.shape[1] > 1:
                np.savetxt(cfg.params_dir + '/' + rs,
                           np.array([[i, j, res_mat[i, j]] for i, j in product(range(res_mat.shape[0]), range(res_mat.shape[1]))]),
                           fmt='%10.8f',
                           comments='',
                           delimiter=',')


if __name__ == '__main__':
    train()
