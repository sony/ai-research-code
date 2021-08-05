import os
import nnabla as nn
from nnabla.ext_utils import get_extension_context
from model import d3_net

openvino_enabled = False
try:
    from openvino.inference_engine import IECore
    ie = IECore()
    openvino_enabled = True
except:
    pass


def load_parameters(filename):
    params = {}
    with nn.parameter_scope('', scope=params):
        nn.load_parameters(filename)
    return params


class D3NetOpenVinoWrapper(object):
    def __init__(self, args, hparams, source):
        if not openvino_enabled:
            raise ValueError(
                'Failed to import openvino! Please make sure you have installed openvino.')
        weight = os.path.join(
            args.openvino_model_dir, source + '.onnx')
        if not os.path.exists(weight):
            raise ValueError(
                '{} does not exist. Please download weight file beforehand. You can see README.md for the detail.'.format(weight))
        self.net = ie.read_network(model=weight)
        self.input_blob = next(iter(self.net.input_info))
        self.out_blob = next(iter(self.net.outputs))
        self.exec_net = ie.load_network(network=self.net, device_name='CPU')

    def run(self, input_var):
        output = self.exec_net.infer(inputs={self.input_blob: input_var})
        return output[list(output.keys())[0]]


class D3Net(nn.Module):
    def __init__(self, hparams):
        self.hparams = hparams

    def call(self, x):
        return d3_net(x, self.hparams, test=True)


class D3NetNNablaWrapper(object):
    def __init__(self, args, hparams, source):
        self.x = None
        self.out_ = None
        self.d3net = D3Net(hparams)
        self.d3net.training = False
        params = load_parameters(args.model)
        self.d3net.set_parameters(params[source])

    def run(self, input_var):
        if not self.x:
            self.x = nn.Variable(input_var.shape)
            self.out_ = self.d3net(self.x)
        self.x.d = input_var
        self.out_.forward(clear_buffer=True)
        out_ = self.out_.data.data
        return out_


class SourceSeparationModel(object):
    def __init__(self, args, hparams, source):
        backend = args.backend
        assert backend in ['nnabla', 'openvino']
        self.wrappers = {
            'nnabla': D3NetNNablaWrapper,
            'openvino': D3NetOpenVinoWrapper
        }
        self.d3netwrapper = self.wrappers[backend](args, hparams, source)

    def run(self, input_var):
        return self.d3netwrapper.run(input_var)
