import os

openvino_enabled = False
try:
    from openvino.inference_engine import IECore
    ie = IECore()
    openvino_enabled = True
except:
    pass


class D3NetOpenVinoWrapper(object):
    def __init__(self, args, source):
        if not openvino_enabled:
            raise ValueError(
                'Failed to import openvino! Please make sure you have installed openvino.')
        weight = os.path.join(
            args.model_dir, source + '.onnx')
        if not os.path.exists(weight):
            raise ValueError(
                '{} does not exist. Please download weight file beforehand. You can see README.md for the detail.'.format(
                    weight))
        self.net = ie.read_network(model=weight)
        self.input_blob = next(iter(self.net.input_info))
        self.out_blob = next(iter(self.net.outputs))
        self.exec_net = ie.load_network(network=self.net, device_name='CPU', config={
                                        'CPU_THREADS_NUM': str(args.cpu_number)})

    def run(self, input_var):
        output = self.exec_net.infer(inputs={self.input_blob: input_var})
        return output[list(output.keys())[0]]
