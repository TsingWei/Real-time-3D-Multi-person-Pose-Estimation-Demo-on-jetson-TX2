import torch
import tensorrt as trt
def simple_conv1d():
    class TestNet(torch.nn.Module):
        def __init__(self):
            super(TestNet, self).__init__()
            self.conv = torch.nn.Conv1d(2, 4, kernel_size=(1,))
            return
        def forward(self, x):
            res = self.conv(x)
            return res
    x = torch.randn(3, 2, 10)
    net = TestNet()
    onnx_file_path = "simple_conv1d.onnx"
    torch.onnx.export(net, x, onnx_file_path, export_params=True, verbose=True, input_names=["data"])
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    TRT_LOGGER.Severity(TRT_LOGGER.VERBOSE)
    with trt.Builder(TRT_LOGGER) as builder:
        with builder.create_network() as network:
            with trt.OnnxParser(network,TRT_LOGGER) as parser:
                with open(onnx_file_path, 'rb') as model:
                    parser.parse(model.read())
                builder.max_workspace_size = 1<<30
                builder.fp16_mode = False
                builder.max_batch_size = 32
                builder.strict_type_constraints = False
                engine = builder.build_cuda_engine(network)
                with open("simple_conv1d.trt", "wb") as f:
                    f.write(engine.serialize())
    return
simple_conv1d()