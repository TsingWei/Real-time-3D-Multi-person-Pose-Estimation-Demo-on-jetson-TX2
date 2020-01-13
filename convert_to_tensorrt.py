import tensorrt as trt
TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
    builder.max_workspace_size = 1 << 30 
    builder.max_batch_size = 1 
    builder.fp16_mode = True 
    with open('human-pose-estimation-3d.onnx', 'rb') as model: 
        parser.parse(model.read()) 
    engine = builder.build_cuda_engine(network) 
    with open('human-pose-estimation-3d.trt', "wb") as f: 
        f.write(engine.serialize())