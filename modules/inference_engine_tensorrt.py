import numpy as np
import torch
import tensorrt as trt
import tqdm
import common

# import pycuda.driver as cuda
# This import causes pycuda to automatically manage CUDA context creation and cleanup.
# import pycuda.autoinit

class InferenceEngineTensorRT:
    def __init__(self, trt_model_path, device,
                 img_mean=np.array([128, 128, 128], dtype=np.float32),
                 img_scale=np.float32(1/255)):
        assert device=='GPU', 'Only supports GPU.'
        TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
        # runtime = trt.Runtime(TRT_LOGGER)
        with open(trt_model_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.img_mean = img_mean
        self.img_scale = img_scale
        # self.inputs, self.outputs, self.bindings, self.stream = self._allocate_buffers(self.engine)
        self.inputs, self.outputs, self.bindings, self.stream = common.allocate_buffers(self.engine)
        self.context = self.engine.create_execution_context()

        # # Determine dimensions and create page-locked memory buffers (i.e. won't be swapped to disk) to hold host inputs/outputs.
        # self.h_input = cuda.pagelocked_empty(trt.volume(self.engine.get_binding_shape(0)), dtype=np.float32)
        # self.h_output = cuda.pagelocked_empty(trt.volume(self.engine.get_binding_shape(1)), dtype=np.float32)
		# # Allocate device memory for inputs and outputs.
        # self.d_input = cuda.mem_alloc(self.h_input.nbytes)
        # self.d_output = cuda.mem_alloc(self.h_output.nbytes)
		# # Create a stream in which to copy inputs/outputs and run inference.
        # self.stream = cuda.Stream()
        # self.stream.synchronize()
        # # print(self.engine.is_execution_binding())
        # # print(self.engine.is_shape_binding())
        # import time
        # print("[TensorRT] Wait for deserialize...")
        # from tqdm import trange
        # for i in trange(10):
        #     time.sleep(1)
        # # time.sleep(450)
        # print("[TensorRT] TRT model deserialize done.")

    def infer(self, img):
        normalized_img = InferenceEngineTensorRT._normalize(img, self.img_mean, self.img_scale)
        input_tensor = np.ascontiguousarray(np.expand_dims(normalized_img.transpose(2, 0, 1),axis=0))
        # print(input_tensor)
        self.inputs[0].host = input_tensor

        # Transfer input data to the GPU.
        # cuda.memcpy_htod_async(self.inputs[0].device, self.inputs[0].host, self.stream)
        # # [cuda.memcpy_htod_async(inp.device, inp.host, self.stream) for inp in self.inputs]
        # # Run inference.
        # self.context.execute_async(batch_size=1, bindings=self.bindings, stream_handle=self.stream.handle)
        # # Transfer predictions back from the GPU.
        # [cuda.memcpy_dtoh_async(out.host, out.device, self.stream) for out in self.outputs]
        trt_outputs = common.do_inference(self.context, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs, stream=self.stream)
        # Synchronize the stream
        self.stream.synchronize()
        # Return only the host outputs.
        output_shapes = [(19,32,56), (38,32,56), (57,32,56)]
        # rtn_outputs = tuple([trt_outputs[0].host.reshape([19,32,56]),
        #                     trt_outputs[1].host.reshape([38,32,56]),
        #                     trt_outputs[2].host.reshape([57,32,56])])
        trt_outputs = [output.reshape(shape) for output, shape in zip(trt_outputs, output_shapes)]
        # print(trt_outputs[0])
        features, heatmaps, pafs= trt_outputs[2], trt_outputs[0], trt_outputs[1]
        # print(features.squeeze().shape)
        # print(features)
        # print(features.shape)
        # print(heatmaps.shape)
        # print(pafs.shape)
        # print(features[-1].squeeze().shape)
        return (features.squeeze(),
                heatmaps.squeeze(), pafs.squeeze())
        # return [out.host for out in self.outputs]

        # with self.engine.create_execution_context() as context:
        #     # Transfer input data to the GPU.
        #     cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
        #     # Run inference.
        #     context.execute_async(bindings=[int(self.d_input), int(self.d_output)], stream_handle=self.stream.handle)
        #     # Transfer predictions back from the GPU.
        #     cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        #     # Synchronize the stream
        #     self.stream.synchronize()
		#     # Return the host output. 
        # return self.h_output

        # normalized_img = InferenceEnginePyTorch._normalize(img, self.img_mean, self.img_scale)
        # data = torch.from_numpy(normalized_img).permute(2, 0, 1).unsqueeze(0).to(self.device)

        # features, heatmaps, pafs = self.net(data)

        # return (features[-1].squeeze().data.cpu().numpy(),
        #         heatmaps[-1].squeeze().data.cpu().numpy(), pafs[-1].squeeze().data.cpu().numpy())

    @staticmethod
    def _normalize(img, img_mean, img_scale):
        normalized_img = (img.astype(np.float32) - img_mean) * img_scale
        return normalized_img

    # Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
    @staticmethod
    def _allocate_buffers(engine):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            print(size)
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(device_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        return inputs, outputs, bindings, stream

# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

