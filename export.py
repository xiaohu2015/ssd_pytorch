import os
from posixpath import pardir
import types
import argparse
import logging
import sys

import torch
import torchvision
import torch.nn.functional as F
import onnx_graphsurgeon as gs
import onnx
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from PIL import Image
from torchvision.transforms import functional as F

from visualize_utils import draw_bounding_boxes
from models.transform import ImageList
import models

assert trt.__version__ >= "8.0"

logging.basicConfig(level=logging.INFO)
logging.getLogger("EngineBuilder").setLevel(logging.INFO)
logger = logging.getLogger("EngineBuilder")

class EngineBuilder:
    """
    Parses an ONNX graph and builds a TensorRT engine from it.
    """

    def __init__(self, max_workspace_size, verbose=False):
        """
        :param max_workspace_size: GB
        :param verbose: If enabled, a higher verbosity level will be set on the TensorRT logger.
        """
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        if verbose:
            self.trt_logger.min_severity = trt.Logger.Severity.VERBOSE

        trt.init_libnvinfer_plugins(self.trt_logger, namespace="")

        self.builder = trt.Builder(self.trt_logger)
        self.config = self.builder.create_builder_config()
        self.config.max_workspace_size = max_workspace_size * (2 ** 30)  # 8 GB

        self.batch_size = None
        self.network = None
        self.parser = None

    def create_network(self, onnx_path):
        """
        Parse the ONNX graph and create the corresponding TensorRT network definition.
        :param onnx_path: The path to the ONNX graph to load.
        """
        network_flags = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

        self.network = self.builder.create_network(network_flags)
        self.parser = trt.OnnxParser(self.network, self.trt_logger)

        onnx_path = os.path.realpath(onnx_path)
        with open(onnx_path, "rb") as f:
            if not self.parser.parse(f.read()):
                logger.error("Failed to load ONNX file: {}".format(onnx_path))
                for error in range(self.parser.num_errors):
                    logger.error(self.parser.get_error(error))
                sys.exit(1)

        inputs = [self.network.get_input(i) for i in range(self.network.num_inputs)]
        outputs = [self.network.get_output(i) for i in range(self.network.num_outputs)]

        logger.info("Network Description")
        for input in inputs:
            self.batch_size = input.shape[0]
            logger.info("Input '{}' with shape {} and dtype {}".format(input.name, input.shape, input.dtype))
        for output in outputs:
            logger.info("Output '{}' with shape {} and dtype {}".format(output.name, output.shape, output.dtype))
        assert self.batch_size > 0
        self.builder.max_batch_size = self.batch_size

    def create_engine(self, engine_path, precision):
        """
        Build the TensorRT engine and serialize it to disk.
        :param engine_path: The path where to serialize the engine to.
        :param precision: The datatype to use for the engine, either 'fp32', 'fp16'.
        """
        engine_path = os.path.realpath(engine_path)
        engine_dir = os.path.dirname(engine_path)
        os.makedirs(engine_dir, exist_ok=True)
        logger.info("Building {} Engine in {}".format(precision, engine_path))

        inputs = [self.network.get_input(i) for i in range(self.network.num_inputs)]

        if precision == "fp16":
            if not self.builder.platform_has_fast_fp16:
                logger.warning("FP16 is not supported natively on this platform/device")
            else:
                self.config.set_flag(trt.BuilderFlag.FP16)

        with self.builder.build_serialized_network(self.network, self.config) as engine, open(engine_path, "wb") as f:
            logger.info("Serializing engine to file: {:}".format(engine_path))
            f.write(engine)


class TensorRTInfer:
    """
    Implements inference for the TensorRT engine.
    """

    def __init__(self, engine_path):
        """
        :param engine_path: The path to the serialized engine to load from disk.
        """
        # Load TRT engine
        self.logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(self.logger, namespace="")
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        assert self.engine
        assert self.context

        # Setup I/O bindings
        self.inputs = []
        self.outputs = []
        self.allocations = []
        for i in range(self.engine.num_bindings):
            is_input = False
            if self.engine.binding_is_input(i):
                is_input = True
            name = self.engine.get_binding_name(i)
            dtype = self.engine.get_binding_dtype(i)
            shape = self.engine.get_binding_shape(i)
            if is_input:
                self.batch_size = shape[0]
            size = np.dtype(trt.nptype(dtype)).itemsize
            for s in shape:
                size *= s
            allocation = cuda.mem_alloc(size)
            binding = {
                'index': i,
                'name': name,
                'dtype': np.dtype(trt.nptype(dtype)),
                'shape': list(shape),
                'allocation': allocation,
            }
            self.allocations.append(allocation)
            if self.engine.binding_is_input(i):
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)

        assert self.batch_size > 0
        assert len(self.inputs) > 0
        assert len(self.outputs) > 0
        assert len(self.allocations) > 0

    def input_spec(self):
        """
        Get the specs for the input tensor of the network. Useful to prepare memory allocations.
        :return: Two items, the shape of the input tensor and its (numpy) datatype.
        """
        return self.inputs[0]['shape'], self.inputs[0]['dtype']

    def output_spec(self):
        """
        Get the specs for the output tensors of the network. Useful to prepare memory allocations.
        :return: A list with two items per element, the shape and (numpy) datatype of each output tensor.
        """
        specs = []
        for o in self.outputs:
            specs.append((o['shape'], o['dtype']))
        return specs

    def infer(self, batch):
        """
        Execute inference on a batch of images. The images should already be batched and preprocessed, as prepared by
        the ImageBatcher class. Memory copying to and from the GPU device will be performed here.
        :param batch: A numpy array holding the image batch.
        :return: the output for each image in the batch.
        """
        # Prepare the output data
        outputs = []
        for shape, dtype in self.output_spec():
            outputs.append(np.zeros(shape, dtype))

        # Process I/O and execute the network
        cuda.memcpy_htod(self.inputs[0]['allocation'], np.ascontiguousarray(batch))
        self.context.execute_v2(self.allocations)
        for o in range(len(outputs)):
            cuda.memcpy_dtoh(outputs[o], self.outputs[o]['allocation'])

        return outputs


def forward(self, images):
    # normalize image
    dtype, device = images.dtype, images.device
    mean = torch.as_tensor(self.transform.image_mean, dtype=dtype, device=device)
    std = torch.as_tensor(self.transform.image_std, dtype=dtype, device=device)

    images = (images / torch.tensor(255., dtype=dtype, device=device) \
        - mean[None, :, None, None]) / std[None, :, None, None]

    # convet to ImageList
    image_sizes = [img.shape[-2:] for img in images]
    images = ImageList(images, image_sizes)

    features = self.backbone(images.tensors)
    features = list(features.values())

    # compute the ssd heads outputs using the features
    head_outputs = self.head(features)

    # create the set of anchors
    anchors = self.anchor_generator(images, features)
    anchors = torch.cat(anchors, dim=0) # cat [N, HWA, 4]

    bbox_regression = head_outputs["bbox_regression"] # [N, HWA, 4]
    pred_scores = F.softmax(head_outputs["cls_logits"], dim=-1) # [N, HWA, C]
    #scores = pred_scores[:, :, 1:] # remove background [batch_size, number_boxes, C]
    scores = pred_scores # [batch_size, number_boxes, C+1]

    # decode box
    N = bbox_regression.size(0)
    boxes = self.box_coder.decode_single(bbox_regression.reshape(-1, 4),
        anchors.reshape(-1, 4))
    # normalize box
    height, width = image_sizes[0]
    wh = torch.tensor([width, height, width, height], dtype=boxes.dtype, device=boxes.device)
    boxes = boxes / wh
    boxes = boxes.reshape(N, -1, 1, 4) # [batch_size, number_boxes, 1, 4]
    return boxes, scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", help="The output ONNX model file to write")
    parser.add_argument("--onnx-nms", help="The output ONNX plus NMS model file to write")
    parser.add_argument("--engine", help="The tensorrt engine file to write")
    parser.add_argument("--score-threshold", type=float, default=0.5, help="the score threshold for the NMS")
    parser.add_argument("--nms-threshold", type=float, default=0.5, help="the nms threshold for the NMS")
    parser.add_argument("--nms-detections", type=int, default=100, help="the max detections for the NMS op")
    parser.add_argument("--workspace", type=int, default=4, help="tensorrt max workspace")
    parser.add_argument("--precision", default="fp32", choices=["fp32", "fp16"],
        help="The precision mode to build in, either 'fp32', 'fp16', default: 'fp32'")
    parser.add_argument("--input-image", default="data/dog.jpg", help="the test image file")
    args = parser.parse_args()

    # export to onnx
    if not os.path.exists(args.onnx):
        batch_size = 1
        dummy_input = torch.randn(batch_size, 3, 300, 300, device='cuda')
        model = models.__dict__["ssd300_vgg16"](pretrained=True).cuda()
        model.forward = types.MethodType(forward, model)
        model.eval()

        torch.onnx.export(model,
            dummy_input, 
            args.onnx,
            verbose=True,
            input_names=["images"],
            output_names=["boxes", "scores"],
            opset_version=11,
        )

    if not os.path.exists(args.onnx_nms):
        # add nms plugin
        graph = gs.import_onnx(onnx.load(args.onnx))
        nms_inputs = graph.outputs
        nms_op = "BatchedNMS_TRT"
        nms_attrs = {
            "plugin_version": "1",
            "shareLocation": True,
            "backgroundLabelId": 0,
            "numClasses": 91,
            "topK": 1024,
            "keepTopK": args.nms_detections,
            "scoreThreshold": args.score_threshold,
            "iouThreshold": args.nms_threshold,
            "isNormalized": True,
            "clipBoxes": True,
            }
        # NMS Outputs
        nms_output_num_detections = gs.Variable(name="num_detections", dtype=np.int32, shape=[batch_size, 1])
        nms_output_boxes = gs.Variable(name="detection_boxes", dtype=np.float32,
                                           shape=[batch_size, args.nms_detections, 4])
        nms_output_scores = gs.Variable(name="detection_scores", dtype=np.float32,
                                            shape=[batch_size, args.nms_detections])
        nms_output_classes = gs.Variable(name="detection_classes", dtype=np.float32,
                                             shape=[batch_size, args.nms_detections])

        nms_outputs = [nms_output_num_detections, nms_output_boxes, nms_output_scores, nms_output_classes]

        # Create the NMS Plugin node with the selected inputs. The outputs of the node will also become the final
        # outputs of the graph.
        graph.layer(
            op=nms_op,
            name="nms",
            inputs=nms_inputs,
            outputs=nms_outputs,
            attrs=nms_attrs)
        graph.outputs = nms_outputs
        # save onnx
        graph.cleanup().toposort()
        model = gs.export_onnx(graph)
        onnx.save(model, args.onnx_nms)

    if not os.path.exists(args.engine):
        builder = EngineBuilder(args.workspace, True)
        builder.create_network(args.onnx_nms)
        builder.create_engine(args.engine, args.precision)

    # do trt inference
    trt_infer = TensorRTInfer(args.engine)
    image_ori = Image.open(args.input_image)
    image = image_ori.resize((300, 300))
    images = np.expand_dims(np.array(image, dtype=np.float32).transpose(2, 0, 1), 0)

    det_outputs = trt_infer.infer(images)

    image_tensor = F.pil_to_tensor(image_ori)
    _, height, width = image_tensor.shape
    num_detections = int(det_outputs[0])
    boxes = torch.as_tensor(det_outputs[1][0, :num_detections] * np.array([width, height, width, height]))
    scores = det_outputs[2][0, :num_detections].tolist()
    labels = det_outputs[3][0, :num_detections].astype(np.int64).tolist()

    coco_class_names = []
    for l in open('data/coco.categories').readlines():
        coco_class_names.append(l.split(',')[0])
    labels = ["{} | {:.2f}".format(coco_class_names[idx], score) for idx, score in zip(labels, scores)]

    out_image = draw_bounding_boxes(image_tensor, boxes, labels, colors=(0, 0, 255), width=2, font='DejaVuSans', font_size=20)
    out_image = Image.fromarray(out_image.numpy().transpose(1, 2, 0))
    out_image.save("det.jpg")
