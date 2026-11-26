import torch
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config, XNNPACKQuantizer,
)
from executorch.exir import to_edge_transform_and_lower
from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e


def export_block(block, calibration):
    example = (calibration[0],)
    graph = torch.export.export(block, example).module()
    config = get_symmetric_quantization_config(is_per_channel=True)
    quantizer = XNNPACKQuantizer().set_global(config)
    prepared = prepare_pt2e(graph, quantizer)
    for real_input in calibration:
        prepared(real_input)
    ep = torch.export.export(convert_pt2e(prepared), example)
    edge = to_edge_transform_and_lower(ep, partitioner=[XnnpackPartitioner()])
    return edge.to_executorch()
