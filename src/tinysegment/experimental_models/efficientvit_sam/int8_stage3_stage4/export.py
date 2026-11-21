import torch
from torch import nn
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config, XNNPACKQuantizer,
)
from executorch.exir import to_edge_transform_and_lower
from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e


class HalfBlock(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block.half()

    def forward(self, x):
        return self.block(x.half()).float()


def describe_graph(edge):
    graph = edge.exported_program()
    ops = [{'name': n.name, 'operator': str(n.target)} for n in graph.graph.nodes
           if n.op == 'call_function']
    delegates = []
    for module in graph.graph_module.modules():
        if not hasattr(module, 'backend_id'):
            continue
        nodes = []
        for n in module.original_module.graph.nodes:
            if n.op != 'call_function':
                continue
            val = n.meta.get('val')
            nodes.append({'operator': str(n.target), 'dtype': str(getattr(val, 'dtype', '')),
                          'shape': list(val.shape) if isinstance(val, torch.Tensor) else None})
        delegates.append({'backend': module.backend_id, 'nodes': nodes})
    return {'execution_order': ops, 'delegates': delegates}


def export_block(name, block, x, calibration):
    if name == 'fp16':
        block = HalfBlock(block).eval()
    if name == 'int8':
        graph = torch.export.export(block, (x,)).module()
        config = get_symmetric_quantization_config(is_per_channel=True)
        quantizer = XNNPACKQuantizer().set_global(config)
        prepared = prepare_pt2e(graph, quantizer)
        for real_input in calibration:
            prepared(real_input)
        block = convert_pt2e(prepared)
    ep = torch.export.export(block, (x,))
    edge = to_edge_transform_and_lower(ep, partitioner=[XnnpackPartitioner()])
    graph = describe_graph(edge)
    program = edge.to_executorch()
    return program, graph
