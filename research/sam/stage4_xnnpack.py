"""Export and inspect the three XNNPACK programs used by the stage 4 benchmark."""
from pathlib import Path
from statistics import median
import tempfile

import torch
from torch import nn
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config, XNNPACKQuantizer,
)
from executorch.devtools.etdump.serialize import deserialize_from_etdump_flatcc
from executorch.exir import to_edge_transform_and_lower
from executorch.runtime import Runtime
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


def export_block(name, block, x, calibration, directory):
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
    def serialize(path):
        with path.open('wb') as stream:
            program.write_to_file(stream)
    return directory.write(f'{name}.pte', serialize), graph


def trace(path, x):
    # XNNPACK profiles its native operators after input conversion/packing and layout
    # preparation. These timings avoid standalone subgraphs that select different kernels.
    program = Runtime.get().load_program(path, enable_etdump=True)
    method = program.load_method('forward')
    for _ in range(6):
        method.execute((x,))
    with tempfile.TemporaryDirectory() as tmp:
        dump = Path(tmp) / 'profile.etdp'
        program.write_etdump_result_to_file(str(dump), str(Path(tmp) / 'debug.bin'))
        data = deserialize_from_etdump_flatcc(dump.read_bytes())
    runs = []
    for section in data.run_data:
        if section.name != 'Execute':
            continue
        events = []
        for event in section.events or []:
            p = event.profile_event
            if p:
                events.append({'operator': p.delegate_debug_id_str or p.name,
                               'backend': 'XNNPACK' if p.delegate_debug_id_str else 'ExecuTorch',
                               'ms': (p.end_time - p.start_time) / 1e6})
        runs.append(events)
    if len(runs) != 6:
        raise RuntimeError(f'Expected six profiled calls, got {len(runs)}')
    kernels = {'expansion': [], 'depthwise': [], 'projection': []}
    for events in runs[1:]:
        convs = [e for e in events if e['backend'] == 'XNNPACK' and
                 ('Convolution' in e['operator'] or 'GEMM' in e['operator'])]
        if len(convs) != 3:
            raise RuntimeError(f'Expected three profiled XNNPACK convolutions, found {convs}')
        for key, event in zip(kernels, convs, strict=True):
            kernels[key].append(event['ms'])
    return {'runtime_trace_ms': runs[1:],
            'prepared_kernel_median_ms': {k: median(v) for k, v in kernels.items()},
            'native_conv_operators': [e['operator'] for e in convs]}
