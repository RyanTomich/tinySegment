"""Export and inspect the three XNNPACK programs used by the stage 4 benchmark."""
from pathlib import Path
from statistics import median
import tempfile

from executorch.devtools.etdump.serialize import deserialize_from_etdump_flatcc
from executorch.runtime import Runtime

from tinysegment.experimental_models.efficientvit_sam.int8_stage3_stage4.export import export_block as compile_block


def export_block(name, block, x, calibration, directory):
    program, graph = compile_block(name, block, x, calibration)

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
