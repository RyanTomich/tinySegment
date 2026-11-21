"""Render the component medians from a TinySegment research report as Markdown."""
import argparse
import json
from pathlib import Path
from statistics import median

COMPONENTS = {
    'jpeg': 'Open and decode JPEG',
    'text': 'CLIP tokenization and text encoding',
    'detection': 'YOLO-World detection',
    'other_request': 'SAM preparation and remaining request overhead',
    'image_encoder': '**SAM image encoder**',
    'predict_box': 'SAM first box to full-resolution mask',
}
OPERATORS = {
    'expansion': '1×1 expansion',
    'gelu_1': 'First GELU',
    'depthwise': '3×3 depthwise convolution',
    'gelu_2': 'Second GELU',
    'projection': '1×1 projection',
    'batch_norm': 'Projection BatchNorm',
}


def component_medians(summary, samples):
    detail = summary['profiled_encode_prompt']
    values = {key: detail[key]['median'] for key in COMPONENTS if key in detail}
    if 'other_request' not in values:
        # Older reports omit this field. Derive the residual for each individual request.
        values['other_request'] = median(
            row['total'] - sum(row[k] for k in COMPONENTS if k != 'other_request')
            for row in samples['profiled_encode_prompt']
        )
    return values


def print_table(headers, rows):
    for row in (headers, ['---'] * len(headers), *rows):
        print('| ' + ' | '.join(row) + ' |')


def render(report):
    if report.get('status') == 'failed':
        raise ValueError(f"Benchmark failed: {report.get('error', 'see report')}")
    if 'original_profile_median_ms' in report:
        values = report['original_profile_median_ms']
        print_table(['Operation', 'Median'], [[label, f'{values[k]:.2f} ms'] for k, label in OPERATORS.items()])
        return
    summary = report['summary_ms']
    samples = report['samples_ms']
    if 'candidate_model' in report or 'original' in summary:
        keys = (report.get('control_model', 'original'), report.get('candidate_model', 'stage4_int8'))
        labels = {'original': 'Original SAM', 'stage4_int8': 'Stage 4 INT8',
                  'stage3_stage4_int8': 'Research SAM (stages 3 + 4 INT8)'}
        columns = [component_medians(summary[k], samples[k]) for k in keys]
        names = [labels[k] for k in keys]
    else:
        columns = [component_medians(summary, samples)]
        names = ['Original SAM']
    rows = [[label, *(f'{col[k]:.1f} ms' for col in columns)] for k, label in COMPONENTS.items()]
    print_table(['JPEG to first mask, median', *names], rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    args = parser.parse_args()
    try:
        render(json.loads(args.report.read_text()))
    except (KeyError, OSError, ValueError) as err:
        parser.exit(1, f'Cannot render {args.report}: {err}\n')
