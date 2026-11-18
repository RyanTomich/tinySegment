"""Dense model MACs for one encoded image and one single-mask box prompt."""
import torch
from fvcore.nn.jit_analysis import JitModelAnalysis
from fvcore.nn import jit_handles as handles

from .evaluate import comparison

# Only dense convolution and tensor contractions contribute to this metric.
MAC_HANDLES = {
    'aten::_convolution': handles.conv_flop_jit,
    'aten::linear': handles.linear_flop_jit,
    'aten::addmm': handles.addmm_flop_jit,
    'aten::bmm': handles.bmm_flop_jit,
    'aten::matmul': handles.matmul_flop_jit,
    'aten::mm': handles.matmul_flop_jit,
    'aten::einsum': handles.einsum_flop_jit,
}
EXCLUDED = set('add add_ sub mul mul_ div batch_norm layer_norm group_norm gelu relu6 hardtanh '
               'softmax sigmoid sin cos pow sqrt mean sum reciprocal cumsum repeat_interleave '
               'repeat clone square pad upsample_bicubic2d upsample_bilinear2d fill_ ones_like'.split())


def count_macs(module, inputs):
    trace = JitModelAnalysis(module, inputs).set_op_handle(**MAC_HANDLES)
    trace.unsupported_ops_warnings(False).uncalled_modules_warnings(False)
    total = int(trace.total())
    uncounted = dict(trace.unsupported_ops())
    unknown = sorted(op for op in uncounted if op.removeprefix('aten::') not in EXCLUDED)
    return total, {'uncounted': uncounted, 'unclassified': unknown, 'complete': not unknown}


class BoxDecoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.prompt_encoder = model.prompt_encoder
        self.mask_decoder = model.mask_decoder

    def forward(self, features, box):
        sparse, dense = self.prompt_encoder(points=None, boxes=box, masks=None)
        return self.mask_decoder(image_embeddings=features, image_pe=self.prompt_encoder.get_dense_pe(),
                                 sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
                                 multimask_output=False)


def profile(stage):
    model = stage.model
    image = torch.zeros(1, 3, 512, 512)
    box = torch.tensor([[0., 0., 512., 512.]])
    encoder, enc_ops = count_macs(model.image_encoder, (image,))
    decoder, dec_ops = count_macs(BoxDecoder(model), (model.image_encoder(image), box))
    values = {'parameters': sum(p.numel() for p in model.parameters()),
              'checkpoint_mib': stage.checkpoint.stat().st_size / 2**20,
              'tensor_payload_mib': sum(t.numel() * t.element_size()
                                        for t in (*model.parameters(), *model.buffers())) / 2**20,
              'encoder_gmac': encoder / 1e9, 'prompt_decoder_gmac': decoder / 1e9,
              'total_gmac': (encoder + decoder) / 1e9}
    for name in ('image_encoder', 'prompt_encoder', 'mask_decoder'):
        values[f'{name}_parameters'] = sum(p.numel() for p in getattr(model, name).parameters())
    return values, {'encoder': enc_ops, 'prompt_decoder': dec_ops}


def size_ops(stages):
    (original, original_ops), (candidate, candidate_ops) = map(profile, stages)
    return {'input_shape': [1, 3, 512, 512], 'boxes': 1, 'multimask_output': False,
            'comparison': {k: comparison(v, candidate[k]) for k, v in original.items()},
            'coverage': {'original': original_ops, 'candidate': candidate_ops},
            'complete': all(part['complete'] for ops in (original_ops, candidate_ops) for part in ops.values())}
