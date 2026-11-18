import tempfile
import unittest
from pathlib import Path

import torch
from torch.ao.quantization.observer import MinMaxObserver

from research.sam.int8_neck import install_neck, load_candidate, neck_layers, quantize_filters, save_candidate


@unittest.skipUnless('qnnpack' in torch.backends.quantized.supported_engines, 'requires QNNPACK')
class Int8NeckTests(unittest.TestCase):
    def test_filter_scales_and_zero_filter(self):
        weights = torch.tensor([[0., 0.], [-1., 0.5], [0., 127.]]).reshape(3, 1, 1, 2)
        result = quantize_filters(weights)
        self.assertEqual(result.q_per_channel_axis(), 0)
        torch.testing.assert_close(result.q_per_channel_scales().float(), torch.tensor([1., 1/127, 1.]))
        self.assertTrue(torch.all(result.int_repr()[0] == 0))
        self.assertLessEqual((weights - result.dequantize()).abs().max().item(), 0.5/127 + 1e-7)

    @torch.inference_mode()
    def test_serialized_neck_executes_eight_integer_convolutions(self):
        from efficientvit.sam_model_zoo import create_efficientvit_sam_model
        previous_engine = torch.backends.quantized.engine
        previous_threads = torch.get_num_threads()
        # Some CPU builds start with 'none' but only permit selecting actual engines.
        if previous_engine in torch.backends.quantized.supported_engines:
            self.addCleanup(setattr, torch.backends.quantized, 'engine', previous_engine)
        self.addCleanup(torch.set_num_threads, previous_threads)
        torch.backends.quantized.engine = 'qnnpack'
        torch.set_num_threads(1)
        model = create_efficientvit_sam_model('efficientvit-sam-l0', pretrained=False).eval()
        observer = MinMaxObserver()
        observer(torch.tensor([-2., 2.]))
        install_neck(model, [(observer, observer)] * 8)
        x = torch.rand(1, 256, 8, 8)
        expected = model.image_encoder.neck.middle(x)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / '.pending-test'
            save_candidate(model, path)
            loaded = load_candidate(path).model
        for layer in neck_layers(loaded):
            self.assertEqual(layer.conv[1].weight().dtype, torch.qint8)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
            actual = loaded.image_encoder.neck.middle(x)
        counts = {event.key: event.count for event in profile.key_averages()}
        self.assertEqual(counts.get('quantized::conv2d'), 8)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
