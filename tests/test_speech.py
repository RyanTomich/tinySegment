import unittest
from importlib.util import find_spec
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, call, patch

from tinysegment.pipeline import speech_prompts


class SpeechPromptTests(unittest.TestCase):
    @patch('models.moonshine_tiny_streaming_en.baseline.loader.completed_lines')
    def test_wake_boundaries_repeated_commands_and_word_limit(self, completed):
        completed.return_value = iter([
            'Reachyville and preachy are ordinary words.',
            'Oh, REACHY! my USB-C cable. Reachy, my red backpack.',
            'Reachy one two three four five six seven eight nine ten eleven twelve',
            'ignore this until Reachy',
            '... ',
            "Sam’s glasses",
            'Reachy!',
        ])
        model, chunks = object(), iter(())
        self.assertEqual(list(speech_prompts(model, chunks)), [
            'my USB-C cable', 'my red backpack',
            'one two three four five six seven eight nine ten', 'Sam’s glasses',
        ])
        completed.assert_called_once_with(model, chunks, 16000, on_text=None)

    @patch('models.moonshine_tiny_streaming_en.baseline.loader.completed_lines')
    def test_previous_prompt_persists_until_command_has_words(self, completed):
        completed.return_value = ['Reachy, water bottle.', 'ordinary conversation', 'Reachy!',
                                  'green backpack']
        self.assertEqual(list(speech_prompts(object(), ())), ['water bottle', 'green backpack'])

    @unittest.skipUnless(find_spec('moonshine_voice'), 'requires the optional speech runtime')
    def test_live_revisions_reach_preview_before_the_final_command(self):
        from moonshine_voice.transcriber import LineCompleted, LineTextChanged

        transcriber = MagicMock()
        stream = transcriber.create_stream.return_value.__enter__.return_value
        on_text = Mock()

        def emit(event_type, text):
            listener = stream.add_listener.call_args.args[0]
            listener(event_type(line=SimpleNamespace(line_id=7, text=text), stream_handle=1))

        stream.add_audio.side_effect = lambda *_: emit(LineTextChanged, 'Reachy, cab')
        stream.stop.side_effect = lambda: emit(LineCompleted, 'Reachy, cat.')

        def chunks():
            yield [0.1]
            on_text.assert_called_once_with(7, 'Reachy, cab')

        self.assertEqual(list(speech_prompts(transcriber, chunks(), on_text=on_text)), ['cat'])
        self.assertEqual(on_text.call_args_list, [call(7, 'Reachy, cab'), call(7, 'Reachy, cat.')])
