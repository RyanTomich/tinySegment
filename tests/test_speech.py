import unittest
from threading import Event
from unittest.mock import patch

from tinysegment.pipeline import latest_speech_prompts, speech_prompts


class SpeechPromptTests(unittest.TestCase):
    @patch('tinysegment.baseline_models.moonshine.completed_lines')
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
        completed.assert_called_once_with(model, chunks, 16000)

    @patch('tinysegment.baseline_models.moonshine.completed_lines')
    def test_previous_prompt_persists_until_command_has_words(self, completed):
        waiting, resume = Event(), Event()

        def conversation():
            yield 'Reachy, water bottle.'
            yield 'ordinary conversation'
            yield 'Reachy!'
            waiting.set()
            resume.wait(5)
            yield 'green backpack'

        completed.return_value = conversation()
        prompts = latest_speech_prompts(object(), iter(()), initial='wallet')
        try:
            self.assertTrue(waiting.wait(5))
            self.assertEqual(prompts.current(), 'water bottle')
        finally:
            resume.set()
            prompts.join()
        self.assertEqual(prompts.current(), 'green backpack')
