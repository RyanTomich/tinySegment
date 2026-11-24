"""Offline Moonshine Tiny Streaming English baseline."""
from collections import deque
from pathlib import Path

from ...weights import default_weights_dir


def load_transcriber(*, weights_dir=None):
    """Preload the local checkpoint. The caller owns the returned transcriber."""
    model_dir = Path(weights_dir or default_weights_dir()) / 'moonshine' / 'tiny-streaming-en'
    if not (model_dir / 'streaming_config.json').is_file():
        raise FileNotFoundError(f'Missing local Moonshine Tiny Streaming checkpoint in {model_dir}')

    from moonshine_voice import ModelArch, Transcriber
    transcriber = Transcriber(model_dir, ModelArch.TINY_STREAMING, options={'keyterm_boost': 5})
    transcriber.set_keyterms(['Reachy'])
    return transcriber


def completed_lines(transcriber, audio_chunks, sample_rate):
    """Stream mono float samples in [-1, 1] and flush the last line at EOF."""
    from moonshine_voice.transcriber import Error, LineCompleted

    pending = deque()

    def collect(event):
        if isinstance(event, (LineCompleted, Error)):
            pending.append(event)

    def drain():
        while pending:
            event = pending.popleft()
            if isinstance(event, Error):
                raise event.error
            yield event.line.text

    with transcriber.create_stream() as stream:
        stream.add_listener(collect)
        stream.start()
        for chunk in audio_chunks:
            if len(chunk):
                stream.add_audio(chunk, sample_rate)
                yield from drain()
        stream.stop()
        yield from drain()
