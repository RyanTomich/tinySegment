"""Offline Moonshine Tiny Streaming English baseline."""
from collections import deque
from pathlib import Path

from models.storage import default_weights_dir


def load_transcriber(*, weights_dir=None):
    """Preload the local checkpoint. The caller owns the returned transcriber."""
    model_dir = Path(weights_dir or default_weights_dir()) / 'moonshine_tiny_streaming_en' / 'baseline' / 'artifacts'
    if not (model_dir / 'streaming_config.json').is_file():
        raise FileNotFoundError(f'Missing local Moonshine Tiny Streaming checkpoint in {model_dir}. Run tinysegment setup-weights --speech.')

    from moonshine_voice import ModelArch, Transcriber
    transcriber = Transcriber(model_dir, ModelArch.TINY_STREAMING, options={'keyterm_boost': 5})
    transcriber.set_keyterms(['Reachy'])
    return transcriber


def completed_lines(transcriber, audio_chunks, sample_rate, *, on_text=None):
    """Yield final lines and optionally report live (line ID, text) revisions."""
    from moonshine_voice.transcriber import Error, LineCompleted, LineStarted, LineTextChanged

    pending = deque()
    events = (LineCompleted, Error) if on_text is None else (LineStarted, LineTextChanged, LineCompleted, Error)

    def collect(event):
        if isinstance(event, events):
            pending.append(event)

    def drain():
        while pending:
            event = pending.popleft()
            if isinstance(event, Error):
                raise event.error
            if on_text is not None:
                on_text(event.line.line_id, event.line.text)
            if isinstance(event, LineCompleted):
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
