"""Single-slot input for producers that can outrun inference."""
from threading import Condition, Thread


class LatestInput:
    def __init__(self, values, initial=None):
        self._changed = Condition()
        self._value = initial
        self._version = 0
        self._closed = False
        self._error = None
        self._reader = Thread(target=self._read, args=(values,), daemon=True)
        self._reader.start()

    def _read(self, values):
        try:
            for value in values:
                with self._changed:
                    if self._closed:
                        break
                    self._value = value
                    self._version += 1
                    self._changed.notify_all()
        except Exception as exc:
            with self._changed:
                self._error = exc
        finally:
            self.close()

    def close(self):
        with self._changed:
            self._closed = True
            self._changed.notify_all()

    def join(self):
        self._reader.join()

    def current(self):
        with self._changed:
            if self._error:
                raise self._error
            return self._value

    def __iter__(self):
        version = 0
        while True:
            with self._changed:
                self._changed.wait_for(lambda: self._version != version or self._closed)
                if self._version == version:
                    if self._error:
                        raise self._error
                    return
                value, version = self._value, self._version
            yield value


def latest_prompts(lines, initial=None):
    prompts = (line.strip() for line in lines)
    return LatestInput((prompt for prompt in prompts if prompt), initial)
