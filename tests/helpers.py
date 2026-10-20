"""Helpers shared by the tests."""


class LibSpy:
    """Wraps the library proxy and records every call made through it."""

    def __init__(self, real):
        self._real = real
        self.calls = []

    def __getattr__(self, name):
        attr = getattr(self._real, name)
        if not callable(attr):
            return attr

        def call(*args):
            self.calls.append((name, args[1:]))  # drop the network pointer
            return attr(*args)

        return call

    def names(self):
        return [name for name, _ in self.calls]

    def args(self, name):
        return [a for n, a in self.calls if n == name]
