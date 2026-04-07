"""CLI decorator — consistent KeyboardInterrupt handling across all scripts."""

import functools
import os
import sys


def cli(fn):
    @functools.wraps(fn)
    def wrapper(*a, **kw):
        try:
            return fn(*a, **kw)
        except KeyboardInterrupt:
            sys.stderr.write("\n")
            sys.stderr.flush()
            os._exit(130)
    return wrapper
