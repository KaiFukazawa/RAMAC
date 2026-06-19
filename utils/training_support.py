# Copyright 2025 University of California, Davis, and Kai Fukazawa.
# SPDX-License-Identifier: Apache-2.0
"""Minimal training loop helpers owned by the RAMAC codebase."""

from __future__ import annotations

import math
import time


def print_banner(message: str, separator: str = '-', num_star: int = 60) -> None:
    line = separator * num_star
    print(line, flush=True)
    print(message, flush=True)
    print(line, flush=True)


class SimpleProgress:
    def __init__(
        self,
        total,
        name='Progress',
        ncol=3,
        max_length=20,
        indent=0,
        line_width=100,
        speed_update_freq=100,
    ):
        self.total = total
        self.name = name
        self.ncol = ncol
        self.max_length = max_length
        self.indent = indent
        self.line_width = line_width
        self.speed_update_freq = speed_update_freq

        self.step = 0
        self.skip_lines = 1
        self.prev_line = '\033[F'
        self.clear_line = ' ' * self.line_width

        self.pbar_size = self.ncol * self.max_length
        self.complete = '#' * self.pbar_size
        self.incomplete = ' ' * self.pbar_size

        self.lines = ['']
        self.fraction = f'0 / {self.total}'
        self.speed = '0.0 Hz'
        self.resume()

    def update(self, description, n=1):
        self.step += n
        if self.step % self.speed_update_freq == 0:
            self.time0 = time.time()
            self.step0 = self.step
        self.set_description(description)

    def resume(self):
        self.skip_lines = 1
        print('\n', end='')
        self.time0 = time.time()
        self.step0 = self.step

    def pause(self):
        self._clear()
        self.skip_lines = 1

    def set_description(self, params=None):
        params = params or []
        if isinstance(params, dict):
            params = sorted(params.items())

        self._clear()
        percent, fraction = self._format_percent(self.step, self.total)
        self.fraction = fraction
        speed = self._format_speed(self.step)
        num_rows = math.ceil(len(params) / self.ncol) if params else 0
        params_string, lines = self._format(self._chunk(params, self.ncol))
        self.lines = lines
        print(f'{percent} | {speed}{params_string}')
        self.skip_lines = num_rows + 1

    def append_description(self, description):
        self.lines.append(description)

    def stamp(self):
        if self.lines != ['']:
            params = ' | '.join(self.lines)
            summary = f'[ {self.name} ] {self.fraction}{params} | {self.speed}'
            self._clear()
            print(summary)
            self.skip_lines = 1
        else:
            self._clear()
            self.skip_lines = 0

    def _clear(self):
        position = self.prev_line * self.skip_lines
        empty = '\n'.join(self.clear_line for _ in range(self.skip_lines))
        print(position, end='')
        print(empty)
        print(position, end='')

    def _format_percent(self, current, total):
        if total:
            percent = current / float(total)
            complete_entries = int(percent * self.pbar_size)
            incomplete_entries = self.pbar_size - complete_entries
            pbar = self.complete[:complete_entries] + self.incomplete[:incomplete_entries]
            fraction = f'{current} / {total}'
            return f'{fraction} [{pbar}] {int(percent * 100):3d}%', fraction
        return f'{current} iterations', str(current)

    def _format_speed(self, current):
        num_steps = current - self.step0
        elapsed = max(time.time() - self.time0, 1e-12)
        if num_steps > 0:
            self.speed = f'{num_steps / elapsed:.1f} Hz'
        return self.speed

    def _chunk(self, values, n):
        return [values[i:i + n] for i in range(0, len(values), n)]

    def _format(self, chunks):
        lines = [self._format_chunk(chunk) for chunk in chunks]
        lines.insert(0, '')
        padding = '\n' + ' ' * self.indent
        return padding.join(lines), lines

    def _format_chunk(self, chunk):
        return ' | '.join(self._format_param(param) for param in chunk)

    def _format_param(self, param):
        key, value = param
        return f'{key} : {value}'[:self.max_length]

    def close(self):
        self.pause()


class NoOpProgress:
    def __init__(self, *args, **kwargs):
        del args, kwargs

    def __getattr__(self, _attr):
        return lambda *args, **kwargs: None


class EarlyStopping:
    def __init__(self, tolerance: int = 5, min_delta: float = 0.0):
        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0

    def __call__(self, train_loss: float, validation_loss: float) -> bool:
        if (validation_loss - train_loss) > self.min_delta:
            self.counter += 1
            return self.counter >= self.tolerance
        self.counter = 0
        return False


Progress = SimpleProgress
Silent = NoOpProgress
