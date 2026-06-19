# Copyright 2025 University of California, Davis, and Kai Fukazawa.
# SPDX-License-Identifier: Apache-2.0
"""Small experiment logger owned by the RAMAC codebase."""

from __future__ import annotations

import csv
import datetime
import json
import os
import os.path as osp
import pickle
import shutil
from enum import Enum

from tabulate import tabulate


def _is_json_safe(data):
    if data is None:
        return True
    if isinstance(data, (bool, int, float, str)):
        return True
    if isinstance(data, (tuple, list)):
        return all(_is_json_safe(item) for item in data)
    if isinstance(data, dict):
        return all(isinstance(key, str) and _is_json_safe(value) for key, value in data.items())
    return False


def _to_json_safe(data):
    if isinstance(data, dict):
        return {
            key: value if _is_json_safe(value) else _to_json_safe(value)
            for key, value in data.items()
        }
    if isinstance(data, (tuple, list)):
        return [item if _is_json_safe(item) else _to_json_safe(item) for item in data]
    if _is_json_safe(data):
        return data
    return str(data)


class _Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, type):
            return {'$class': f'{obj.__module__}.{obj.__name__}'}
        if isinstance(obj, Enum):
            return {'$enum': f'{obj.__module__}.{obj.__class__.__name__}.{obj.name}'}
        if callable(obj):
            return {'$function': f'{obj.__module__}.{obj.__name__}'}
        return super().default(obj)


class _TerminalTablePrinter:
    def __init__(self):
        self.headers = None
        self.rows = []

    def print_tabular(self, tabular_rows):
        if self.headers is None:
            self.headers = [key for key, _ in tabular_rows]
        self.rows.append([value for _, value in tabular_rows])
        self.refresh()

    def refresh(self):
        rows = max(shutil.get_terminal_size((120, 30)).lines, 6)
        visible_rows = self.rows[-(rows - 3):]
        print('\033[2J\033[H', end='')
        print(tabulate(visible_rows, self.headers))


class RunLogger:
    def __init__(self):
        self.reset()

    def reset(self):
        self._prefixes = []
        self._prefix_str = ''
        self._tabular = []
        self._text_outputs = []
        self._tabular_outputs = []
        self._text_fds = {}
        self._tabular_fds = {}
        self._snapshot_dir = None
        self._snapshot_mode = 'last'
        self._snapshot_gap = 1
        self._log_tabular_only = False
        self._header_printed = False
        self.table_printer = _TerminalTablePrinter()

    def push_prefix(self, prefix: str):
        self._prefixes.append(prefix)
        self._prefix_str = ''.join(self._prefixes)

    def log(self, message: str):
        line = self._prefix_str + str(message)
        print(line, flush=True)
        for fd in self._text_fds.values():
            fd.write(line + '\n')
            fd.flush()

    def _add_output(self, file_name, outputs, fds, mode='a'):
        if file_name in outputs:
            return
        directory = osp.dirname(file_name)
        if directory:
            os.makedirs(directory, exist_ok=True)
        outputs.append(file_name)
        fds[file_name] = open(file_name, mode, encoding='utf-8', newline='')

    def add_text_output(self, file_name):
        self._add_output(file_name, self._text_outputs, self._text_fds, mode='a')

    def add_tabular_output(self, file_name, relative_to_snapshot_dir: bool = False):
        if relative_to_snapshot_dir and self._snapshot_dir is not None:
            file_name = osp.join(self._snapshot_dir, file_name)
        self._add_output(file_name, self._tabular_outputs, self._tabular_fds, mode='w')

    def set_snapshot_dir(self, directory: str):
        self._snapshot_dir = directory

    def set_snapshot_mode(self, mode: str):
        self._snapshot_mode = mode

    def set_snapshot_gap(self, gap: int):
        self._snapshot_gap = gap

    def set_log_tabular_only(self, flag: bool):
        self._log_tabular_only = flag

    def record_tabular(self, key, value):
        self._tabular.append((key, value))

    def dump_tabular(self):
        if not self._tabular:
            return
        if not self._log_tabular_only:
            self.table_printer.print_tabular(self._tabular)

        headers = [key for key, _ in self._tabular]
        values = [value for _, value in self._tabular]
        for fd in self._tabular_fds.values():
            writer = csv.writer(fd)
            if not self._header_printed:
                writer.writerow(headers)
            writer.writerow(values)
            fd.flush()
        self._header_printed = True
        self._tabular.clear()

    def log_variant(self, log_file, variant_data):
        with open(log_file, 'w', encoding='utf-8') as handle:
            json.dump(_to_json_safe(variant_data), handle, indent=2, sort_keys=True, cls=_Encoder)

    def save_itr_params(self, itr, params):
        if self._snapshot_dir is None or self._snapshot_mode == 'none':
            return
        if self._snapshot_mode == 'all':
            file_name = osp.join(self._snapshot_dir, f'itr_{itr}.pkl')
        elif self._snapshot_mode == 'gap':
            if itr % self._snapshot_gap != 0:
                return
            file_name = osp.join(self._snapshot_dir, f'itr_{itr}.pkl')
        elif self._snapshot_mode == 'last':
            file_name = osp.join(self._snapshot_dir, 'params.pkl')
        else:
            raise ValueError(f'Unknown snapshot mode: {self._snapshot_mode}')
        with open(file_name, 'wb') as handle:
            pickle.dump(params, handle)


logger = RunLogger()


def create_exp_name(exp_prefix, exp_id=0, seed=0):
    timestamp = datetime.datetime.now().astimezone().strftime('%Y_%m_%d_%H_%M_%S')
    return f'{exp_prefix}_{timestamp}_{exp_id:04d}--s-{seed}'


def create_log_dir(exp_prefix, exp_id=0, seed=0, base_log_dir=None, include_exp_prefix_sub_dir=True):
    exp_name = create_exp_name(exp_prefix, exp_id=exp_id, seed=seed)
    base_log_dir = base_log_dir or './data'
    if include_exp_prefix_sub_dir:
        log_dir = osp.join(base_log_dir, exp_prefix.replace('_', '-'), exp_name)
    else:
        log_dir = osp.join(base_log_dir, exp_name)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def setup_logger(
    exp_prefix='default',
    variant=None,
    text_log_file='debug.log',
    variant_log_file='variant.json',
    tabular_log_file='progress.csv',
    snapshot_mode='last',
    snapshot_gap=1,
    log_tabular_only=False,
    log_dir=None,
    git_infos=None,
    script_name=None,
    **create_log_dir_kwargs,
):
    del git_infos
    first_time = log_dir is None
    if first_time:
        log_dir = create_log_dir(exp_prefix, **create_log_dir_kwargs)

    if variant is not None:
        logger.log('Variant:')
        logger.log(json.dumps(_to_json_safe(variant), indent=2))
        logger.log_variant(osp.join(log_dir, variant_log_file), variant)

    logger.add_text_output(osp.join(log_dir, text_log_file))
    if first_time:
        logger.add_tabular_output(osp.join(log_dir, tabular_log_file))
    else:
        logger._add_output(
            osp.join(log_dir, tabular_log_file),
            logger._tabular_outputs,
            logger._tabular_fds,
            mode='a',
        )
        logger._header_printed = True

    logger.set_snapshot_dir(log_dir)
    logger.set_snapshot_mode(snapshot_mode)
    logger.set_snapshot_gap(snapshot_gap)
    logger.set_log_tabular_only(log_tabular_only)
    logger.push_prefix(f'[{osp.basename(log_dir)}] ')

    if script_name is not None:
        with open(osp.join(log_dir, 'script_name.txt'), 'w', encoding='utf-8') as handle:
            handle.write(script_name)
    return log_dir
