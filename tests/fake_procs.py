"""psutil.Process look-alikes for process discovery tests."""

from types import SimpleNamespace

import psutil


class FakeProc:
    def __init__(self, pid, ppid, name, cmdline, rss=100 * 2**20, environ=None,
                 cpu=5.0, threads=8, swap=0, deny_environ=False, has_full_info=True):
        self.pid = pid
        self._ppid = ppid
        self._name = name
        self._cmdline = cmdline
        self._rss = rss
        self._environ = environ or {}
        self._cpu = cpu
        self._threads = threads
        self._swap = swap
        self._deny = deny_environ
        self._full = has_full_info

    def name(self):
        return self._name

    def cmdline(self):
        return list(self._cmdline)

    def ppid(self):
        return self._ppid

    def environ(self):
        if self._deny:
            raise psutil.AccessDenied(self.pid)
        return dict(self._environ)

    def memory_info(self):
        return SimpleNamespace(rss=self._rss)

    def memory_full_info(self):
        if not self._full:
            raise psutil.AccessDenied(self.pid)
        return SimpleNamespace(rss=self._rss, swap=self._swap)

    def cpu_percent(self, interval=None):
        return self._cpu

    def num_threads(self):
        return self._threads

    def create_time(self):
        return 1000.0 + self.pid

    def is_running(self):
        return True
