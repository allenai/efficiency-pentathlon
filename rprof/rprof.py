import os
import platform
import sysconfig
import sys
import ctypes as ct

def load_library(basename):
    if platform.system() == 'Windows':
       ext = '.dll'
    else:
       ext = sysconfig.get_config_var('SO')
    return ct.cdll.LoadLibrary(os.path.join(
        os.path.dirname(__file__) or os.path.curdir,
        basename + ext))

_rprof = load_library('librprof')

def start_profiling(profile_interval, timeout):
    assert profile_interval > 0
    assert timeout > 0
    return _rprof.rprof(profile_interval, timeout)