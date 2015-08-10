from __future__ import print_function
from builtins import str

import sys
import platform as p
import uuid
import hashlib


def fingerprint():
    md5 = hashlib.md5()
    # Hostname, OS, CPU, MAC,
    data = [p.node(), p.system(), p.machine(), str(uuid.getnode())]
    md5.update(''.join(data).encode('utf8'))
    return "%s-pygraphistry-%s" % (md5.hexdigest()[:8], sys.modules['graphistry'].__version__)


def in_ipython():
        try:
            __IPYTHON__
            return True
        except NameError:
            return False

def warn(msg):
    if in_ipython:
        import IPython
        IPython.utils.warn.warn(msg)
    else:
        print('WARNING: ', msg, file=sys.stderr)

def error(msg):
    if in_ipython:
        import IPython
        IPython.utils.warn.error(msg)
    raise ValueError(msg)
