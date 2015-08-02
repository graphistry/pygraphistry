import sys
import platform as p
import uuid
import hashlib


def fingerprint():
    md5 = hashlib.md5()
    # Hostname, OS, CPU, MAC,
    data = [p.node(), p.system(), p.machine(), str(uuid.getnode())]
    md5.update(''.join(data))
    return "%s-pygraphistry-%s" % (md5.hexdigest()[:8], sys.modules['graphistry'].__version__)


def in_ipython():
        try:
            __IPYTHON__
            return True
        except NameError:
            return False
