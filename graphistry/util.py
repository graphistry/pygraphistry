from __future__ import print_function
from builtins import str

import sys
import platform as p
import uuid
import hashlib

def make_iframe(raw_url, height, protocol=None):
    if protocol:
        return '''<iframe src="%s"
                          style="width:100%%; height:%dpx; border: 1px solid #DDD">
                </iframe>''' % (protocol + ':' + raw_url, height)

    id = uuid.uuid4()
    script = '''<script>var p = document.location.protocol;
                        if(p === "file:") {p = "http:";}
                        $("#%s").attr("src", p + "%s").show();
                </script>''' % (id, raw_url)
    iframe = '''<iframe id="%s"
                        style="display:none; width:100%%; height:%dpx; border: 1px solid #DDD">
                </iframe>''' % (id, height)
    return iframe + script

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
