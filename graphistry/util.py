from __future__ import print_function
from past.builtins import cmp
from builtins import str

import sys
import platform as p
import uuid
import hashlib
import random
import string
from distutils.version import LooseVersion, StrictVersion


def make_iframe(raw_url, height, protocol=None):
    id = uuid.uuid4()
    scrollbug_workaround='''<script>
            $("#%s").bind('mousewheel', function(e) {
                e.preventDefault();
            });
        </script>''' % id

    if protocol:
        return '''<iframe id="%s" src="%s"
                          style="width:100%%; height:%dpx; border: 1px solid #DDD">
                </iframe>''' % (id, protocol + ':' + raw_url, height) + scrollbug_workaround

    script = '''<script>var p = document.location.protocol;
                        if(p === "file:") {p = "http:";}
                        $("#%s").attr("src", p + "%s").show();
                </script>''' % (id, raw_url)
    iframe = '''<iframe id="%s"
                        style="display:none; width:100%%; height:%dpx; border: 1px solid #DDD">
                </iframe>''' % (id, height)
    return iframe + script + scrollbug_workaround


def fingerprint():
    md5 = hashlib.md5()
    # Hostname, OS, CPU, MAC,
    data = [p.node(), p.system(), p.machine(), str(uuid.getnode())]
    md5.update(''.join(data).encode('utf8'))
    return "%s-pygraphistry-%s" % (md5.hexdigest()[:8], sys.modules['graphistry'].__version__)


def random_string(length):
    gibberish = [random.choice(string.ascii_uppercase + string.digits) for _ in range(length)]
    return ''.join(gibberish)


def compare_versions(v1, v2):
    try:
        return cmp(StrictVersion(v1), StrictVersion(v2))
    except ValueError:
        return cmp(LooseVersion(v1), LooseVersion(v2))


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
