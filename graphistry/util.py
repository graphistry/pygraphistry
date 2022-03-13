import hashlib, logging, os, platform as p, random, string, sys, uuid, warnings
from distutils.version import LooseVersion, StrictVersion


def cmp(x, y):
    return (x > y) - (x < y)


def make_iframe(url, height, extra_html = "", override_html_style = None):
    id = uuid.uuid4()

    height_str = f'{height}px' if isinstance(height, int) or isinstance(height, float) else str(height)

    scrollbug_workaround = '''
            <script>
                try {
                  $("#%s").bind('mousewheel', function(e) { e.preventDefault(); });
                } catch (e) { console.error('exn catching scroll', e); }
            </script>
        ''' % id

    style = None
    if override_html_style is not None:
        style = override_html_style
    else:
        style = "width:100%%; height:%s; border: 1px solid #DDD; overflow: hidden" % height_str

    iframe = '''
            <iframe id="%s" src="%s"
                    allowfullscreen="true" webkitallowfullscreen="true" mozallowfullscreen="true"
                    oallowfullscreen="true" msallowfullscreen="true"
                    style="%s"
                    %s
            >
            </iframe>
        ''' % (id, url, style, extra_html)

    return iframe + scrollbug_workaround


def fingerprint():
    md5 = hashlib.md5()
    # Hostname, OS, CPU, MAC,
    data = [p.node(), p.system(), p.machine(), str(uuid.getnode())]
    md5.update(''.join(data).encode('utf8'))

    from ._version import get_versions
    __version__ = get_versions()['version']

    return "%s-pygraphistry-%s" % (md5.hexdigest()[:8], __version__)


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
        if hasattr(__builtins__, '__IPYTHON__'):
            return True
    except NameError:
        pass
    try:
        from IPython import get_ipython
        cfg = get_ipython()
        if not (cfg is None) and ('IPKernelApp' in get_ipython().config):
            return True
    except ImportError:
        pass
    return False


def in_databricks():
    # FIXME: this is a hack
    if 'DATABRICKS_RUNTIME_VERSION' in os.environ:
        return True
    return False


def warn(msg):
    try:
        if in_ipython():
            import IPython
            IPython.utils.warn.warn(msg)
            return
    except:
        'ok'
    warnings.warn(RuntimeWarning(msg))


def error(msg):
    raise ValueError(msg)


def merge_two_dicts(a, b):
    c = a.copy()
    c.update(b)
    return c


def deprecated(message):
    """
      Marks a method as deprecated.

      :param message: Info regarding the deprecation.
    """

    def deprecated_decorator(func):
        def deprecated_func(*args, **kwargs):
            warnings.warn("{} is a deprecated function. {}".format(func.__name__, message),
                          category = DeprecationWarning,
                          stacklevel = 2)
            warnings.simplefilter('default', DeprecationWarning)
            return func(*args, **kwargs)

        return deprecated_func

    return deprecated_decorator
