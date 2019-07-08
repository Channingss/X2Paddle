import os
import sys
import subprocess

SHARED_CAFFE_RESOLVER = None


def import_caffepb():
   p = os.path.realpath(__file__)
    p = os.path.dirname(p)
    p = os.path.join(p, '../../proto')
    sys.path.insert(0, p)
    s = sys.version
    if s.startswith('2'):
        import commands
        pb_version = commands.getstatusoutput('protoc --version')[1]
    else:
        import subprocess
        pb_version = subprocess.getstatusoutput('protoc --version')[1]
    ver_str = pb_version.split(' ')[-1].replace('.', '')
    ver_int = int(ver_str)
    assert ver_int >= 360, 'The version of protobuf must be larger than 3.6.0!'
    import caffe_pb2
    return caffe_pb2


class CaffeResolver(object):
    def __init__(self):
        self.import_caffe()

    def import_caffe(self):
        self.caffe = None
        try:
            # Try to import PyCaffe first
            import caffe
            self.caffe = caffe
        except ImportError:
            # Fall back to the protobuf implementation
            self.caffepb = import_caffepb()
            show_fallback_warning()
        if self.caffe:
            # Use the protobuf code from the imported distribution.
            # This way, Caffe variants with custom layers will work.
            self.caffepb = self.caffe.proto.caffe_pb2
        self.NetParameter = self.caffepb.NetParameter

    def has_pycaffe(self):
        return self.caffe is not None


def get_caffe_resolver():
    global SHARED_CAFFE_RESOLVER
    if SHARED_CAFFE_RESOLVER is None:
        SHARED_CAFFE_RESOLVER = CaffeResolver()
    return SHARED_CAFFE_RESOLVER


def has_pycaffe():
    return get_caffe_resolver().has_pycaffe()


def show_fallback_warning():
    msg = '''
------------------------------------------------------------
    WARNING: PyCaffe not found!
    Falling back to a pure protocol buffer implementation.
    * Conversions will be drastically slower.
------------------------------------------------------------

'''
    sys.stderr.write(msg)
