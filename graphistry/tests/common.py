import unittest
import graphistry
import graphistry.pygraphistry



class NoAuthTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        graphistry.pygraphistry.PyGraphistry._is_authenticated = True
        graphistry.register(api=2)
