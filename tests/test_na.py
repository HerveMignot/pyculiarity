import os

import numpy as np
import pandas as pd

from pyculiarity import detect_ts, detect_vec
from unittest import TestCase


def eq_(a, b, msg=None):
    """Shorthand for 'assert a == b, "%r != %r" % (a, b)
    """
    if not a == b:
        raise AssertionError(msg or "%r != %r" % (a, b))


def make_decorator(func):
    """
    Wraps a test decorator so as to properly replicate metadata
    of the decorated function, including nose's additional stuff
    (namely, setup and teardown).
    """
    def decorate(newfunc):
        if hasattr(func, 'compat_func_name'):
            name = func.compat_func_name
        else:
            name = func.__name__
        newfunc.__dict__ = func.__dict__
        newfunc.__doc__ = func.__doc__
        newfunc.__module__ = func.__module__
        if not hasattr(newfunc, 'compat_co_firstlineno'):
            # newfunc.compat_co_firstlineno = func.func_code.co_firstlineno
            newfunc.compat_co_firstlineno = func.__code__.co_firstlineno
        try:
            newfunc.__name__ = name
        except TypeError:
            # can't set func name in 2.3
            newfunc.compat_func_name = name
        return newfunc
    return decorate


def raises(*exceptions):
    """Test must raise one of expected exceptions to pass.
    Example use::
      @raises(TypeError, ValueError)
      def test_raises_type_error():
          raise TypeError("This test passes")
      @raises(Exception)
      def test_that_fails_by_passing():
          pass
    If you want to test many assertions about exceptions in a single test,
    you may want to use `assert_raises` instead.
    """
    valid = ' or '.join([e.__name__ for e in exceptions])
    def decorate(func):
        name = func.__name__
        def newfunc(*arg, **kw):
            try:
                func(*arg, **kw)
            except exceptions:
                pass
            except:
                raise
            else:
                message = "%s() did not raise %s" % (name, valid)
                raise AssertionError(message)
        newfunc = make_decorator(func)(newfunc)
        return newfunc
    return decorate


class TestNAs(TestCase):
    def setUp(self):
        self.path = os.path.dirname(os.path.realpath(__file__))
        self.raw_data = pd.read_csv(os.path.join(self.path,
                                                 'raw_data.csv'),
                                    usecols=['timestamp', 'count'])

    def test_handling_of_leading_trailing_nas(self):
        for i in list(range(10)) + [len(self.raw_data) - 1]:
            self.raw_data.at[i, 'count'] = np.nan

        results = detect_ts(self.raw_data, max_anoms=0.02,
                            direction='both', plot=False)
        eq_(len(results['anoms'].columns), 2)
        eq_(len(results['anoms'].iloc[:,1]), 131)

    @raises(ValueError)
    def test_handling_of_middle_nas(self):
        self.raw_data.at[len(self.raw_data) / 2, 'count'] = np.nan
        detect_ts(self.raw_data, max_anoms=0.02, direction='both')
