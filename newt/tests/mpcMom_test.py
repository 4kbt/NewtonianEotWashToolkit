# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 19:05:14 2020

@author: John
"""
import newt.read_multiFor as rmf
import numpy as np

MPCPATH = './newt/tests/mpc/'
MOMPATH = './newt/tests/mom/'


def test_sph():
    testsph = rmf.read_mpc(10, 'testsph.mpc', MPCPATH)
    testsph2 = rmf.read_gsq('testsph.gsq', MOMPATH)
    assert (np.abs(testsph - testsph2) < 1e7*np.finfo(float).eps).all()


def test_tetrahedron():
    testtet = rmf.read_mpc(10, 'testtet.mpc', MPCPATH)
    testtet2 = rmf.read_gsq('testtet.gsq', MOMPATH)
    assert (np.abs(testtet - testtet2) < 1e7*np.finfo(float).eps).all()


def test_triangle():
    testtri = rmf.read_mpc(10, 'testtri.mpc', MPCPATH)
    testtri2 = rmf.read_gsq('testtri.gsq', MOMPATH)
    assert (np.abs(testtri - testtri2) < 1e7*np.finfo(float).eps).all()


def test_platehole():
    testph = rmf.read_mpc(10, 'testph.mpc', MPCPATH)
    testph2 = rmf.read_gsq('testph.gsq', MOMPATH)
    assert (np.abs(testph - testph2) < 2e7*np.finfo(float).eps).all()


def test_cone():
    testco = rmf.read_mpc(10, 'testcone.mpc', MPCPATH)
    testco2 = rmf.read_gsq('testcone.gsq', MOMPATH)
    assert (np.abs(testco - testco2) < 1e7*np.finfo(float).eps).all()


def test_rectangle():
    testre = rmf.read_mpc(10, 'testrect.mpc', MPCPATH)
    testre2 = rmf.read_gsq('testrect.gsq', MOMPATH)
    assert (np.abs(testre - testre2) < 2e8*np.finfo(float).eps).all()


def test_cylhole():
    testch = rmf.read_mpc(10, 'testch.mpc', MPCPATH)
    testch2 = rmf.read_gsq('testch.gsq', MOMPATH)
    assert (np.abs(testch - testch2) < 5e8*np.finfo(float).eps).all()


def test_cyl():
    testcyl = rmf.read_mpc(10, 'testcyl0.mpc', MPCPATH)
    testcyl1 = rmf.read_gsq('testcyl0.gsq', MOMPATH)
    assert (np.abs(testcyl - testcyl1) < 1e6*np.finfo(float).eps).all()


def test_pyr():
    testpyr = rmf.read_mpc(10, 'testpyr.mpc', MPCPATH)
    testpyr2 = rmf.read_gsq('testpyr', MOMPATH)
    assert (np.abs(testpyr - testpyr2) < 1e7*np.finfo(float).eps).all()


def test_partcyl():
    """
    Internal consistency of MULTIN
    XXX: Looks like q(7,1) and q(8,6) have small errors in part-cylinder
    """
    testpc2 = rmf.read_gsq('testpc2.gsq', MOMPATH)
    testpc3 = rmf.read_gsq('testpc3.gsq', MOMPATH)
    assert (np.abs(testpc3 - testpc2) < 2e8*np.finfo(float).eps).all()


def test_partcyl2():
    """
    Part-cylinder vs NEWT calc.
    """
    testpc = rmf.read_mpc(10, 'testpc3.mpc', MPCPATH)
    testpc2 = rmf.read_gsq('testpc3.gsq', MOMPATH)
    assert (np.abs(testpc - testpc2) < 2e8*np.finfo(float).eps).all()


def test_trap():
    testtrap = rmf.read_gsq('testtrap.gsq', MOMPATH)
    testtrap2 = rmf.read_gsq('testtrap.gsq', MOMPATH)
    assert (np.abs(testtrap - testtrap2) < 2e8*np.finfo(float).eps).all()


def test_load():
    testtrap = rmf.read_gsq('testtrap.gsq', MOMPATH)
    testtrap2 = rmf.read_mpc(10, 'testload.mpc', MPCPATH)
    assert (np.abs(testtrap - testtrap2) < 2e8*np.finfo(float).eps).all()
