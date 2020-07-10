# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 15:50:56 2020

@author: John Greendeer Lee
"""
import numpy as np
import newt.glibShapes as gshp


def test_annulus():
    """
    Check that center of mass for a full annulus centered on the axes is at
    [0, 0, 0] whether we have an even or odd number of grid points in x or z.

    Tests
    -----
    annulus : function
    """
    pEven = 16
    pOdd = 15
    cyl = gshp.annulus(1, 0, 1, 1, pEven, pEven)
    cyl2 = gshp.annulus(1, 0, 1, 1, pOdd, pOdd)
    assert (np.average(cyl[:, 1:], 0) < 10*np.finfo(float).eps).all()
    assert (np.average(cyl2[:, 1:], 0) < 10*np.finfo(float).eps).all()


def test_rectangle():
    """
    Check that center of mass for a rectangular prism centered on the axes is
    at [0, 0, 0] whether we have an even or odd number of grid points in x, y,
    or z.

    Tests
    -----
    rectangle : function
    """
    pEven = 16
    pOdd = 15
    rect = gshp.rectangle(1, 1, 1, 1, pEven, pEven, pEven)
    rect2 = gshp.rectangle(1, 1, 1, 1, pOdd, pOdd, pOdd)
    assert (np.average(rect[:, 1:], 0) < 10*np.finfo(float).eps).all()
    assert (np.average(rect2[:, 1:], 0) < 10*np.finfo(float).eps).all()


def test_cone():
    """
    Check that center of mass for a rectangular prism centered on the axes is
    at [0, 0, 0] whether we have an even or odd number of grid points in x, y,
    or z.

    Tests
    -----
    rectangle : function
    """
    pEven = 16
    pOdd = 15
    cone = gshp.cone(1, 1, 1, pEven, pEven)
    cone2 = gshp.cone(1, 1, 1, pOdd, pOdd)
    assert (np.average(cone[:, 1:3], 0) < 10*np.finfo(float).eps).all()
    assert (np.average(cone2[:, 1:3], 0) < 10*np.finfo(float).eps).all()


def test_tri_prism():
    """
    Check that center of mass for a triangular prism centered on the z-axis, on
    xy plane, whether we have an even or odd number of grid points in z.

    Tests
    -----
    tri_prism : function
    """
    pEven = 16
    pOdd = 15
    tri = gshp.tri_prism(1, 1, 1, 2, 1, pEven, pEven, pEven)
    tri2 = gshp.tri_prism(1, 1, 1, 2, 1, pOdd, pOdd, pOdd)
    assert (np.average(tri[:, 3]) < 10*np.finfo(float).eps).all()
    assert (np.average(tri2[:, 3]) < 10*np.finfo(float).eps).all()


def test_wedge():
    """
    Check that center of mass for a annular section centered about y and z,
    whether we have an even or odd number of grid points in y or z.

    Tests
    -----
    wedge : function
    """
    pEven = 16
    pOdd = 15
    wedge = gshp.wedge(1, 1, 2, 1, np.pi/6, pEven, pEven)
    wedge2 = gshp.wedge(1, 1, 2, 1, np.pi/6, pOdd, pOdd)
    assert (np.average(wedge[:, 3]) < 10*np.finfo(float).eps).all()
    assert (np.average(wedge2[:, 3]) < 10*np.finfo(float).eps).all()
    assert (np.average(wedge[:, 2]) < 10*np.finfo(float).eps).all()
    assert (np.average(wedge2[:, 2]) < 10*np.finfo(float).eps).all()


def test_trap():
    """
    Check that center of mass for a annular section centered about y and z,
    whether we have an even or odd number of grid points in y or z.

    Tests
    -----
    trapezoid : function
    """
    pEven = 16
    pOdd = 15
    trapezoid = gshp.trapezoid(1, 1, 2, 1, np.pi/6, pEven, pEven)
    trapezoid2 = gshp.trapezoid(1, 1, 2, 1, np.pi/6, pOdd, pOdd)
    assert (np.average(trapezoid[:, 3]) < 10*np.finfo(float).eps).all()
    assert (np.average(trapezoid2[:, 3]) < 10*np.finfo(float).eps).all()
    assert (np.average(trapezoid[:, 2]) < 10*np.finfo(float).eps).all()
    assert (np.average(trapezoid2[:, 2]) < 10*np.finfo(float).eps).all()


def test_outer_cone():
    """
    Check that center of mass for a annular cone section is centered about y,
    whether we have an even or odd number of grid points in y or z.

    Tests
    -----
    outer_cone : function
    """
    pEven = 16
    pOdd = 15
    outCone = gshp.outer_cone(1, 1, 2, 1, np.pi/5, pEven, pEven)
    outCone2 = gshp.outer_cone(1, 1, 2, 1, np.pi/5, pOdd, pOdd)
    assert (np.average(outCone[:, 2]) < 10*np.finfo(float).eps).all()
    assert (np.average(outCone2[:, 2]) < 10*np.finfo(float).eps).all()


def test_shell():
    """
    Check that the radius and mass and quadrupole check out.

    Tests
    -----
    spherical_random_shell : function
    """
    mass = 1.
    r = 10
    n = 100000
    shell = gshp.spherical_random_shell(mass, r, n)
    nNew = len(shell)
    threshold = 100.*np.sqrt(nNew)*np.finfo(float).eps
    # Check the radius
    assert abs(np.sqrt(np.sum(shell[:, 1:]**2, 1))-r).all() < threshold
    # Check the mass
    assert abs(np.sum(shell[:, 0])-mass) < threshold
    # Make sure each top quarter is roughly evenly distributed
    zq = np.array([shell[k] for k in range(nNew) if shell[k, 3] > r/2.])
    yq = np.array([shell[k] for k in range(nNew) if shell[k, 2] > r/2.])
    xq = np.array([shell[k] for k in range(nNew) if shell[k, 1] > r/2.])
    q = np.array([np.sum(xq[:, 0]), np.sum(yq[:, 0]), np.sum(zq[:, 0])])
    assert (np.max(q)-np.min(q))/np.average(q) < 2.*np.sqrt(n)
