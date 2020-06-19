# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 19:59:20 2020

@author: John Greendeer Lee
"""
import numpy as np
import newt.glib as glb
import newt.gShapeGL as gshp


def test_sphere():
    """
    Check that a sphere with Gauss-Legendre distributed and weighted points
    follows a 1/r^2 law and that the accuracy is pretty good.
    """
    m1 = np.array([[1, 5, 0, 0]])
    N = 4
    ftru = glb.BIG_G/(np.arange(N)+2)**2
    sph2 = gshp.sphere(1, 1, 10)
    f1s = np.zeros(N)
    f2s = np.zeros(N)
    for k in range(N):
        sph = gshp.sphere(1, 1, 2**(k+2))
        m2 = np.array([[1, 2+k, 0, 0]])
        f, t = glb.point_matrix_gravity(sph, m1)
        f2, t2 = glb.point_matrix_gravity(sph2, m2)
        f1s[k] = f[0]
        f2s[k] = f2[0]
    assert (np.abs(f1s-ftru[3])/ftru[3] < .001).all()
    assert (np.abs(f2s-ftru)/ftru < .003).all()


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
