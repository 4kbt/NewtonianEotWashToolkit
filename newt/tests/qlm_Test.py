# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 18:13:35 2020

@author: John Greendeer Lee
"""
import numpy as np
import newt.qlm as qlm
import newt.qlmACH as qlmA
import newt.translations as trs
import newt.rotations as rot


def test_cylinder():
    """
    Tests recursive cylinder function vs explicit formulas up to L=5.
    """
    qcyl = qlm.cylinder(5, 1, 1, 1)
    qcyl2 = qlmA.annulus(5, 1/np.pi, 1, 0, 1, 0, np.pi)
    assert (abs(qcyl-qcyl2) < 10*np.finfo(float).eps).all()


def test_cylinder2():
    """
    Tests recursive cylinder function vs annulus formula up to L=10. That is
    the annulus formula should be consistent with the recursive cylinder
    calculation.
    """
    qcyl = qlm.cylinder(10, 1, 1, 1)
    qcyl2 = qlm.annulus(10, 1, 1, 0, 1, 0, np.pi)
    assert (abs(qcyl-qcyl2) < 10*np.finfo(float).eps).all()


def test_annulus():
    """
    Tests annulus function vs explicit annulus formula up to L=5. Full annulus.
    """
    qcyl = qlm.annulus(5, 1, 1, 2, 3, 0, np.pi)
    qcyl2 = qlmA.annulus(5, 1/(5*np.pi), 1, 2, 3, 0, np.pi)
    assert (abs(qcyl-qcyl2) < 30*np.finfo(float).eps).all()


def test_annulus2():
    """
    Tests annulus function vs explicit annulus formula up to L=5. Partial.
    """
    qcyl = qlm.annulus(5, 1, 1, 2, 3, np.pi/3, np.pi/8)
    rho = 8/(np.pi*(3**2-2**2))
    qcyl2 = qlmA.annulus(5, rho, 1, 2, 3, np.pi/3, np.pi/8)
    assert (abs(qcyl-qcyl2) < 90*np.finfo(float).eps).all()


def test_tri_prism():
    """
    Tests isosceles triangular prism function vs explicit formula up to L=5.
    """
    trip = qlm.tri_iso_prism(5, 1, 1, 1, 1, 0)
    trip2 = qlmA.tri_prism(5, 2, 1, 1, .5, -.5)
    r = np.sqrt(1**2 + .5**2)
    trip3 = qlm.tri_iso_prism2(5, 1, 1, r, 0, np.arctan(.5))
    assert (abs(trip-trip2) < 90*np.finfo(float).eps).all()
    assert (abs(trip3-trip2) < 90*np.finfo(float).eps).all()


def test_tri_prism2():
    """
    Tests arbitrary triangular prism function vs explicit formula up to L=5.
    """
    trip = qlm.tri_prism(5, 1, 1, 2, 1.7, 3)
    trip2 = qlmA.tri_prism(5, 1/(1.3), 1, 2, 1.7, 3)
    assert (abs(trip-trip2) < 200*np.finfo(float).eps).all()


def test_rect_prism():
    """
    Tests rectangular prism function vs explicit formula up to L=5.
    """
    rect = qlm.rect_prism(5, 1, 1, 1, 15, 0)
    rect2 = qlmA.rect_prism(5, 1/15, 1, 15, 1)
    # Rectangle from triang ACH matches ACH values
    trip2 = qlmA.tri_prism(5, 1/15, 1, 7.5, .5, -.5)
    trip = qlmA.tri_prism(5, 1/15, 1, .5, 7.5, -7.5)
    trip2 = rot.rotate_qlm(trip2, np.pi/2, 0, 0)
    trip3 = rot.rotate_qlm(trip2, np.pi, 0, 0)
    trip4 = rot.rotate_qlm(trip, np.pi, 0, 0)
    rect3 = trip+trip2+trip3+trip4
    # Rectangle from triang prisms matches ACH values
    trip5 = qlm.tri_iso_prism(5, .25, 1, 15, .5, 0)
    trip6 = qlm.tri_iso_prism(5, .25, 1, 1, 7.5, 0)
    trip6 = rot.rotate_qlm(trip6, np.pi/2, 0, 0)
    trip7 = rot.rotate_qlm(trip5, np.pi, 0, 0)
    trip8 = rot.rotate_qlm(trip6, np.pi, 0, 0)
    rect4 = trip5+trip6+trip7+trip8
    assert (abs(rect-rect2) < 400*np.finfo(float).eps).all()


def test_cone():
    """
    Tests cone function vs explicit formula up to L=5.
    """
    con = qlm.cone(5, 1, 1, 1, 0, np.pi)
    con3 = rot.rotate_qlm(con, 0, np.pi, 0)
    con3 = trs.translate_qlm(con3, [0, 0, .5])
    con2 = qlmA.cone(5, 3/np.pi, 1, 1, 0)
    assert (abs(con3-con2) < 10*np.finfo(float).eps).all()
    con3 = trs.translate_qlm(con, [0, 0, -.5])
    con2 = qlmA.cone(5, 3/np.pi, 1, 0, 1)
    assert (abs(con3-con2) < 10*np.finfo(float).eps).all()
