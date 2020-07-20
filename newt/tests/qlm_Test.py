# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 18:13:35 2020

@author: John Greendeer Lee
"""
import numpy as np
import newt.qlm as qlm
import newt.qlmACH as qlmA
import newt.qlmNum as qlmN
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
    qcyl3 = qlmN.cyl_mom(5, rho, np.pi/8, 2, 3, -.5, .5)
    qcyl3 = rot.rotate_qlm(qcyl3, np.pi/3, 0, 0)
    assert (abs(qcyl-qcyl2) < 90*np.finfo(float).eps).all()
    # Test explicit formula for L<5 (we'll do L=3)
    qcyl2 = qlmA.annulus(3, rho, 1, 2, 3, np.pi/3, np.pi/8)
    assert (np.shape(qcyl2) == (4, 7))


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
    # Test explicit formula for L<5 (we'll do L=3)
    trip2 = qlmA.tri_prism(3, 1/(1.3), 1, 2, 1.7, 3)
    assert (np.shape(trip2) == (4, 7))


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
    assert (abs(rect-rect3) < 5e3*np.finfo(float).eps).all()
    assert (abs(rect-rect4) < 5e3*np.finfo(float).eps).all()
    # Test explicit formula for L<5 (we'll do L=3)
    rect2 = qlmA.rect_prism(3, 1/15, 1, 15, 1)
    assert (np.shape(rect2) == (4, 7))


def test_ngon():
    """
    Tests regular 5-gon prism function vs combination of isosceles triangular
    prisms up to L=5.
    """
    trip5 = qlm.tri_iso_prism2(5, .2, 1, 5, 0, np.pi/5)
    trip6 = rot.rotate_qlm(trip5, 2*np.pi/5, 0, 0)
    trip7 = rot.rotate_qlm(trip6, 2*np.pi/5, 0, 0)
    trip8 = rot.rotate_qlm(trip7, 2*np.pi/5, 0, 0)
    trip9 = rot.rotate_qlm(trip8, 2*np.pi/5, 0, 0)
    pent = qlm.ngon_prism(5, 1, 1, 10*np.sin(np.pi/5), 0, 5)
    pent2 = trip5+trip6+trip7+trip8+trip9
    assert (abs(pent-pent2) < 350*np.finfo(float).eps).all()


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
    # Test explicit formula for L<5 (we'll do L=3)
    con2 = qlmA.cone(3, 3/np.pi, 1, 0, 1)
    assert (np.shape(con2) == (4, 7))


def test_tetrahedron():
    """
    Tests tetrahedron function against explicit forumulas.
    """
    teta = qlmA.tetrahedron(5, 1, 2, 1, 3)
    tetb = qlm.tetrahedron(5, 1, 2, 1, 3)
    tetc = qlmA.tetrahedron(5, 1, 1, 2, 3)
    tetd = qlm.tetrahedron(5, 1, 1, 2, 3)
    assert (abs(teta-tetb) < 100*np.finfo(float).eps).all()
    assert (abs(tetc-tetd) < 100*np.finfo(float).eps).all()
    # Test explicit formula for L<5 (we'll do L=3)
    tetd = qlm.tetrahedron(3, 1, 1, 2, 3)
    assert (np.shape(tetd) == (4, 7))


def test_pyramid():
    """
    Tests pyramid function four x,y symmetric right tetrahedrons.
    """
    tetb = qlm.tetrahedron(5, 1, np.sqrt(2), np.sqrt(2), 3)
    tetb = rot.rotate_qlm(tetb, -np.pi/4, 0, 0)
    tet2 = rot.rotate_qlm(tetb, np.pi/2, 0, 0)
    tet3 = rot.rotate_qlm(tetb, np.pi, 0, 0)
    tet4 = rot.rotate_qlm(tet2, np.pi, 0, 0)
    tet5 = qlmA.tetrahedron(5, 1, np.sqrt(2), np.sqrt(2), 3)
    tet6 = qlmA.tetrahedron(5, 1, np.sqrt(2), np.sqrt(2), 3)
    tet5 = rot.rotate_qlm(tet5, -np.pi/4, 0, 0)
    tet6 = rot.rotate_qlm(tet6, np.pi/4, 0, 0)
    tet7 = rot.rotate_qlm(tet5, np.pi, 0, 0)
    tet8 = rot.rotate_qlm(tet6, np.pi, 0, 0)
    pyr3 = tet5 + tet6 + tet7 + tet8
    pyr2 = tetb + tet2 + tet3 + tet4
    pyr = qlmA.pyramid(5, 1, 3, 2, 2)
    pyr4 = qlm.pyramid(5, 4, 1, 1, 3)
    assert (abs(pyr2-pyr)[:4] < 10*np.finfo(float).eps).all()
    assert (abs(pyr3-pyr)[:4] < 10*np.finfo(float).eps).all()
    assert (abs(pyr4-pyr2) < 10*np.finfo(float).eps).all()
    # Test explicit formula for L<5 (we'll do L=3)
    pyr = qlmA.pyramid(3, 1, 3, 2, 2)
    assert (np.shape(pyr) == (4, 7))


def test_cylhole():
    """
    Tests Steinmetz solid

    XXX : fails for q22 and q44
    """
    stein3 = qlmA.cylhole(5, 1, 2, 5)
    stein4 = qlmN.steinmetz(5, 1, 2, 5)
    stein4 = rot.rotate_qlm(stein4, np.pi/2, 0, 0)
    assert (abs(stein3-stein4)[:2] < 1e3*np.finfo(float).eps).all()
    # Test explicit formula for L<5 (we'll do L=3)
    stein2 = qlmA.cylhole(3, 1, 1, 2)
    assert (np.shape(stein2) == (4, 7))


def test_platehole():
    """
    Tests platehole against numerical calculation
    """
    ph = qlmN.platehole(5, 1, 1, .5, np.pi/3)
    ph = rot.rotate_qlm(ph, 0, np.pi/3, 0)
    pha = qlmA.platehole(5, 1, 1, .5, np.pi/3)
    assert (abs(ph-pha) < 2e3*np.finfo(float).eps).all()
    # Test explicit formula for L<5 (we'll do L=3)
    pha = qlmA.platehole(3, 1, 1, .5, np.pi/3)
    assert (np.shape(pha) == (4, 7))
