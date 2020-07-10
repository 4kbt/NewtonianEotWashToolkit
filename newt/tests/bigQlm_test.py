# -*- coding: utf-8 -*-
"""
Created on Mon May 25 08:03:49 2020

@author: John
"""
import numpy as np
import newt.glib as glb
import newt.glibShapes as gshp
import newt.pg2Multi as pgm
import newt.bigQlm as bqlm
import newt.bigQlmNum as bqlmn


def test_bAnnulus():
    """
    Test of explicit annulus outer moments against point gravity.

    Tests
    -----
    bqlm.annulus : function
    """
    mout = gshp.wedge(1, 1, 2, 1, np.pi/2, 60, 60)
    mmout2 = pgm.Qmomentsb(5, glb.translate_point_array(mout, [0, 0, .5]))
    Qlmb2 = bqlm.annulus(5, 2/(np.pi*(2**2-1)), 1, 1, 2, 0, np.pi/2)
    assert (abs(Qlmb2-mmout2) < .002).all()
    mmout3 = pgm.Qmomentsb(5, glb.translate_point_array(mout, [0, 0, -.5]))
    Qlmb3 = bqlm.annulus(5, 2/(np.pi*(2**2-1)), -1, 1, 2, 0, np.pi/2)
    assert (abs(Qlmb3-mmout3) < .002).all()
    mout3 = gshp.wedge(1, 1, 2, 1, np.pi/3, 60, 60)
    mmout3 = pgm.Qmomentsb(5, glb.translate_point_array(mout3, [0, 0, .5]))
    Qlmb3 = bqlm.annulus(5, 3/(np.pi*(2**2-1)), 1, 1, 2, 0, np.pi/3)
    assert (abs(Qlmb3-mmout3) < .002).all()
    mout4 = gshp.wedge(1, 1, 2, 1, np.pi/4, 60, 60)
    mmout4 = pgm.Qmomentsb(5, glb.translate_point_array(mout4, [0, 0, .5]))
    Qlmb4 = bqlm.annulus(5, 4/(np.pi*(2**2-1)), 1, 1, 2, 0, np.pi/4)
    assert (abs(Qlmb4-mmout4) < .002).all()
    mout5 = gshp.wedge(1, 1, 2, 1, np.pi/5, 60, 60)
    mmout5 = pgm.Qmomentsb(5, glb.translate_point_array(mout5, [0, 0, .5]))
    Qlmb5 = bqlm.annulus(5, 5/(np.pi*(2**2-1)), 1, 1, 2, 0, np.pi/5)
    assert (abs(Qlmb5-mmout5) < .002).all()
    # Test that a L<5 works (we'll do 3)
    Qlmb3 = bqlm.annulus(3, 3/(np.pi*(2**2-1)), 1, 1, 2, 0, np.pi/3)
    mmout3 = pgm.Qmomentsb(3, glb.translate_point_array(mout3, [0, 0, .5]))
    assert (np.shape(Qlmb3) == (4, 7))
    assert (abs(Qlmb3-mmout3) < .002).all()


def test_bAnnulusn():
    """
    Test of numerical annulus outer moments against point gravity.

    Tests
    -----
    bqlmn.annulus : function
    """
    mout = gshp.wedge(1, 1, 2, 1, np.pi/2, 60, 60)
    mmout2 = pgm.Qmomentsb(5, glb.translate_point_array(mout, [0, 0, .5]))
    Qlmb2 = bqlmn.annulus(5, 2/(np.pi*(2**2-1)), 0, 1, 1, 2, np.pi/2)
    assert (abs(Qlmb2-mmout2) < .002).all()
    mmout3 = pgm.Qmomentsb(5, glb.translate_point_array(mout, [0, 0, -.5]))
    Qlmb3 = bqlmn.annulus(5, 2/(np.pi*(2**2-1)), -1, 0, 1, 2, np.pi/2)
    assert (abs(Qlmb3-mmout3) < .002).all()
    mout3 = gshp.wedge(1, 1, 2, 1, np.pi/3, 60, 60)
    mmout3 = pgm.Qmomentsb(5, glb.translate_point_array(mout3, [0, 0, .5]))
    Qlmb3 = bqlmn.annulus(5, 3/(np.pi*(2**2-1)), 0, 1, 1, 2, np.pi/3)
    assert (abs(Qlmb3-mmout3) < .002).all()
    mout4 = gshp.wedge(1, 1, 2, 1, np.pi/4, 60, 60)
    mmout4 = pgm.Qmomentsb(5, glb.translate_point_array(mout4, [0, 0, .5]))
    Qlmb4 = bqlmn.annulus(5, 4/(np.pi*(2**2-1)), 0, 1, 1, 2, np.pi/4)
    assert (abs(Qlmb4-mmout4) < .002).all()
    mout5 = gshp.wedge(1, 1, 2, 1, np.pi/5, 60, 60)
    mmout5 = pgm.Qmomentsb(5, glb.translate_point_array(mout5, [0, 0, .5]))
    Qlmb5 = bqlmn.annulus(5, 5/(np.pi*(2**2-1)), 0, 1, 1, 2, np.pi/5)
    assert (abs(Qlmb5-mmout5) < .002).all()


def test_bTrapezoidn():
    """
    Test of numerical trapezoid outer moments against point gravity.

    Tests
    -----
    bqlmn.trapezoid : function
    """
    iR, oR = 1, 2
    beta = np.pi/3
    t = 1
    vol = (oR-iR)*np.cos(beta)*(oR+iR)*np.sin(beta)*t
    mout = gshp.trapezoid(1, iR, oR, t, beta, 60, 60)
    mmout2 = pgm.Qmomentsb(5, glb.translate_point_array(mout, [0, 0, .5]))
    Qlmb2 = bqlmn.trapezoid(5, 1/vol, 0, t, iR, oR, beta)
    assert (abs(Qlmb2-mmout2) < .01).all()
    mmout3 = pgm.Qmomentsb(5, glb.translate_point_array(mout, [0, 0, -.5]))
    Qlmb3 = bqlmn.trapezoid(5, 1/vol, -t, 0, iR, oR, beta)
    assert (abs(Qlmb3-mmout3) < .01).all()
    beta = np.pi/4
    vol = (oR-iR)*np.cos(beta)*(oR+iR)*np.sin(beta)*t
    mout4 = gshp.trapezoid(1, iR, oR, t, beta, 60, 60)
    mmout4 = pgm.Qmomentsb(5, glb.translate_point_array(mout4, [0, 0, .5]))
    Qlmb4 = bqlmn.trapezoid(5, 1/vol, 0, t, iR, oR, beta)
    assert (abs(Qlmb4-mmout4) < .01).all()
    beta = np.pi/5
    vol = (oR-iR)*np.cos(beta)*(oR+iR)*np.sin(beta)*t
    mout5 = gshp.trapezoid(1, iR, oR, t, beta, 60, 60)
    mmout5 = pgm.Qmomentsb(5, glb.translate_point_array(mout5, [0, 0, .5]))
    Qlmb5 = bqlmn.trapezoid(5, 1/vol, 0, t, iR, oR, beta)
    assert (abs(Qlmb5-mmout5) < .01).all()


def test_bOuter_conen():
    """
    Test of numerical outer cone outer moments against point gravity.

    Tests
    -----
    bqlmn.outer_cone : function
    """
    iR, oR = 1, 2
    beta = np.pi/3
    H = 1
    Hp = H*oR/(oR-iR)
    vol = beta*(Hp*oR**2/3-H*iR**2-(Hp-H)*iR**2/3)
    mout = gshp.outer_cone(1, iR, oR, H, beta, 60, 60)
    mmout2 = pgm.Qmomentsb(5, mout)
    Qlmb2 = bqlmn.outer_cone(5, 1/vol, H, iR, oR, beta)
    assert (abs(Qlmb2-mmout2) < .005).all()
    beta = np.pi/4
    vol = beta*(Hp*oR**2/3-H*iR**2-(Hp-H)*iR**2/3)
    mout4 = gshp.outer_cone(1, iR, oR, H, beta, 60, 60)
    mmout4 = pgm.Qmomentsb(5, mout4)
    Qlmb4 = bqlmn.outer_cone(5, 1/vol, H, iR, oR, beta)
    assert (abs(Qlmb4-mmout4) < .01).all()
    beta = np.pi/5
    vol = beta*(Hp*oR**2/3-H*iR**2-(Hp-H)*iR**2/3)
    mout5 = gshp.outer_cone(1, iR, oR, H, beta, 60, 60)
    mmout5 = pgm.Qmomentsb(5, mout5)
    Qlmb5 = bqlmn.outer_cone(5, 1/vol, H, iR, oR, beta)
    assert (abs(Qlmb5-mmout5) < .01).all()
