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
