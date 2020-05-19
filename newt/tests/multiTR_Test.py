# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 20:10:14 2020

@author: John
"""
import numpy as np
import newt.glib as glb
import newt.pg2Multi as pgm
import newt.translations as trs
import newt.rotations as rot


def test_q2Q():
    """
    Check that the inner to outer translate method matches PointGravity.
    """
    d = 1
    R = 100
    m, M = 1, 1
    dr = 99
    m1 = np.array([[m, d, 0, 0], [m, -d, 0, 0]])
    m2 = np.array([[M, R, 0, 0], [M, -R, 0, 0]])
    # Create inner moments of each points at +/-r
    qm0 = pgm.qmoments(10, np.array([m1[0]]))
    qm0b = pgm.qmoments(10, np.array([m1[1]]))
    # Get outer moments of points at +/-R
    Qm2 = pgm.Qmomentsb(10, m2)
    # Find outer moments from inner qm0 and qm0b translated to +/-R
    Qlm = trs.translate_q2Q(qm0, [dr, 0, 0], 10)
    Qlmb = trs.translate_q2Q(qm0b, [-dr, 0, 0], 10)
    assert (abs(Qlm+Qlmb-Qm2) < 11*np.finfo(float).eps).all()


def test_q2q():
    """
    Check that the inner to inner translate method matches PointGravity.
    """
    d = 1
    m = 1
    m1 = np.array([[m, d, 0, 0], [m, -d, 0, 0]])
    # Find inner moments around origin
    qm1 = pgm.qmoments(10, m1)
    # Find inner moments if translated by [.1, 0, 0]
    qm1p = pgm.qmoments(10, glb.translate_point_array(m1, [.1, 0, 0]))
    # Find moments translated by [.1, 0, 0]
    qlmp = trs.translate_qlm(qm1, [0.1, 0, 0], 10)
    assert (abs(qlmp-qm1p) < 11*np.finfo(float).eps).all()


def test_q2q_durso():
    """
    Check that the inner to inner translate method matches analytic expression.
    """
    ar = 1
    rp = .1
    m = 1
    m1 = np.array([[m, ar, 0, 0], [-m, -ar, 0, 0]])
    # Find inner moments around origin
    qm1 = pgm.qmoments(4, m1)
    # Find moments translated by [.1, 0, 0]
    qlmp = trs.translate_qlm(qm1, [rp, 0, 0], 4)
    # Analytic q44 moment
    q44 = 3*np.sqrt(35/8/np.pi)*(rp*ar**3 + ar*rp**3)
    assert abs(qlmp[4, 8] - q44) < 11*np.finfo(float).eps


def test_Q2Q():
    """
    Check that the outer to outer translate method matches PointGravity.
    """
    R = 100
    M = 1
    m2 = np.array([[M, R, 0, 0], [M, -R, 0, 0]])
    # Get outer moments of points at +/-R
    Qm2 = pgm.Qmomentsb(10, m2)
    # Get outer moments of translated points
    Qm2b = pgm.Qmomentsb(10, glb.translate_point_array(m2, [0.1, 0, 0]))
    # Find outer moments from inner qm0 and qm0b translated to +/-R
    Qlmp2 = trs.translate_Qlmb(Qm2, [0.1, 0, 0], 10)
    assert (abs(Qlmp2-Qm2b) < 10*np.finfo(float).eps).all()


def test_Q2Q_durso():
    """
    Check that the outer to outer translate method matches analytic.
    """
    ar = 50
    rp = 1
    m = 1
    m1 = np.array([[m, ar, 0, 0], [-m, -ar, 0, 0]])
    # Find outer moments around origin
    Qm1 = pgm.Qmomentsb(10, m1)
    # Find outer moments translated by [1, 0, 0]
    Qlmp = trs.translate_Qlmb(Qm1, [rp, 0, 0], 10)
    # Analytic Q22 moment
    Q22 = (1/4)*np.sqrt(15/2/np.pi)*((ar+rp)**(-3) - (ar-rp)**(-3))
    assert abs(Qlmp[2, 12] - Q22) < 10*np.finfo(float).eps


def test_Q2Q2():
    """
    Check that the outer to outer translate method matches PointGravity.
    """
    R = 100
    M = 1
    m2 = np.array([[M, 0, 0, R], [M, 0, 0, -R]])
    # Get outer moments of points at +/-R
    Qm2 = pgm.Qmomentsb(10, m2)
    # Get outer moments of translated points
    Qm2b = pgm.Qmomentsb(10, glb.translate_point_array(m2, [5, 0, 0]))
    # Find outer moments from inner qm0 and qm0b translated to +/-R
    Qlmp2 = trs.translate_Qlmb(Qm2, [5, 0, 0], 10)
    assert (abs(Qlmp2-Qm2b) < 10*np.finfo(float).eps).all()


def test_rotateB():
    d = 1
    m = 1
    m1 = np.array([[m, d, 0, 0], [m, -d, 0, 0]])
    qm1 = pgm.qmoments(10, m1)
    beta = np.pi/4
    qm1b = pgm.qmoments(10, glb.rotate_point_array(m1, beta, [0, 1, 0]))
    qm1c = rot.rotate_qlm(qm1, 0, beta, 0)
    assert (abs(qm1c-qm1b) < 10*np.finfo(float).eps).all()


def test_rotateB2():
    d = 1
    m = 1
    m1 = np.array([[m, d, 0, 0], [m, -d, 0, 0]])
    qm1 = pgm.qmoments(10, m1)
    beta = -np.pi/4
    qm1b = pgm.qmoments(10, glb.rotate_point_array(m1, beta, [0, 1, 0]))
    qm1c = rot.rotate_qlm(qm1, 0, beta, 0)
    assert (abs(qm1c-qm1b) < 10*np.finfo(float).eps).all()


def test_rotateA():
    d = 1
    m = 1
    m1 = np.array([[m, d, 0, 0], [m, -d, 0, 0]])
    qm1 = pgm.qmoments(10, m1)
    alpha = np.pi/4
    qm1b = pgm.qmoments(10, glb.rotate_point_array(m1, alpha, [0, 0, 1]))
    qm1c = rot.rotate_qlm(qm1, alpha, 0, 0)
    assert (abs(qm1c-qm1b) < 15*np.finfo(float).eps).all()


def test_rotateA2():
    d = 1
    m = 1
    m1 = np.array([[m, d, 0, 0], [m, -d, 0, 0]])
    qm1 = pgm.qmoments(10, m1)
    alpha = -np.pi/4
    qm1b = pgm.qmoments(10, glb.rotate_point_array(m1, alpha, [0, 0, 1]))
    qm1c = rot.rotate_qlm(qm1, alpha, 0, 0)
    assert (abs(qm1c-qm1b) < 20*np.finfo(float).eps).all()
