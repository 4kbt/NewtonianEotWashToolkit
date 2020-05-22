# -*- coding: utf-8 -*-
"""
Created on Tue May 12 19:17:52 2020

@author: John Greendeer Lee
"""
import numpy as np
import newt.pg2Multi as pgm
import newt.multiForce as mfor


def test_force1():
    """
    Test gravitational force from point mass at a meter on a point mass at the
    origin.

    Tests
    -----
    mfor.multipole_force : function
    """
    m1 = np.array([[1, 0, 0, 0]])
    m2 = np.array([[1, 1, 0, 0]])
    q1 = pgm.qmoments(10, m1)
    q2 = pgm.Qmomentsb(10, m2)
    force = mfor.multipole_force(10, q1, q2, 0, 0, 0)
    assert (abs(force[0] - mfor.big_G) < 10*np.finfo(float).eps)
    m2 = np.array([[1, 0, 1, 0]])
    q2 = pgm.Qmomentsb(10, m2)
    force = mfor.multipole_force(10, q1, q2, 0, 0, 0)
    assert (abs(force[1] - mfor.big_G) < 10*np.finfo(float).eps)
    m2 = np.array([[1, 0, 0, 1]])
    q2 = pgm.Qmomentsb(10, m2)
    force = mfor.multipole_force(10, q1, q2, 0, 0, 0)
    assert (abs(force[2] - mfor.big_G) < 10*np.finfo(float).eps)


def test_force2():
    """
    Test gravitational force from point mass at two meters on point mass away
    from the origin.

    Tests
    -----
    mfor.multipole_force : function
    """
    m1b = np.array([[1, 0, 0, 1]])
    m2 = np.array([[1, 0, 0, 2]])
    q1b = pgm.qmoments(20, m1b)
    q2 = pgm.Qmomentsb(20, m2)
    # Check force for where m1 is placed
    force = mfor.multipole_force(20, q1b, q2, 0, 0, 0)
    assert (abs(force[2] - mfor.big_G) < 10*np.finfo(float).eps)
    # Check force at if displace inner moments to origin
    force = mfor.multipole_force(20, q1b, q2, 0, 0, -1)
    assert (abs(force[2] - mfor.big_G/4) < 10*np.finfo(float).eps)
    m1b = np.array([[1, 1, 0, 0]])
    m2 = np.array([[1, 2, 0, 0]])
    q1b = pgm.qmoments(20, m1b)
    q2 = pgm.Qmomentsb(20, m2)
    # Check force for where m1 is placed
    force = mfor.multipole_force(20, q1b, q2, 0, 0, 0)
    assert (abs(force[0] - mfor.big_G) < 10*np.finfo(float).eps)
    # Check force at if displace inner moments to origin
    force = mfor.multipole_force(20, q1b, q2, -1, 0, 0)
    assert (abs(force[0] - mfor.big_G/4) < 10*np.finfo(float).eps)
    m1b = np.array([[1, 0, 1, 0]])
    m2 = np.array([[1, 0, 2, 0]])
    q1b = pgm.qmoments(20, m1b)
    q2 = pgm.Qmomentsb(20, m2)
    # Check force for where m1 is placed
    force = mfor.multipole_force(20, q1b, q2, 0, 0, 0)
    assert (abs(force[1] - mfor.big_G) < 10*np.finfo(float).eps)
    # Check force at if displace inner moments to origin
    force = mfor.multipole_force(20, q1b, q2, 0, -1, 0)
    assert (abs(force[1] - mfor.big_G/4) < 10*np.finfo(float).eps)
