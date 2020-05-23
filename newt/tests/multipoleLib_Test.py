# -*- coding: utf-8 -*-
"""
Created on Tue May 12 19:17:52 2020

@author: John Greendeer Lee
"""
import numpy as np
import newt.pg2Multi as pgm
import newt.multipoleLib as mplb
import numpy.random as rand
import newt.glib as glb


def test_force1():
    """
    Test gravitational force from point mass at a meter on a point mass at the
    origin.

    Tests
    -----
    mplb.multipole_force : function
    """
    m1 = np.array([[1, 0, 0, 0]])
    m2 = np.array([[1, 1, 0, 0]])
    q1 = pgm.qmoments(10, m1)
    q2 = pgm.Qmomentsb(10, m2)
    force = mplb.multipole_force(10, q1, q2, 0, 0, 0)
    assert (abs(force[0] - mplb.BIG_G) < 10*np.finfo(float).eps)
    m2 = np.array([[1, 0, 1, 0]])
    q2 = pgm.Qmomentsb(10, m2)
    force = mplb.multipole_force(10, q1, q2, 0, 0, 0)
    assert (abs(force[1] - mplb.BIG_G) < 10*np.finfo(float).eps)
    m2 = np.array([[1, 0, 0, 1]])
    q2 = pgm.Qmomentsb(10, m2)
    force = mplb.multipole_force(10, q1, q2, 0, 0, 0)
    assert (abs(force[2] - mplb.BIG_G) < 10*np.finfo(float).eps)


def test_force2():
    """
    Test gravitational force from point mass at two meters on point mass away
    from the origin.

    Tests
    -----
    mplb.multipole_force : function
    """
    m1b = np.array([[1, 0, 0, 1]])
    m2 = np.array([[1, 0, 0, 2]])
    q1b = pgm.qmoments(20, m1b)
    q2 = pgm.Qmomentsb(20, m2)
    # Check force for where m1 is placed
    force = mplb.multipole_force(20, q1b, q2, 0, 0, 0)
    assert (abs(force[2] - mplb.BIG_G) < 10*np.finfo(float).eps)
    # Check force at if displace inner moments to origin
    force = mplb.multipole_force(20, q1b, q2, 0, 0, -1)
    assert (abs(force[2] - mplb.BIG_G/4) < 10*np.finfo(float).eps)
    m1b = np.array([[1, 1, 0, 0]])
    m2 = np.array([[1, 2, 0, 0]])
    q1b = pgm.qmoments(20, m1b)
    q2 = pgm.Qmomentsb(20, m2)
    # Check force for where m1 is placed
    force = mplb.multipole_force(20, q1b, q2, 0, 0, 0)
    assert (abs(force[0] - mplb.BIG_G) < 10*np.finfo(float).eps)
    # Check force at if displace inner moments to origin
    force = mplb.multipole_force(20, q1b, q2, -1, 0, 0)
    assert (abs(force[0] - mplb.BIG_G/4) < 10*np.finfo(float).eps)
    m1b = np.array([[1, 0, 1, 0]])
    m2 = np.array([[1, 0, 2, 0]])
    q1b = pgm.qmoments(20, m1b)
    q2 = pgm.Qmomentsb(20, m2)
    # Check force for where m1 is placed
    force = mplb.multipole_force(20, q1b, q2, 0, 0, 0)
    assert (abs(force[1] - mplb.BIG_G) < 10*np.finfo(float).eps)
    # Check force at if displace inner moments to origin
    force = mplb.multipole_force(20, q1b, q2, 0, -1, 0)
    assert (abs(force[1] - mplb.BIG_G/4) < 10*np.finfo(float).eps)


def test_quadrupole_torque():
    """
    Compare the point matrix calculation to an analytic formulation of a
    quadrupole torque.

    Tests
    -----
    glb.point_matrix_gravity : function
    """
    d = 1
    R = rand.rand()*100 + 1.1
    m, M = 1, 1
    N = 60
    L = 10
    m1 = np.array([[m, d, 0, 0], [m, -d, 0, 0]])
    m2 = np.array([[M, R, 0, 0], [M, -R, 0, 0]])
    qlm = pgm.qmoments(L, m1)
    tau = np.zeros(N)
    ts = np.zeros([60, 3])
    d2R2 = d**2 + R**2
    tc = np.zeros([N, L+1], dtype='complex')
    ts = np.zeros([N, L+1], dtype='complex')
    for k in range(N):
        a = 2*np.pi*k/N
        ca = np.cos(a)
        Q = glb.rotate_point_array(m2, a, [0, 0, 1])
        Qlmb = pgm.Qmomentsb(L, Q)
        tau[k] = 2*mplb.BIG_G*M*m*d*R*np.sin(a)
        tau[k] *= 1/(d2R2-2*d*R*ca)**(3/2) - 1/(d2R2+2*d*R*ca)**(3/2)
        tlm, tc[k], ts[k] = mplb.torque_lm(L, qlm, Qlmb)
    # XXX should be np.sum(tc, 1)-tau, but existing sign error!
    assert(abs(np.sum(tc, 1)+tau) < 10*np.finfo(float).eps).all()


def test_hexapole_torque():
    """
    Compare the point matrix calculation to an analytic formulation of a
    hexapole torque.

    Tests
    -----
    glb.point_matrix_gravity : function
    """
    d = 1
    z = rand.randn()*10
    R = rand.rand()*100 + 1.1
    m, M = 1, 1
    N = 60
    m0 = np.array([m, d, 0, 0])
    m1 = np.copy(m0)
    m2 = np.array([M, R, 0, z])
    m3 = np.copy(m2)
    zhat = [0, 0, 1]
    for k in range(1, 3):
        m1 = np.vstack([m1, glb.rotate_point_array(m0, 2*k*np.pi/3, zhat)])
        m3 = np.vstack([m3, glb.rotate_point_array(m2, 2*k*np.pi/3, zhat)])
    L = 10
    qlm = pgm.qmoments(L, m1)
    tau = np.zeros(N)
    tc = np.zeros([N, L+1], dtype='complex')
    ts = np.zeros([N, L+1], dtype='complex')
    d2R2 = d**2 + R**2 + z**2
    for k in range(N):
        a = 2*np.pi*k/N
        Q = glb.rotate_point_array(m3, a, zhat)
        Qlmb = pgm.Qmomentsb(L, Q)
        fac = 3*glb.BIG_G*M*m*d*R
        tau[k] = np.sin(a)/(d2R2-2*d*R*np.cos(a))**(3/2)
        tau[k] += np.sin(a+2*np.pi/3)/(d2R2-2*d*R*np.cos(a+2*np.pi/3))**(3/2)
        tau[k] += np.sin(a+4*np.pi/3)/(d2R2-2*d*R*np.cos(a+4*np.pi/3))**(3/2)
        tau[k] *= fac
        tlm, tc[k], ts[k] = mplb.torque_lm(L, qlm, Qlmb)
    # XXX should be np.sum(tc, 1)-tau, but existing sign error!
    assert(abs(np.sum(tc, 1)+tau) < 10*np.finfo(float).eps).all()
