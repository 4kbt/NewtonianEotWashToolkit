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
        tlm, tc[k], ts[k] = mplb.torque_lm(qlm, Qlmb)
    # Check that torque from multipoles matches analytic
    assert(abs(np.sum(tc, 1)-tau) < 10*np.finfo(float).eps).all()


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
        tlm, tc[k], ts[k] = mplb.torque_lm(qlm, Qlmb)
    # Check that torque from multipoles matches analytic
    assert(abs(np.sum(tc, 1)-tau) < 10*np.finfo(float).eps).all()


def test_torque_a():
    """
    Z-torque from rotating inner moments about x-axis, no translation

    Tests
    -----
    torque
    """
    # Rotating inner moments about x-axis, z-torque on inner
    m0 = np.array([[1, 1, 0, 0], [1, -.5, np.sqrt(3)/2, 0], [1, -.5, -np.sqrt(3)/2, 0]])
    m1 = np.array([[1, 0, 2, 0], [1, -np.sqrt(3), -1, 0], [1, -np.sqrt(3), 1, 0]])
    m0 = glb.rotate_point_array(m0, np.pi/8, [0, 0, 1])
    m0b = glb.rotate_point_array(m0, np.pi/2, [0, 1, 0])

    angles = np.arange(0, 360, 10)*np.pi/180
    torques_z = np.zeros(len(angles))

    # Compute as moments immediately
    L = 20
    d0 = pgm.qmoments(L, m0)
    d1 = pgm.Qmomentsb(L, m1)
    tx, tcx, tsx = mplb.torque(d0, d1, [0, np.pi/2, 0], [0, 0, 0])
    signal = mplb.torques_at_angle(tcx, tsx, angles)

    for k, angle in enumerate(angles):
        m0b2 = glb.rotate_point_array(m0b, angle, [1, 0, 0])
        _, torq = glb.point_matrix_gravity(m0b2, m1)
        torques_z[k] = torq[2]
    assert(abs(torques_z-signal) < 10*np.finfo(float).eps).all()


def test_torque_b():
    """
    Z-torque from rotating outer moments about x-axis, no translation

    Tests
    -----
    torque
    """
    # Rotating outer moments about x-axis, z-torque on inner
    m0 = np.array([[1, 1, .5, 0], [1, -1, -.5, 0]])
    m1 = np.array([[1, 0, 2, 0], [1, 0, -2, 0]])
    m1b = glb.rotate_point_array(m1, np.pi/2, [0, 1, 0])

    angles = np.arange(0, 360, 10)*np.pi/180
    torques_z = np.zeros(len(angles))

    # Compute as moments immediately
    L = 20
    d0 = pgm.qmoments(L, m0)
    d1 = pgm.Qmomentsb(L, m1)
    tx, tcx, tsx = mplb.torque(d0, d1, [0, np.pi/2, 0], [0, 0, 0],
                               'outer-outer')
    signal = mplb.torques_at_angle(tcx, tsx, angles, outer=True)

    for k, angle in enumerate(angles):
        m1b2 = glb.rotate_point_array(m1b, angle, [1, 0, 0])
        #m1b2 = glb.translate_point_array(m1b2, [0, 0, 4])
        _, torq = glb.point_matrix_gravity(m0, m1b2)
        torques_z[k] = torq[2]
    assert(abs(torques_z-signal) < 10*np.finfo(float).eps).all()


def test_torque_c():
    """
    Z-torque from rotating outer moments about x-axis and translated from inner
    moments to outer moments (large translation compared to outer moment radius)

    Tests
    -----
    torque
    """
    m0 = np.array([[1, 1, .5, 0], [1, -1, -.5, 0]])
    m1 = np.array([[1, 0, 1, 0], [1, 0, -1, 0]])
    m1b = glb.rotate_point_array(m1, np.pi/2, [0, 1, 0])

    angles = np.arange(0, 360, 10)*np.pi/180
    torques_z = np.zeros(len(angles))

    # Compute as moments immediately
    L = 20
    d0 = pgm.qmoments(L, m0)
    d1 = pgm.qmoments(L, m1)
    dz = 5
    tx, tcx, tsx = mplb.torque(d0, d1, [0, np.pi/2, 0], [0, 0, dz],
                               'inner-outer')
    signal = mplb.torques_at_angle(tcx, tsx, angles)

    for k, angle in enumerate(angles):
        m1b2 = glb.rotate_point_array(m1b, angle, [1, 0, 0])
        m1b2 = glb.translate_point_array(m1b2, [0, 0, dz])
        _, torq = glb.point_matrix_gravity(m0, m1b2)
        torques_z[k] = torq[2]
    assert(abs(torques_z-signal) < 10*np.finfo(float).eps).all()


def test_torque_d():
    """
    Z-torque from rotating outer moments about x-axis and translated from outer
    moments to outer moments (small translation compared to outer moment radius)

    Tests
    -----
    torque
    """
    m0 = np.array([[1, 1, .5, 0], [1, -1, -.5, 0]])
    m1 = np.array([[1, 0, 2, 0], [1, 0, -2, 0]])
    m1b = glb.rotate_point_array(m1, np.pi/2, [0, 1, 0])

    angles = np.arange(0, 360, 10)*np.pi/180
    torques_z = np.zeros(len(angles))

    # Compute as moments immediately
    L = 20
    d0 = pgm.qmoments(L, m0)
    d1 = pgm.Qmomentsb(L, m1)
    dz = .9
    tx, tcx, tsx = mplb.torque(d0, d1, [0, np.pi/2, 0], [0, 0, dz], 'outer-outer')
    signal = mplb.torques_at_angle(tcx, tsx, angles, outer=True)

    for k, angle in enumerate(angles):
        m1b2 = glb.rotate_point_array(m1b, angle, [1, 0, 0])
        m1b2 = glb.translate_point_array(m1b2, [0, 0, dz])
        _, torq = glb.point_matrix_gravity(m0, m1b2)
        torques_z[k] = torq[2]
    assert(abs(torques_z-signal) < 1e3*np.finfo(float).eps).all()


def test_torque_e():
    """
    Z-torque from rotating inner moments about x-axis and translated from inner
    moments to inner moments (moments still separated in radius)

    Tests
    -----
    torque
    """
    m0 = np.array([[1, 1, 0, 0], [1, -1, 0, 0]])
    m1 = np.array([[1, .5, 2, 0], [1, -.5, -2, 0]])
    m0b = glb.rotate_point_array(m0, np.pi/2, [0, 1, 0])

    angles = np.arange(0, 360, 10)*np.pi/180
    torques_z = np.zeros(len(angles))

    # Compute as moments immediately
    L = 20
    d0 = pgm.qmoments(L, m0)
    d1 = pgm.Qmomentsb(L, m1)
    dz = .5
    tx, tcx, tsx = mplb.torque(d0, d1, [0, np.pi/2, 0], [0, 0, dz])
    signal = mplb.torques_at_angle(tcx, tsx, angles)

    for k, angle in enumerate(angles):
        m0b2 = glb.rotate_point_array(m0b, angle, [1, 0, 0])
        m0b2 = glb.translate_point_array(m0b2, [0, 0, dz])
        _, torq = glb.point_matrix_gravity(m0b2, m1)
        torques_z[k] = torq[2]
    assert(abs(torques_z-signal) < 1e3*np.finfo(float).eps).all()


def test_torque_a2():
    """
    Z-torque from rotating inner moments about y-axis, no translation

    Tests
    -----
    torque
    """
    # Rotating inner moments about y-axis, z-torque on inner
    m0 = np.array([[1, 1, 0, 0], [1, -1, 0, 0]])
    m1 = np.array([[1, .5, 2, 0], [1, -.5, -2, 0]])
    # rotate inner moment about x-axis into yz-plane
    m0b = glb.rotate_point_array(m0, np.pi/2, [1, 0, 0])
    
    angles = np.arange(0, 360, 10)*np.pi/180
    torques_z = np.zeros(len(angles))
    
    # Compute as moments immediately
    L = 20
    d0 = pgm.qmoments(L, m0)
    d1 = pgm.Qmomentsb(L, m1)
    # Euler angle is -pi/2, pi/2, pi/2 to get to x-axis rotation
    tx, tcx, tsx = mplb.torque(d0, d1, [np.pi/2, -np.pi/2, -np.pi/2], [0, 0, 0])
    signal = mplb.torques_at_angle(tcx, tsx, angles)
    
    # points rotating about y-axis
    for k, angle in enumerate(angles):
        m0b2 = glb.rotate_point_array(m0b, angle, [0, 1, 0])
        _, torq = glb.point_matrix_gravity(m0b2, m1)
        torques_z[k] = torq[2]
    assert(abs(torques_z-signal) < 10*np.finfo(float).eps).all()


def test_torque_a3():
    """
    Z-torque from rotating inner moments about y-axis, no translation

    Tests
    -----
    torque
    """
    # Rotating inner moments about y-axis, z-torque on inner
    m0 = np.array([[1, 1, 0, 0], [1, -.5, np.sqrt(3)/2, 0], [1, -.5, -np.sqrt(3)/2, 0]])
    m1 = np.array([[1, 0, 2, 0], [1, -np.sqrt(3), -1, 0], [1, -np.sqrt(3), 1, 0]])
    m0 = glb.rotate_point_array(m0, np.pi/8, [0, 0, 1])
    # rotate inner moment about x-axis into yz-plane
    m0b = glb.rotate_point_array(m0, np.pi/2, [1, 0, 0])
    
    angles = np.arange(0, 360, 10)*np.pi/180
    torques_z = np.zeros(len(angles))
    
    # Compute as moments immediately
    L = 20
    d0 = pgm.qmoments(L, m0)
    d1 = pgm.Qmomentsb(L, m1)
    # Euler angle is -pi/2, pi/2, pi/2 to get to x-axis rotation
    tx, tcx, tsx = mplb.torque(d0, d1, [np.pi/2, -np.pi/2, -np.pi/2], [0, 0, 0])
    signal = mplb.torques_at_angle(tcx, tsx, angles)
    
    # points rotating about y-axis
    for k, angle in enumerate(angles):
        m0b2 = glb.rotate_point_array(m0b, angle, [0, 1, 0])
        _, torq = glb.point_matrix_gravity(m0b2, m1)
        torques_z[k] = torq[2]
    assert(abs(torques_z-signal) < 10*np.finfo(float).eps).all()


def test_torque_a4():
    """
    Z-torque from rotating inner moments about x-axis, no translation

    Tests
    -----
    torque
    """
    # Rotating inner moments about x-axis, z-torque on inner
    m0 = np.array([[1, 1, 0, 0], [1, -1, 0, 0]])
    m1 = np.array([[1, .5, 2, 0], [1, -.5, -2, 0]])
    m0b = glb.rotate_point_array(m0, np.pi/2, [0, 1, 0])

    angles = np.arange(0, 360, 10)*np.pi/180
    torques_z = np.zeros(len(angles))

    # Compute as moments immediately
    L = 20
    d0 = pgm.qmoments(L, m0)
    d1 = pgm.Qmomentsb(L, m1)
    tx, tcx, tsx = mplb.torque(d0, d1, [0, np.pi/2, 0], [0, 0, 0])
    signal = mplb.torques_at_angle(tcx, tsx, angles)

    for k, angle in enumerate(angles):
        m0b2 = glb.rotate_point_array(m0b, angle, [1, 0, 0])
        _, torq = glb.point_matrix_gravity(m0b2, m1)
        torques_z[k] = torq[2]
    assert(abs(torques_z-signal) < 10*np.finfo(float).eps).all()



def test_torque_b2():
    """
    Z-torque from rotating outer moments about y-axis, no translation

    Tests
    -----
    torque
    """
    # Rotating outer moments about y-axis, z-torque on inner
    m0 = np.array([[1, 1, 0, 0], [1, -.5, np.sqrt(3)/2, 0], [1, -.5, -np.sqrt(3)/2, 0]])
    m1 = np.array([[1, 0, 2, 0], [1, -np.sqrt(3), -1, 0], [1, -np.sqrt(3), 1, 0]])
    m0 = glb.rotate_point_array(m0, np.pi/8, [0, 0, 1])
    m1b = glb.rotate_point_array(m1, np.pi/2, [1, 0, 0])
    
    angles = np.arange(0, 360, 10)*np.pi/180
    torques_z = np.zeros(len(angles))
    
    # Compute as moments immediately
    L = 20
    d0 = pgm.qmoments(L, m0)
    d1 = pgm.Qmomentsb(L, m1)
    tx, tcx, tsx = mplb.torque(d0, d1, [np.pi/2, -np.pi/2, -np.pi/2], [0, 0, 0],
                               'outer-outer')
    signal = mplb.torques_at_angle(tcx, tsx, angles, outer=True)

    for k, angle in enumerate(angles):
        m1b2 = glb.rotate_point_array(m1b, angle, [0, 1, 0])
        #m1b2 = glb.translate_point_array(m1b2, [0, 0, 4])
        _, torq = glb.point_matrix_gravity(m0, m1b2)
        torques_z[k] = torq[2]
    assert(abs(torques_z-signal) < 10*np.finfo(float).eps).all()


def test_torque_b3():
    """
    Z-torque from rotating outer moments about y-axis, no translation

    Tests
    -----
    torque
    """
    # Rotating outer moments about y-axis, z-torque on inner
    m0 = np.array([[1, 1, .5, 0], [1, -1, -.5, 0]])
    m1 = np.array([[1, 0, 2, 0], [1, 0, -2, 0]])
    m1b = glb.rotate_point_array(m1, np.pi/2, [1, 0, 0])
    
    angles = np.arange(0, 360, 10)*np.pi/180
    torques_z = np.zeros(len(angles))
    
    # Compute as moments immediately
    L = 20
    d0 = pgm.qmoments(L, m0)
    d1 = pgm.Qmomentsb(L, m1)
    tx, tcx, tsx = mplb.torque(d0, d1, [np.pi/2, -np.pi/2, -np.pi/2], [0, 0, 0],
                               'outer-outer')
    signal = mplb.torques_at_angle(tcx, tsx, angles, outer=True)

    for k, angle in enumerate(angles):
        m1b2 = glb.rotate_point_array(m1b, angle, [0, 1, 0])
        #m1b2 = glb.translate_point_array(m1b2, [0, 0, 4])
        _, torq = glb.point_matrix_gravity(m0, m1b2)
        torques_z[k] = torq[2]
    assert(abs(torques_z-signal) < 10*np.finfo(float).eps).all()


def test_torque_b4():
    """
    Z-torque from rotating outer moments about x-axis, no translation

    Tests
    -----
    torque
    """
    # Rotating outer moments about y-axis, z-torque on inner
    m0 = np.array([[1, 1, 0, 0], [1, -.5, np.sqrt(3)/2, 0], [1, -.5, -np.sqrt(3)/2, 0]])
    m1 = np.array([[1, 0, 2, 0], [1, -np.sqrt(3), -1, 0], [1, -np.sqrt(3), 1, 0]])
    m0 = glb.rotate_point_array(m0, np.pi/8, [0, 0, 1])
    m1b = glb.rotate_point_array(m1, np.pi/2, [0, 1, 0])
    
    angles = np.arange(0, 360, 10)*np.pi/180
    torques_z = np.zeros(len(angles))
    
    # Compute as moments immediately
    L = 20
    d0 = pgm.qmoments(L, m0)
    d1 = pgm.Qmomentsb(L, m1)
    tx, tcx, tsx = mplb.torque(d0, d1, [0, np.pi/2, 0], [0, 0, 0],
                               'outer-outer')
    signal = mplb.torques_at_angle(tcx, tsx, angles, outer=True)

    for k, angle in enumerate(angles):
        m1b2 = glb.rotate_point_array(m1b, angle, [1, 0, 0])
        #m1b2 = glb.translate_point_array(m1b2, [0, 0, 4])
        _, torq = glb.point_matrix_gravity(m0, m1b2)
        torques_z[k] = torq[2]
    assert(abs(torques_z-signal) < 10*np.finfo(float).eps).all()



def test_torque_c2():
    """
    Z-torque from rotating outer moments about x-axis and translated from inner
    moments to outer moments (large translation compared to outer moment radius)

    Tests
    -----
    torque
    """
    m0 = np.array([[1, 1, .5, 0], [1, -1, -.5, 0]])
    m1 = np.array([[1, 0, 1, 0], [1, 0, -1, 0]])
    m1b = glb.rotate_point_array(m1, np.pi/2, [1, 0, 0])

    angles = np.arange(0, 360, 10)*np.pi/180
    torques_z = np.zeros(len(angles))

    # Compute as moments immediately
    L = 20
    d0 = pgm.qmoments(L, m0)
    d1 = pgm.qmoments(L, m1)
    dz = 5
    tx, tcx, tsx = mplb.torque(d0, d1, [np.pi/2, -np.pi/2, -np.pi/2], [0, 0, dz],
                               'inner-outer')
    signal = mplb.torques_at_angle(tcx, tsx, angles)

    for k, angle in enumerate(angles):
        m1b2 = glb.rotate_point_array(m1b, angle, [0, 1, 0])
        m1b2 = glb.translate_point_array(m1b2, [0, 0, dz])
        _, torq = glb.point_matrix_gravity(m0, m1b2)
        torques_z[k] = torq[2]
    assert(abs(torques_z-signal) < 10*np.finfo(float).eps).all()


def test_torque_c3():
    """
    Z-torque from rotating outer moments about x-axis and translated from inner
    moments to outer moments (large translation compared to outer moment radius)

    Tests
    -----
    torque
    """
    m0 = np.array([[1, 1, 0, 0], [1, -.5, np.sqrt(3)/2, 0], [1, -.5, -np.sqrt(3)/2, 0]])
    m1 = np.array([[1, 0, 1, 0], [1, -np.sqrt(3)/2, -1/2, 0], [1, -np.sqrt(3)/2, 1/2, 0]])
    m0 = glb.rotate_point_array(m0, np.pi/8, [0, 0, 1])
    m1b = glb.rotate_point_array(m1, np.pi/2, [0, 1, 0])

    angles = np.arange(0, 360, 10)*np.pi/180
    torques_z = np.zeros(len(angles))

    # Compute as moments immediately
    L = 20
    d0 = pgm.qmoments(L, m0)
    d1 = pgm.qmoments(L, m1)
    dz = 5
    tx, tcx, tsx = mplb.torque(d0, d1, [0, np.pi/2, 0], [0, 0, dz],
                               'inner-outer')
    signal = mplb.torques_at_angle(tcx, tsx, angles)

    for k, angle in enumerate(angles):
        m1b2 = glb.rotate_point_array(m1b, angle, [1, 0, 0])
        m1b2 = glb.translate_point_array(m1b2, [0, 0, dz])
        _, torq = glb.point_matrix_gravity(m0, m1b2)
        torques_z[k] = torq[2]
    assert(abs(torques_z-signal) < 10*np.finfo(float).eps).all()


def test_torque_c4():
    """
    Z-torque from rotating outer moments about y-axis and translated from inner
    moments to outer moments (large translation compared to outer moment radius)

    Tests
    -----
    torque
    """
    m0 = np.array([[1, 1, 0, 0], [1, -.5, np.sqrt(3)/2, 0], [1, -.5, -np.sqrt(3)/2, 0]])
    m1 = np.array([[1, 0, 1, 0], [1, -np.sqrt(3)/2, -1/2, 0], [1, -np.sqrt(3)/2, 1/2, 0]])
    m0 = glb.rotate_point_array(m0, np.pi/8, [0, 0, 1])
    m1b = glb.rotate_point_array(m1, np.pi/2, [1, 0, 0])

    angles = np.arange(0, 360, 10)*np.pi/180
    torques_z = np.zeros(len(angles))

    # Compute as moments immediately
    L = 20
    d0 = pgm.qmoments(L, m0)
    d1 = pgm.qmoments(L, m1)
    dz = 5
    tx, tcx, tsx = mplb.torque(d0, d1, [np.pi/2, -np.pi/2, -np.pi/2], [0, 0, dz],
                               'inner-outer')
    signal = mplb.torques_at_angle(tcx, tsx, angles)

    for k, angle in enumerate(angles):
        m1b2 = glb.rotate_point_array(m1b, angle, [0, 1, 0])
        m1b2 = glb.translate_point_array(m1b2, [0, 0, dz])
        _, torq = glb.point_matrix_gravity(m0, m1b2)
        torques_z[k] = torq[2]
    assert(abs(torques_z-signal) < 10*np.finfo(float).eps).all()


def test_torque_d2():
    """
    Z-torque from rotating outer moments about x-axis and translated from outer
    moments to outer moments (small translation compared to outer moment radius)

    Tests
    -----
    torque
    """
    m0 = np.array([[1, 1, .5, 0], [1, -1, -.5, 0]])
    m1 = np.array([[1, 0, 2, 0], [1, 0, -2, 0]])
    m1b = glb.rotate_point_array(m1, np.pi/2, [1, 0, 0])

    angles = np.arange(0, 360, 10)*np.pi/180
    torques_z = np.zeros(len(angles))

    # Compute as moments immediately
    L = 20
    d0 = pgm.qmoments(L, m0)
    d1 = pgm.Qmomentsb(L, m1)
    dz = .9
    tx, tcx, tsx = mplb.torque(d0, d1, [np.pi/2, -np.pi/2, -np.pi/2], [0, 0, dz], 'outer-outer')
    signal = mplb.torques_at_angle(tcx, tsx, angles, outer=True)

    for k, angle in enumerate(angles):
        m1b2 = glb.rotate_point_array(m1b, angle, [0, 1, 0])
        m1b2 = glb.translate_point_array(m1b2, [0, 0, dz])
        _, torq = glb.point_matrix_gravity(m0, m1b2)
        torques_z[k] = torq[2]
    assert(abs(torques_z-signal) < 4e3*np.finfo(float).eps).all()


def test_torque_d3():
    """
    Z-torque from rotating outer moments about x-axis and translated from outer
    moments to outer moments (small translation compared to outer moment radius)

    Tests
    -----
    torque
    """
    m0 = np.array([[1, 1, 0, 0], [1, -.5, np.sqrt(3)/2, 0], [1, -.5, -np.sqrt(3)/2, 0]])
    m1 = np.array([[1, 0, 2, 0], [1, -np.sqrt(3), -1, 0], [1, -np.sqrt(3), 1, 0]])
    m0 = glb.rotate_point_array(m0, np.pi/8, [0, 0, 1])
    m1b = glb.rotate_point_array(m1, np.pi/2, [0, 1, 0])

    angles = np.arange(0, 360, 10)*np.pi/180
    torques_z = np.zeros(len(angles))

    # Compute as moments immediately
    L = 20
    d0 = pgm.qmoments(L, m0)
    d1 = pgm.Qmomentsb(L, m1)
    dz = .9
    tx, tcx, tsx = mplb.torque(d0, d1, [0, np.pi/2, 0], [0, 0, dz], 'outer-outer')
    signal = mplb.torques_at_angle(tcx, tsx, angles, outer=True)

    for k, angle in enumerate(angles):
        m1b2 = glb.rotate_point_array(m1b, angle, [1, 0, 0])
        m1b2 = glb.translate_point_array(m1b2, [0, 0, dz])
        _, torq = glb.point_matrix_gravity(m0, m1b2)
        torques_z[k] = torq[2]
    assert(abs(torques_z-signal) < 4e3*np.finfo(float).eps).all()


def test_torque_d4():
    """
    Z-torque from rotating outer moments about y-axis and translated from outer
    moments to outer moments (small translation compared to outer moment radius)

    Tests
    -----
    torque
    """
    m0 = np.array([[1, 1, 0, 0], [1, -.5, np.sqrt(3)/2, 0], [1, -.5, -np.sqrt(3)/2, 0]])
    m1 = np.array([[1, 0, 2, 0], [1, -np.sqrt(3), -1, 0], [1, -np.sqrt(3), 1, 0]])
    m0 = glb.rotate_point_array(m0, np.pi/8, [0, 0, 1])
    m1b = glb.rotate_point_array(m1, np.pi/2, [1, 0, 0])

    angles = np.arange(0, 360, 10)*np.pi/180
    torques_z = np.zeros(len(angles))

    # Compute as moments immediately
    L = 20
    d0 = pgm.qmoments(L, m0)
    d1 = pgm.Qmomentsb(L, m1)
    dz = .9
    tx, tcx, tsx = mplb.torque(d0, d1, [np.pi/2, -np.pi/2, -np.pi/2], [0, 0, dz], 'outer-outer')
    signal = mplb.torques_at_angle(tcx, tsx, angles, outer=True)

    for k, angle in enumerate(angles):
        m1b2 = glb.rotate_point_array(m1b, angle, [0, 1, 0])
        m1b2 = glb.translate_point_array(m1b2, [0, 0, dz])
        _, torq = glb.point_matrix_gravity(m0, m1b2)
        torques_z[k] = torq[2]
    assert(abs(torques_z-signal) < 4e3*np.finfo(float).eps).all()



def test_torque_e2():
    """
    Z-torque from rotating inner moments about x-axis and translated from inner
    moments to inner moments (moments still separated in radius)

    Tests
    -----
    torque
    """
    m0 = np.array([[1, 1, 0, 0], [1, -1, 0, 0]])
    m1 = np.array([[1, .5, 2, 0], [1, -.5, -2, 0]])
    m0b = glb.rotate_point_array(m0, np.pi/2, [1, 0, 0])

    angles = np.arange(0, 360, 10)*np.pi/180
    torques_z = np.zeros(len(angles))

    # Compute as moments immediately
    L = 20
    d0 = pgm.qmoments(L, m0)
    d1 = pgm.Qmomentsb(L, m1)
    dz = .5
    tx, tcx, tsx = mplb.torque(d0, d1, [np.pi/2, -np.pi/2, -np.pi/2], [0, 0, dz])
    signal = mplb.torques_at_angle(tcx, tsx, angles)

    for k, angle in enumerate(angles):
        m0b2 = glb.rotate_point_array(m0b, angle, [0, 1, 0])
        m0b2 = glb.translate_point_array(m0b2, [0, 0, dz])
        _, torq = glb.point_matrix_gravity(m0b2, m1)
        torques_z[k] = torq[2]
    assert(abs(torques_z-signal) < 1e3*np.finfo(float).eps).all()


def test_torque_e3():
    """
    Z-torque from rotating inner moments about x-axis and translated from inner
    moments to inner moments (moments still separated in radius)

    Tests
    -----
    torque
    """
    m0 = np.array([[1, 1, 0, 0], [1, -.5, np.sqrt(3)/2, 0], [1, -.5, -np.sqrt(3)/2, 0]])
    m1 = np.array([[1, 0, 2, 0], [1, -np.sqrt(3), -1, 0], [1, -np.sqrt(3), 1, 0]])
    m1 = glb.rotate_point_array(m1, np.pi/8, [0, 0, 1])
    m0b = glb.rotate_point_array(m0, np.pi/2, [0, 1, 0])
    
    angles = np.arange(0, 360, 10)*np.pi/180
    torques_z = np.zeros(len(angles))
    
    # Compute as moments immediately
    L = 20
    d0 = pgm.qmoments(L, m0)
    d1 = pgm.Qmomentsb(L, m1)
    dz = .5
    tx, tcx, tsx = mplb.torque(d0, d1, [0, np.pi/2, 0], [0, 0, dz])
    signal = mplb.torques_at_angle(tcx, tsx, angles)
    
    for k, angle in enumerate(angles):
        m0b2 = glb.rotate_point_array(m0b, angle, [1, 0, 0])
        m0b2 = glb.translate_point_array(m0b2, [0, 0, dz])
        _, torq = glb.point_matrix_gravity(m0b2, m1)
        torques_z[k] = torq[2]
    assert(abs(torques_z-signal) < 1e3*np.finfo(float).eps).all()


def test_torque_e4():
    """
    Z-torque from rotating inner moments about y-axis and translated from inner
    moments to inner moments (moments still separated in radius)

    Tests
    -----
    torque
    """
    m0 = np.array([[1, 1, 0, 0], [1, -.5, np.sqrt(3)/2, 0], [1, -.5, -np.sqrt(3)/2, 0]])
    m1 = np.array([[1, 0, 2, 0], [1, -np.sqrt(3), -1, 0], [1, -np.sqrt(3), 1, 0]])
    m1 = glb.rotate_point_array(m1, np.pi/8, [0, 0, 1])
    m0b = glb.rotate_point_array(m0, np.pi/2, [1, 0, 0])

    angles = np.arange(0, 360, 10)*np.pi/180
    torques_z = np.zeros(len(angles))

    # Compute as moments immediately
    L = 20
    d0 = pgm.qmoments(L, m0)
    d1 = pgm.Qmomentsb(L, m1)
    dz = .5
    tx, tcx, tsx = mplb.torque(d0, d1, [np.pi/2, -np.pi/2, -np.pi/2], [0, 0, dz])
    signal = mplb.torques_at_angle(tcx, tsx, angles)

    for k, angle in enumerate(angles):
        m0b2 = glb.rotate_point_array(m0b, angle, [0, 1, 0])
        m0b2 = glb.translate_point_array(m0b2, [0, 0, dz])
        _, torq = glb.point_matrix_gravity(m0b2, m1)
        torques_z[k] = torq[2]
    assert(abs(torques_z-signal) < 1e3*np.finfo(float).eps).all()
