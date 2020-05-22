# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 13:19:23 2020

@author: John Greendeer Lee
"""
import numpy as np
import numpy.random as rand
import newt.glib as glb
import newt.glibShapes as gshp
import newt.pg2Multi as pgm


def test_gmmr2_force():
    """
    Check that the force between to 1kg points at a meter = BIG_G in xHat
    and zero in y,z. Also, expect no torques on point at origin.

    Tests
    -----
    glb.point_matrix_gravity : function
    """
    m1 = np.array([1, 0, 0, 0])
    m2 = np.array([1, 1, 0, 0])
    f, t = glb.point_matrix_gravity(m1, m2)
    fG = glb.BIG_G
    assert abs(f[0] - fG) < 2.*np.finfo(float).eps
    assert abs(f[1:].all()) < 2.*np.finfo(float).eps
    assert abs(t.all()) < 2*np.finfo(float).eps


def test_yuk_force():
    """
    Check that the force between to 1kg points at a meter = BIG_G in xHat
    and zero in y,z. Also, expect no torques on point at origin.

    Tests
    -----
    glb.point_matrix_gravity : function
    """
    m1 = np.array([1, 0, 0, 0])
    m2 = np.array([1, 1, 0, 0])
    f, t = glb.point_matrix_yukawa(m1, m2, 1, 1)
    fY = glb.BIG_G*2/np.e
    assert abs(f[0] - fY) < 2.*np.finfo(float).eps
    assert (abs(f[1:]) < 2.*np.finfo(float).eps).all()
    assert (abs(t) < 2*np.finfo(float).eps).all()


def test_gmmr2_torque():
    """
    Check that the force between to 1kg points at a meter = BIG_G in xHat
    and zero in y,z. Also, expect no torques on point at origin.

    Tests
    -----
    glb.point_matrix_gravity : function
    """
    m1 = np.array([1, 0, 1, 0])
    m2 = np.array([1, 1, 1, 0])
    f, t = glb.point_matrix_gravity(m1, m2)
    fG = glb.BIG_G
    assert abs(f[0] - fG) < 2.*np.finfo(float).eps
    assert (abs(f[1:]) < 2.*np.finfo(float).eps).all()
    assert (abs(t[:2]) < 2*np.finfo(float).eps).all()
    assert abs(t[2] + fG) < 2*np.finfo(float).eps


def test_yuk_torque():
    """
    Check that the force between to 1kg points at a meter = BIG_G in xHat
    and zero in y,z. Also, expect no torques on point at origin.

    Tests
    -----
    glb.point_matrix_gravity : function
    """
    m1 = np.array([1, 0, 1, 0])
    m2 = np.array([1, 1, 1, 0])
    f, t = glb.point_matrix_yukawa(m1, m2, 1, 1)
    fY = glb.BIG_G*2/np.e
    assert abs(f[0] - fY) < 2.*np.finfo(float).eps
    assert (abs(f[1:]) < 2.*np.finfo(float).eps).all()
    assert (abs(t[:2]) < 2*np.finfo(float).eps).all()
    assert abs(t[2] + fY) < 2*np.finfo(float).eps


def test_ISL():
    """
    Tests
    -----
    glb.point_matrix_gravity : function
    """
    m = gshp.annulus(1, 0, 1, 1, 10, 5)
    for k in range(100):
        # translate randomly in z in range (2, 10**5)m
        exp = rand.rand()*5+2
        d = 10.**exp
        # translate randomly around y=0 ~N(0,.01)
        r = rand.randn()*0.01
        md = glb.translate_point_array(m, [0, r, d])
        f, t = glb.point_matrix_gravity(m, md)
        # Check that there's roughly no torque
        assert np.sum(abs(t)) < 6*np.finfo(float).eps
        # check that the force falls like 1/d**2
        assert (abs(f[2]-glb.BIG_G/d**2)/(glb.BIG_G/d**2)) < 0.001
    return True


def test_sheet_uniformity():
    """
    Check that the gravitational force from an infinite sheet is approximately
    a constant as a function of distance.

    Tests
    -----
    glb.point_matrix_gravity : function
    """
    m = np.array([1, 0, 0, 0])
    N = 100
    r = 140
    t = 0.01
    xspace = 0.005
    rspace = 0.5
    mass = np.pi*r*r*t
    sheet = gshp.annulus(mass, 0, r, t, int(r/rspace), int(t/xspace))
    f = np.zeros([N, 3])
    ds = np.zeros(N)
    for k in range(N):
        x = rand.randn()*rspace
        y = rand.randn()*rspace
        ds[k] = 2*rand.rand() + rspace
        s = glb.translate_point_array(m, [x, y, ds[k]])
        f[k], torq = glb.point_matrix_gravity(s, sheet)
    expectedF = 2*np.pi*glb.BIG_G*t
    fs = np.array([f[k, 2] for k in range(N) if abs(ds[k]) > 1])
    assert (abs(abs(fs/expectedF) - 1) < 0.2).all()


def test_shell_theorem():
    """
    Check that Newton's shell theorem holds. That is a spherical shell behaves
    as a point of equal mass at the center of the shell. Additionally, a point
    inside the sphere experiences no force.

    Tests
    -----
    glb.point_matrix_gravity : function
    """
    d = 20
    R = 10
    shell = gshp.spherical_random_shell(1, R, 100000)
    # A point outside should see force from point at origin of shell mass 1kg
    m1 = np.array([1, d, 0, 0])
    F, T = glb.point_matrix_gravity(m1, shell)
    threshold = 6*np.finfo(float).eps
    assert abs(F[0]/(glb.BIG_G/d**2)+1) < .01
    assert abs(F[1]) < threshold
    assert abs(F[2]) < threshold
    # point inside should see no force
    m2 = np.array([1, 0, 0, 0])
    # translate randomly in box contained in shell
    Rlim = R*.5/np.sqrt(3)
    p = rand.rand(3)*2*Rlim - Rlim
    m2 = glb.translate_point_array(m2, p)
    F2, T2 = glb.point_matrix_gravity(m2, shell)
    # Fails so increased threshold
    assert (abs(F2) < 10*threshold).all()
    # assert (abs(T) < threshold).all()


def test_yukawa_shell_theorem():
    """
    The shell theorem also partly holds for the Yukawa potential. A point at
    the origin experiences no force from a spherical shell.
    """
    R = 10
    shell = gshp.spherical_random_shell(1, R, 100000)
    # A point outside should see force from point at origin of shell mass 1kg
    m1 = np.array([1, 0, 0, 0])
    F, T = glb.point_matrix_yukawa(m1, shell, 1, 1)
    threshold = 6*np.finfo(float).eps
    assert (abs(F) < 10*threshold).all()


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
    m1 = np.array([[m, d, 0, 0], [m, -d, 0, 0]])
    m2 = np.array([[M, R, 0, 0], [M, -R, 0, 0]])
    tau = np.zeros(N)
    ts = np.zeros([60, 3])
    d2R2 = d**2 + R**2
    for k in range(N):
        a = 2*np.pi*k/N
        ca = np.cos(a)
        Q = glb.rotate_point_array(m2, a, [0, 0, 1])
        tau[k] = 2*glb.BIG_G*M*m*d*R*np.sin(a)
        tau[k] *= 1/(d2R2-2*d*R*ca)**(3/2) - 1/(d2R2+2*d*R*ca)**(3/2)
        f, ts[k] = glb.point_matrix_gravity(m1, Q)
    assert (abs(tau-ts[:, 2]) < 10*np.finfo(float).eps).all()


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
    tau = np.zeros(N)
    ts = np.zeros([60, 3])
    d2R2 = d**2 + R**2 + z**2
    for k in range(N):
        a = 2*np.pi*k/N
        Q = glb.rotate_point_array(m3, a, zhat)
        fac = 3*glb.BIG_G*M*m*d*R
        tau[k] = np.sin(a)/(d2R2-2*d*R*np.cos(a))**(3/2)
        tau[k] += np.sin(a+2*np.pi/3)/(d2R2-2*d*R*np.cos(a+2*np.pi/3))**(3/2)
        tau[k] += np.sin(a+4*np.pi/3)/(d2R2-2*d*R*np.cos(a+4*np.pi/3))**(3/2)
        tau[k] *= fac
        f, ts[k] = glb.point_matrix_gravity(m1, Q)
    assert (abs(tau-ts[:, 2]) < 10*np.finfo(float).eps).all()


def test_rotate():
    """
    Check that you can rotate from xHat to yHat

    Tests
    -----
    glb.rotate_point_array : function
    """
    m = np.array([1, 1, 0, 0])
    o = glb.rotate_point_array(m, np.pi/2., [0, 0, 1])
    print(o)
    assert np.sum(o-np.array([1, 0, 1, 0])) < 4.*np.finfo(float).eps


def test_rotate2():
    """
    Check that rotating by 2 pi in N steps gets you back to start.

    Tests
    -----
    glb.rotate_point_array : function
    """
    m = np.array([1, 1, 0, 0])
    err = np.zeros([100, 4])
    for n in range(1, 101):
        q = np.copy(m)
        rvec = rand.rand(3)
        for k in range(1, n+1):
            q = glb.rotate_point_array(q, 2*np.pi/n, rvec)
        err[n-1] = m-q
    assert (abs(err) < 40*np.finfo(float).eps).all()


def test_rotate3():
    """
    Check that rotating by a random angle makes sense

    Tests
    -----
    glb.rotate_point_array : function
    """
    q = np.array([1, 1, 0, 0])
    N = 100
    v = np.zeros([N, 2])
    for k in range(1, N+1):
        a = 2*np.pi*rand.rand()
        p = glb.rotate_point_array(q, a, [0, 0, 1])
        v[k-1] = [a, (np.arctan(p[2]/p[1]) % np.pi/2) - (a % np.pi/2)]
    assert (abs(v[:, 1] < 10*np.finfo(float).eps)).all()


def test_translate():
    """
    Check that moving a point mass at the origin to [1, 1, 1] works.

    Tests
    -----
    glb.translate_point_array : function
    """
    m = np.array([1, 0, 0, 0])
    o = glb.translate_point_array(m, [1, 1, 1])
    assert (o == [1, 1, 1, 1]).all()


def test_translate2():
    """
    Check that translating 6 normally distributed masses keeps masses the same
    and deviation the same.

    Tests
    -----
    glb.translate_point_array : function
    """
    N = 6
    for k in range(100):
        v = rand.randn(3)
        m = rand.randn(N, 4)
        o = glb.translate_point_array(m, v)
        # check we didn't change mass
        assert (m[:, 0] == o[:, 0]).all()
        # Check positions match
        assert (abs(m[:, 1:] - o[:, 1:] + v) < 6*np.finfo(float).eps).all()


def test_ylm():
    """
    Check that the Y00 gives the correct value.

    Tests
    -----
    pgm.qmoment : function
    """
    y00 = pgm.qmoment(0, 0, np.array([[1, 1, 0, 0]]))
    assert abs(y00 - 1/np.sqrt(4*np.pi)) < 4*np.finfo(float).eps


def test_qmom():
    """
    Check the expression for the q33 moments of a point mass.

    Tests
    -----
    pgm.qmoment : function

    Reference
    ---------
    https://journals.aps.org/prd/abstract/10.1103/PhysRevD.55.7970
    """
    m = np.array([[1, 1, 0, 0], [-1, -1, 0, 0]])
    q33 = pgm.qmoment(3, 3, m)
    q11 = pgm.qmoment(1, 1, m)
    mb = np.array([[1, 2, 0, 0], [-1, -2, 0, 0]])
    q33b = pgm.qmoment(3, 3, mb)
    q11b = pgm.qmoment(1, 1, mb)
    assert abs(q33 + np.sqrt(35/16/np.pi)) < 10*np.finfo(float).eps
    assert abs(q33b + 8*np.sqrt(35/16/np.pi)) < 10*np.finfo(float).eps
    assert abs(q11 + np.sqrt(3/2/np.pi)) < 10*np.finfo(float).eps
    assert abs(q11b + 2*np.sqrt(3/2/np.pi)) < 10*np.finfo(float).eps


def test_Qmom():
    """
    Check the expression for the q33 moments of a point mass.

    Tests
    -----
    pgm.Qmomentb : function

    Reference
    ---------
    https://journals.aps.org/prd/abstract/10.1103/PhysRevD.55.7970
    """
    m = np.array([[1, 1, 0, 0], [-1, -1, 0, 0]])
    Q31 = pgm.Qmomentb(3, 1, m)
    Q33 = pgm.Qmomentb(3, 3, m)
    mb = np.array([[1, 2, 0, 0], [-1, -2, 0, 0]])
    Q31b = pgm.Qmomentb(3, 1, mb)
    Q33b = pgm.Qmomentb(3, 3, mb)
    assert abs(Q31 - np.sqrt(21/16/np.pi)) < 10*np.finfo(float).eps
    assert abs(Q33 + np.sqrt(35/16/np.pi)) < 10*np.finfo(float).eps
    assert abs(Q31b - np.sqrt(21/16/np.pi)/16) < 10*np.finfo(float).eps
    assert abs(Q33b + np.sqrt(35/16/np.pi)/16) < 10*np.finfo(float).eps
