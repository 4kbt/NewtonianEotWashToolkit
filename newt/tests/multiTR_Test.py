# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 20:10:14 2020

@author: John
"""
import numpy as np
import newt.glib as glb
import newt.pg2Multi as pgm
import newt.translations as trs
import newt.translationRecurs as trr
import newt.rotations as rot


def test_q2Q():
    """
    Check that the inner to outer translate method matches PointGravity. This
    translation method is worse for smaller translations. For R=10, the error
    approaches 1e7*epsilon.
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
    Qlm = trs.translate_q2Q(qm0, [dr, 0, 0])
    Qlmb = trs.translate_q2Q(qm0b, [-dr, 0, 0])
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
    qlmp = trs.translate_qlm(qm1, [0.1, 0, 0])
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
    qlmp = trs.translate_qlm(qm1, [rp, 0, 0])
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
    Qlmp2 = trs.translate_Qlmb(Qm2, [0.1, 0, 0])
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
    Qlmp = trs.translate_Qlmb(Qm1, [rp, 0, 0])
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
    Qlmp2 = trs.translate_Qlmb(Qm2, [5, 0, 0])
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


def test_rotateAB():
    d = 1
    m = 1
    m1 = np.array([[m, d, 0, 0]])
    qm1 = pgm.qmoments(10, m1)
    alpha = -np.pi/4
    beta = np.pi/4
    # Rotate around z first by alpha
    m2 = glb.rotate_point_array(m1, alpha, [0, 0, 1])
    # Rotate around y second by beta and get new moments
    qm1b = pgm.qmoments(10, glb.rotate_point_array(m2, beta, [0, 1, 0]))
    qm1c = rot.rotate_qlm(qm1, 0, beta, alpha)
    assert (abs(qm1c-qm1b) < 20*np.finfo(float).eps).all()


def test_rotateABC():
    d = 1
    m = 1
    m1 = np.array([[m, d, 0, 0], [m, 0, d, 0], [m, 0, 0, d]])
    qm1 = pgm.qmoments(10, m1)
    alpha = np.pi/4
    beta = np.pi/2
    gamma = np.pi/3
    m1b = glb.rotate_point_array(m1, alpha, [0, 0, 1])
    m1b2 = glb.rotate_point_array(m1b, beta, m1b[1, 1:]/d)
    m1b3 = glb.rotate_point_array(m1b2, gamma, m1b2[2, 1:]/d)
    qm1b = pgm.qmoments(10, m1b3)
    qm1c = rot.rotate_qlm(qm1, alpha, beta, gamma)
    assert (abs(qm1c-qm1b) < 15*np.finfo(float).eps).all()


def test_rotateABC2():
    d = 1
    m = 1
    m1 = np.array([[m, d, 0, 0], [m, 0, d, 0], [m, 0, 0, d]])
    qm1 = pgm.Qmomentsb(10, m1)
    alpha = np.pi/4
    beta = np.pi/2
    gamma = np.pi/3
    m1b = glb.rotate_point_array(m1, alpha, [0, 0, 1])
    m1b2 = glb.rotate_point_array(m1b, beta, m1b[1, 1:]/d)
    m1b3 = glb.rotate_point_array(m1b2, gamma, m1b2[2, 1:]/d)
    qm1b = pgm.Qmomentsb(10, m1b3)
    qm1c = rot.rotate_qlm(qm1, -alpha, beta, -gamma)
    assert (abs(qm1c-qm1b) < 15*np.finfo(float).eps).all()


def test_rotateABC_rand():
    d = 1
    m = 1
    m1 = np.array([[m, d, 0, 0], [m, 0, d, 0], [m, 0, 0, d]])
    qm1 = pgm.qmoments(10, m1)
    for k in range(10):
        alpha, beta, gamma = np.random.rand(3)*np.pi*2
        m1b = glb.rotate_point_array(m1, alpha, [0, 0, 1])
        m1b2 = glb.rotate_point_array(m1b, beta, m1b[1, 1:]/d)
        m1b3 = glb.rotate_point_array(m1b2, gamma, m1b2[2, 1:]/d)
        qm1b = pgm.qmoments(10, m1b3)
        qm1c = rot.rotate_qlm(qm1, alpha, beta, gamma)
        assert (abs(qm1c-qm1b) < 100*np.finfo(float).eps).all()


def test_rotateABC_rand2():
    d = 1
    m = 1
    m1 = np.array([[m, d, 0, 0], [m, 0, d, 0], [m, 0, 0, d]])
    qm1 = pgm.Qmomentsb(10, m1)
    for k in range(10):
        alpha, beta, gamma = np.random.rand(3)*np.pi*2
        m1b = glb.rotate_point_array(m1, alpha, [0, 0, 1])
        m1b2 = glb.rotate_point_array(m1b, beta, m1b[1, 1:]/d)
        m1b3 = glb.rotate_point_array(m1b2, gamma, m1b2[2, 1:]/d)
        qm1b = pgm.Qmomentsb(10, m1b3)
        qm1c = rot.rotate_qlm(qm1, -alpha, beta, -gamma)
        assert (abs(qm1c-qm1b) < 100*np.finfo(float).eps).all()


def test_q2q_trr():
    """
    Check that the inner to inner translate method matches P&S translation.
    """
    d = 1
    m = 1
    m1 = np.array([[m, d, 0, 0], [m, 0, d, 0], [m, 0, 0, d]])
    qm1 = pgm.qmoments(10, m1)
    r = np.array([-.5, .5, 1.5])
    qm1p = pgm.qmoments(10, glb.translate_point_array(m1, r))
    qm1t = trr.translate_qlm(qm1, r)
    assert (abs(qm1p-qm1t) < 5e5*np.finfo(float).eps).all()


def test_Q2Q_trr():
    """
    Check that the outer to outer translate method matches P&S translation.
    """
    d = 20
    m = 1
    L = 10
    m1 = np.array([[m, d, 0, 0], [m, 0, d, 0], [m, 0, 0, d]])
    qm1 = pgm.Qmomentsb(L, m1)
    r = np.array([-.5, .5, .7])
    qm1p = pgm.Qmomentsb(L, glb.translate_point_array(m1, r))
    qm1t = trr.translate_Qlmb(qm1, r)
    assert (abs(qm1p-qm1t) < 1e5*np.finfo(float).eps).all()


def test_q2Q_trr():
    """
    Check that the outer to outer translate method matches P&S translation.
    """
    d = 1
    m = 1
    L = 10
    m1 = np.array([[m, d, 0, 0], [m, 0, d, 0], [m, 0, 0, d]])
    qm1 = pgm.qmoments(L, m1)
    r = np.array([-.5, .5, .7])*10
    qm1p = pgm.Qmomentsb(L, glb.translate_point_array(m1, r))
    qm1t = trr.translate_q2Q(qm1, r)
    assert (abs(qm1p-qm1t) < 1e5*np.finfo(float).eps).all()


def test_q2q_RR():
    """
    Check that the inner to inner translate method matches EGA translation.
    """
    d = 1
    m = 1
    m1 = np.array([[m, d, 0, 0], [m, -d, 0, 0]])
    # Find inner moments around origin
    qm1 = pgm.qmoments(10, m1)
    # Find inner moments if translated by [.1, 0, 0]
    qm1p = pgm.qmoments(10, glb.translate_point_array(m1, [0, 0, 0.1]))
    # Find moments translated by [.1, 0, 0]
    rrms = trr.transl_newt_z_RR(10, .1)
    qlmp = trr.apply_trans_mat(qm1, rrms)
    assert (abs(qlmp-qm1p) < 11*np.finfo(float).eps).all()


def test_Q2Q_SS():
    """
    Check that the outer to outer translate method matches CS translation.
    """
    R = 100
    M = 1
    m2 = np.array([[M, R, 0, 0], [M, -R, 0, 0]])
    # Get outer moments of points at +/-R
    Qm2 = pgm.Qmomentsb(10, m2)
    # Get outer moments of translated points
    Qm2b = pgm.Qmomentsb(10, glb.translate_point_array(m2, [0, 0, 0.1]))
    # Find outer moments from inner qm0 and qm0b translated to +/-R
    ssms = trr.transl_newt_z_SS(10, .1)
    Qlmp2 = trr.apply_trans_mat(Qm2, ssms)
    assert (abs(Qlmp2-Qm2b) < 10*np.finfo(float).eps).all()


def test_q2Q_SR():
    """
    Check that the inner to outer translate method matches PointGravity. This
    translation method is worse for smaller translations. For R=10, the error
    approaches 1e7*epsilon.
    """
    d = 1
    R = 100
    m = 1
    m1 = np.array([[m, d, 0, 0], [m, -d, 0, 0]])
    m2 = glb.translate_point_array(m1, [0, 0, R])
    # Create inner moments of each points at +/-r
    qm0 = pgm.qmoments(10, np.array(m1))
    # Get outer moments of points at +/-R
    Qm2 = pgm.Qmomentsb(10, m2)
    # Find outer moments from inner qm0 and qm0b translated to +/-R
    srms = trr.transl_newt_z_SR(10, R)
    Qlm = trr.apply_trans_mat(qm0, srms, q2Q=True)
    assert (abs(Qlm-Qm2) < 11*np.finfo(float).eps).all()


def translate_q2Q_full(qlm, rPrime):
    r"""
    Takes in an inner set of moments, q_lm, and returns the translated outer
    moments Q_LM. The translation vector is rPrime.

    Inputs
    ------
    qlm : ndarray
        (l+1)x(2l+1) array of lowest order inner multipole moments.
    rPrime : ndarray
        [x,y,z] translation from the origin where q_lm components were computed

    Returns
    -------
    QLM : ndarray
        (l+1)x(2l+1) array of lowest order outer multipole moments. Moments are
        for the translated moments by rPrime.

    References
    ----------
    https://journals.aps.org/prd/abstract/10.1103/PhysRevD.60.107501
    """
    rP = np.sqrt(rPrime[0]**2+rPrime[1]**2+rPrime[2]**2)
    phiP = np.arctan2(rPrime[1], rPrime[0])
    thetaP = np.arccos(rPrime[2]/rP)
    # Restrict LMax to size of qlm for now
    Lqlm = np.shape(qlm)[0] - 1
    LMax = Lqlm

    # Conjugate spherical harmonics
    ylmS = np.zeros([2*LMax+1, 4*LMax+1], dtype='complex')
    QLM = np.zeros([LMax+1, 2*LMax+1], dtype='complex')
    srms = []
    for m in range(LMax+1):
        srm = np.zeros([LMax+1-m, LMax+1-m], dtype=complex)
        srms.append(srm)

    for l in range(2*LMax+1):
        ms = np.arange(-l, l+1)
        sphHarmL = trs.sp.sph_harm(ms, l, phiP, thetaP)
        ylmS[l, 2*LMax-l:2*LMax+l+1] = sphHarmL

    for L in range(LMax+1):
        for M in range(-L, L+1):
            for lP in range(LMax+1):
                l = L+lP
                rPl1 = rP**(l+1)
                gamsum = trs.sp.gammaln(2*l+1)
                gamsum -= trs.sp.gammaln(2*lP+2)+trs.sp.gammaln(2*L+1)
                fac = np.sqrt(4.*np.pi*np.exp(gamsum))
                fac /= rPl1
                for mP in range(-lP, lP+1):
                    m = M + mP
                    if (abs(m) <= l):
                        cFac = fac*trs.cg.cgCoeff(lP, l, -mP, m, L, M)*(-1)**mP
                        if m == 0 and M >= 0:
                            srms[M][L-M, lP-M] = cFac*ylmS[l, 2*LMax+m]
                        #else:
                            #print(cFac*ylmS[l, LMax+m])
                        QLM[L, LMax+M] += cFac*ylmS[l, 2*LMax+m]*qlm[lP, LMax+mP]

    return QLM, srms


def test_srms_xy():
    d = .1
    R = 10
    m = 1
    LMax = 10
    m1 = np.array([[m, d, 0, 0], [m, 0, d, 0]])
    m2 = glb.translate_point_array(m1, [0, 0, R])
    # Create inner moments of each points at +/-r
    qm0 = pgm.qmoments(LMax, np.array(m1))
    # Get outer moments of points at +/-R
    Qm2 = pgm.Qmomentsb(LMax, m2)
    # Find outer moments from inner qm0 and qm0b translated to +/-R
    srms = trr.transl_newt_z_SR(LMax, R)
    Qlm = trr.apply_trans_mat(qm0, srms, q2Q=True)
    QLM, SRMS = translate_q2Q_full(qm0, [0, 0, R])
    SRMS = [np.real(SRM) for SRM in SRMS]
    Qm3 = trr.apply_trans_mat(qm0, SRMS, q2Q=True)
    for k in range(LMax+1):
        assert (np.abs((SRMS[k]-srms[k])/SRMS[k]) < 1e3*np.finfo(float).eps).all()
