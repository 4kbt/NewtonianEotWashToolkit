# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 16:34:18 2020

@author: John Greendeer Lee
"""
import numpy as np
import newt.rotations as rot


def test_d1():
    """
    Uses explicit d^1_m',m formulae to test recursive matrices
    """
    theta = np.arange(361)*np.pi/180
    d111 = .5*(1+np.cos(theta))
    d110 = -np.sin(theta)/np.sqrt(2)
    d11m1 = .5*(1-np.cos(theta))
    d100 = np.cos(theta)
    d1m11 = d11m1
    d1m1m1 = d111
    d1m10 = -d110
    d101 = (-1)*d110
    d10m1 = d110
    for k in range(len(theta)):
        beta = theta[k]
        pred = np.array([[d1m1m1[k], d1m10[k], d1m11[k]],
                         [d10m1[k], d100[k], d101[k]],
                         [d11m1[k], d110[k], d111[k]]])
        dlmns = rot.dlmn(2, beta)
        dlmns = rot.wignerDl(2, 0, beta, 0)
        d1 = rot.Dl(1, 0, beta, 0)
        assert (np.abs(dlmns[1] - pred) < 100*np.finfo(float).eps).all()
        assert (np.abs(d1 - pred) < 100*np.finfo(float).eps).all()


def test_d2():
    """
    Uses explicit d^2_m',m formulae to test recursive matrices
    """
    theta = np.arange(361)*np.pi/180
    stheta = np.sin(theta)
    ctheta = np.cos(theta)
    d222 = .25*(1+ctheta)**2
    d221 = -.5*stheta*(1+ctheta)
    d220 = np.sqrt(3/8)*stheta**2
    d22m1 = -.5*stheta*(1-ctheta)
    d22m2 = .25*(1-ctheta)**2
    d211 = .5*(2*ctheta**2+ctheta-1)
    d210 = -np.sqrt(3/8)*2*stheta*ctheta
    d21m1 = .5*(-2*ctheta**2+ctheta+1)
    d200 = .5*(3*ctheta**2-1)
    d2m2m2 = d222   # b
    d2m1m1 = d211   # b
    d212 = -d221    # a
    d2m2m1 = d212   # b
    d2m1m2 = -d2m2m1  # a
    d202 = d220     # a
    d201 = -d210    # a
    d2m12 = -d22m1  # a
    d2m11 = d21m1   # a
    d2m22 = d22m2   # a
    d20m1 = d210    # b
    d2m10 = -d20m1  # a
    d21m2 = d22m1   # b
    d2m21 = -d21m2  # a
    d20m2 = d220    # b
    d2m20 = d20m2
    for k in range(len(theta)):
        beta = theta[k]
        pred = np.array([[d2m2m2[k], d2m2m1[k], d2m20[k], d2m21[k], d2m22[k]],
                         [d2m1m2[k], d2m1m1[k], d2m10[k], d2m11[k], d2m12[k]],
                         [d20m2[k], d20m1[k], d200[k], d201[k], d202[k]],
                         [d21m2[k], d21m1[k], d210[k], d211[k], d212[k]],
                         [d22m2[k], d22m1[k], d220[k], d221[k], d222[k]]])
        dlmns = rot.dlmn(3, beta)
        d2 = rot.Dl(2, 0, beta, 0)
        assert (np.abs(dlmns[2] - pred) < 100*np.finfo(float).eps).all()
        assert (np.abs(d2 - pred) < 100*np.finfo(float).eps).all()


def test_symm():
    """
    d^j_{m'm}(pi) = -1^{j-m} delta_{m' m}
    """
    beta = np.pi
    lmax = 10
    dmpms = rot.dlmn(lmax, beta)
    for k in range(lmax):
        ms = np.arange(-k, k+1)
        nm = 2*k+1
        predk = np.fliplr(np.eye(nm))*np.outer(np.ones(nm), (-1)**np.abs(k-ms))
        assert (np.abs(dmpms[k] - predk) < 100*np.finfo(float).eps).all()


def test_symm2():
    """
    Wikipedia consistency checks on recursive d^j_{m'm}:

    Checks
    ------
    - d^j_{m'm} = -1^{m-m'} d^j_{m m'}
    - d^j_{m'm} = d^j_{-m -m'}
    - d^j_{m'm}(pi-beta) = (-1)^{j+m'}d^j_{m' -m}(beta)
    - d^j_{m'm}(pi+beta) = (-1)^{j-m}d^j_{m' -m}(beta)
    - d^j_{m'm}(-beta) = d^j_{m m'}(beta)
    - d^j_{m'm}(-beta) = (-1)^{m'-m} d^j_{m' m}(beta)
    """
    angs = np.array([-6, -5, -4, -3, -2, -1.5, -1, 1, 1.5, 2, 3, 4, 5, 6])
    for fac in angs:
        beta = np.pi/fac
        lmax = 10
        dmpms = rot.dlmn(lmax, beta)
        dmpms2 = rot.dlmn(lmax, np.pi-beta)
        dmpms3 = rot.dlmn(lmax, np.pi+beta)
        dmpms4 = rot.dlmn(lmax, -beta)
        for k in range(lmax):
            # d^j_{m'm} = -1^{m-m'} d^j_{m m'}
            nm = 2*k+1
            ms = np.arange(-k, k+1)
            fack = np.outer((-1)**np.abs(ms), (-1)**np.abs(-ms))
            predk = fack*np.transpose(dmpms[k])
            assert (np.abs(dmpms[k] - predk) < 100*np.finfo(float).eps).all()
            # d^j_{m'm} = d^j_{-m -m'}
            d_m_mp = np.flipud(np.fliplr(np.transpose(dmpms[k])))
            assert (np.abs(dmpms[k] - d_m_mp) < 100*np.finfo(float).eps).all()
            # d^j_{m'm}(pi-beta) = (-1)^{j+m'}d^j_{m' -m}(beta)
            fack = np.outer((-1)**np.abs(k+ms), np.ones(nm))
            predk = fack*np.fliplr(dmpms[k])
            assert (np.abs(dmpms2[k]-predk) < 100*np.finfo(float).eps).all()
            # d^j_{m'm}(pi+beta) = (-1)^{j-m}d^j_{m' -m}(beta)
            fack = np.outer(np.ones(nm), (-1)**np.abs(k-ms))
            predk = fack*np.fliplr(dmpms[k])
            assert (np.abs(dmpms3[k]-predk) < 100*np.finfo(float).eps).all()
            # d^j_{m'm}(-beta) = d^j_{m m'}(beta)
            predk = np.transpose(dmpms[k])
            assert (np.abs(dmpms4[k]-predk) < 100*np.finfo(float).eps).all()
            # d^j_{m'm}(-beta) = (-1)^{m'-m} d^j_{m' m}(beta)
            predk = np.outer((-1)**np.abs(ms), (-1)**np.abs(-ms))*dmpms[k]
            assert (np.abs(dmpms4[k]-predk) < 100*np.finfo(float).eps).all()


def test_Dl_symm():
    """
    Wikipedia consistency checks on Dl(0, beta, 0) function:

    Checks
    ------
    - d^j_{m'm} = -1^{m-m'} d^j_{m m'}
    - d^j_{m'm} = d^j_{-m -m'}
    - d^j_{m'm}(pi-beta) = (-1)^{j+m'}d^j_{m' -m}(beta)
    - d^j_{m'm}(pi+beta) = (-1)^{j-m}d^j_{m' -m}(beta)
    - d^j_{m'm}(-beta) = d^j_{m m'}(beta)
    - d^j_{m'm}(-beta) = (-1)^{m'-m} d^j_{m' m}(beta)
    """
    angs = np.array([-6, -5, -4, -3, -2, -1.5, -1, 1, 1.5, 2, 3, 4, 5, 6])
    for fac in angs:
        beta = np.pi/fac
        lmax = 3
        for k in range(lmax):
            dmpms = rot.Dl(k, 0, beta, 0)
            dmpms2 = rot.Dl(k, 0, np.pi-beta, 0)
            dmpms3 = rot.Dl(k, 0, np.pi+beta, 0)
            dmpms4 = rot.Dl(k, 0, -beta, 0)
            # d^j_{m'm} = -1^{m-m'} d^j_{m m'}
            nm = 2*k+1
            ms = np.arange(-k, k+1)
            fack = np.outer((-1)**np.abs(ms), (-1)**np.abs(-ms))
            predk = fack*np.transpose(dmpms)
            assert (np.abs(dmpms - predk) < 100*np.finfo(float).eps).all()
            # d^j_{m'm} = d^j_{-m -m'}
            d_m_mp = np.flipud(np.fliplr(np.transpose(dmpms)))
            assert (np.abs(dmpms - d_m_mp) < 100*np.finfo(float).eps).all()
            # d^j_{m'm}(pi-beta) = (-1)^{j+m'}d^j_{m' -m}(beta)
            fack = np.outer((-1)**np.abs(k+ms), np.ones(nm))
            predk = fack*np.fliplr(dmpms)
            assert (np.abs(dmpms2-predk) < 100*np.finfo(float).eps).all()
            # d^j_{m'm}(pi+beta) = (-1)^{j-m}d^j_{m' -m}(beta)
            fack = np.outer(np.ones(nm), (-1)**np.abs(k-ms))
            predk = fack*np.fliplr(dmpms)
            assert (np.abs(dmpms3-predk) < 100*np.finfo(float).eps).all()
            # d^j_{m'm}(-beta) = d^j_{m m'}(beta)
            predk = np.transpose(dmpms)
            assert (np.abs(dmpms4-predk) < 100*np.finfo(float).eps).all()
            # d^j_{m'm}(-beta) = (-1)^{m'-m} d^j_{m' m}(beta)
            predk = np.outer((-1)**np.abs(ms), (-1)**np.abs(-ms))*dmpms
            assert (np.abs(dmpms4-predk) < 100*np.finfo(float).eps).all()
