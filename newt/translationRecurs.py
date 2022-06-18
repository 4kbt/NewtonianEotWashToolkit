# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 08:16:19 2020

@author: John Greendeer Lee
"""
import numpy as np
import scipy.special as sp
import scipy.linalg as sla
import newt.rotations as rot


def transl_yuk_z_SR_recurs(LMax, dr, k):
    """
    Translate coaxially from regular to singular, Gumerov and Duraiswami.

    Inputs
    ------
    l : int
        Order of multipole expansion to output rotation matrix coefficient H

    Reference
    ---------
    "Recursions for the computation of multipole translation and rotation
    coefficients for the 3-d helmholtz equation."

    https://arxiv.org/pdf/1403.7698v1.pdf
    """
    # Create (E|F)_{l,m}^{m}
    eflm = np.zeros([2*LMax+1, LMax+1])
    # Start (E|F)_{l, 0}^{0}
    ls = np.arange(2*LMax+1)
    eflm[:, 0] = (-1)**ls*np.sqrt(2*ls+1)*sp.spherical_kn(ls, k*dr)
    # now recursion 4.86 for (E|F)_{l, m}^{m}
    for m in range(LMax):
        for l in range(2*LMax-m):
            # (n=m)
            blmm = get_bnm(l, -m-1)
            blmp = get_bnm(l+1, m)
            bmm = get_bnm(m+1, -m-1)
            eflm[l, m+1] = (eflm[l-1, m]*blmm - eflm[l+1, m]*blmp)/bmm

    efms = []
    # Now create (E|F)^m matrices (p-|m|)x(p-|m|)
    # Start with (2*p-|m|)x(p-|m|) and truncate
    for m in range(LMax+1):
        efm = np.zeros([2*LMax+1-2*m, LMax-m+1])
        efm[:, 0] = eflm[m:2*LMax+1-m, m]
        for n in range(LMax-m):
            for l in range(n+1, 2*(LMax-m)-n):
                anm = get_anm(n+m, m)
                alm = get_anm(l+m, m)
                almm = get_anm(l-1+m, m)
                anmm = get_anm(n-1+m, m)
                print(l, n+1, l+m, anm, anmm, alm, almm)
                if anmm != 0:
                    print(l, n, m)
                    efm[l, n+1] = (almm*efm[l-1, n] - alm*efm[l+1, n]
                                   + anmm*efm[l, n-1])/anm
                else:
                    efm[l, n+1] = (alm*efm[l+1, n] - almm*efm[l-1, n])/anm
        efms.append(efm)

    for m in range(LMax+1):
        efms[m] = efms[m][:LMax-m+1, :LMax-m+1]
        lm = len(efms[m])
        # Start from m since (E|F)^m matrix starts at (m,m) corner
        nls = (np.arange(m, m+lm))
        enl = np.transpose(efms[m])*np.outer((-1)**(nls), (-1)**(nls))
        efms[m] += enl
        efms[m] -= np.diag(np.diag(enl))

    return eflm, efms


def transl_yuk_z_RR_recurs(LMax, dr, k):
    """
    Translate coaxially from regular to regular, Gumerov and Duraiswami.

    Inputs
    ------
    l : int
        Order of multipole expansion to output rotation matrix coefficient H

    Reference
    ---------
    "Recursions for the computation of multipole translation and rotation
    coefficients for the 3-d helmholtz equation."

    https://arxiv.org/pdf/1403.7698v1.pdf
    """
    # Create (E|F)_{l,m}^{m}
    eflm = np.zeros([2*LMax+1, LMax+1])
    # Start (E|F)_{l, 0}^{0}
    ls = np.arange(2*LMax+1)
    eflm[:, 0] = (-1)**ls*np.sqrt(2*ls+1)*sp.spherical_in(ls, k*dr)
    # now recursion 4.86 for (E|F)_{l, m}^{m}
    for m in range(LMax):
        for l in range(2*LMax-m):
            # (n=m)
            blmm = get_bnm(l, -m-1)
            blmp = get_bnm(l+1, m)
            bmm = get_bnm(m+1, -m-1)
            eflm[l, m+1] = (eflm[l-1, m]*blmm - eflm[l+1, m]*blmp)/bmm

    efms = []
    # Now create (E|F)^m matrices (p-|m|)x(p-|m|)
    # Start with (2*p-|m|)x(p-|m|) and truncate
    for m in range(LMax+1):
        efm = np.zeros([2*LMax+1-2*m, LMax-m+1])
        efm[:, 0] = eflm[m:2*LMax+1-m, m]
        for n in range(LMax-m):
            for l in range(n+1, 2*(LMax-m)-n):
                anm = get_anm(n+m, m)
                alm = get_anm(l+m, m)
                almm = get_anm(l-1+m, m)
                anmm = get_anm(n-1+m, m)
                print(l, n+1, l+m, anm, anmm, alm, almm)
                if anmm != 0:
                    print(l, n, m)
                    efm[l, n+1] = (almm*efm[l-1, n] - alm*efm[l+1, n]
                                   + anmm*efm[l, n-1])/anm
                else:
                    efm[l, n+1] = (alm*efm[l+1, n] - almm*efm[l-1, n])/anm
        efms.append(efm)

    for m in range(LMax+1):
        efms[m] = efms[m][:LMax-m+1, :LMax-m+1]
        lm = len(efms[m])
        # Start from m since (E|F)^m matrix starts at (m,m) corner
        nls = (np.arange(m, m+lm))
        enl = np.transpose(efms[m])*np.outer((-1)**(nls), (-1)**(nls))
        efms[m] += enl
        efms[m] -= np.diag(np.diag(enl))

    return eflm, efms


def transl_newt_z_RR(LMax, dr):
    """
    Translate coaxially from regular to regular, Gumerov and Duraiswami. In a
    slight contradiction to the paper, our choice of normalization (alpha_n^m)
    does not require all three normalization factors in the translation matrix,
    see eqn 19.

    Inputs
    ------
    l : int
        Order of multipole expansion to output rotation matrix coefficient H

    Reference
    ---------
    "Comparison of the efficiency of translation operators used in the fast
    multipole method for the 3D Laplace equation"

    http://legacydirs.umiacs.umd.edu/~gumerov/PDFs/cs-tr-4701.pdf
    """
    # Descending array of lp from LMax through 0
    lp = np.arange(LMax+1)
    cols = (-np.sign(dr))**lp*np.exp(lp*np.log(np.abs(dr)) - sp.gammaln(lp+1))
    rows = [cols[0], *np.zeros(LMax)]
    rrms = []
    for m in range(LMax+1):
        fac = alphanm(lp[m:], m)
        fac2 = np.outer(1/fac, fac)
        rrm = sla.toeplitz(cols[:LMax+1-m], rows[:LMax+1-m])*fac2
        rrms.append(rrm)
    return rrms


def transl_newt_z_SS(LMax, dr):
    """
    Translate coaxially from singular to singular, Gumerov and Duraiswami. In a
    slight contradiction to the paper, our choice of normalization (alpha_n^m)
    does not require all three normalization factors in the translation matrix,
    see eqn 19.

    Inputs
    ------
    l : int
        Order of multipole expansion to output rotation matrix coefficient H

    Reference
    ---------
    "Comparison of the efficiency of translation operators used in the fast
    multipole method for the 3D Laplace equation"

    http://legacydirs.umiacs.umd.edu/~gumerov/PDFs/cs-tr-4701.pdf
    """
    lp = np.arange(LMax+1)
    rows = (-np.sign(dr))**lp*np.exp(lp*np.log(np.abs(dr)) - sp.gammaln(lp+1))
    cols = [rows[0], *np.zeros(LMax)]
    ssms = []
    for m in range(LMax+1):
        fac = betanm(lp[m:], m)
        fac2 = np.outer(1/fac, fac)
        ssm = sla.toeplitz(cols[:LMax+1-m], rows[:LMax+1-m])*fac2
        ssms.append(ssm)
    return ssms


def transl_newt_z_SR(LMax, dr):
    """
    Translate coaxially from regular to singular, Gumerov and Duraiswami. In a
    slight contradiction to the paper, our choice of normalization (alpha_n^m)
    does not require all three normalization factors in the translation matrix,
    see eqn 19.

    Inputs
    ------
    l : int
        Order of multipole expansion to output rotation matrix coefficient H

    Reference
    ---------
    "Comparison of the efficiency of translation operators used in the fast
    multipole method for the 3D Laplace equation"

    http://legacydirs.umiacs.umd.edu/~gumerov/PDFs/cs-tr-4701.pdf
    """
    lp = np.arange(2*LMax+1)
    svals = np.exp(sp.gammaln(lp+1) - (lp+1)*np.log(dr))
    srms = []
    for m in range(LMax+1):
        faca = alphanm(lp[m:LMax+1], m)
        facb = betanm(lp[m:LMax+1], m)
        fac2 = np.outer(1/facb, faca)
        srm = sla.hankel(svals[2*m:LMax+m+1], svals[LMax+m:])*fac2
        srms.append(srm)
    return srms


def alphanm(n, m):
    """
    Normalization function for inner moments in Gumerov & Duraiswami formalism.
    """
    anm = (-1)**n*1j**(-np.abs(m))
    anm *= np.sqrt(4*np.pi/((2*n+1)*sp.factorial(n-m)*sp.factorial(n+m)))
    return anm


def betanm(n, m):
    """
    Normalization function for outer moments in Gumerov & Duraiswami formalism.
    """
    bnm = 1j**(np.abs(m))
    bnm *= np.sqrt(4*np.pi*sp.factorial(n-m)*sp.factorial(n+m)/(2*n+1))
    return bnm


def apply_trans_mat(qlm, efms):
    """
    Applies each set of the coaxial translation matrices which mix degree (l)
    for fixed order (m).

    Inputs
    ------
    qlm : ndarray
        Complex multipole coefficients of shape (L+1)x(2L+1)
    efms : list
        List of (lxl) coaxial translation matrices, for each degree, (L, L-1,
        ..., 1)

    Returns
    -------
    qNew : ndarray
        Coaxially translated complex multipole coeffcients of shape
        (L+1)x(2L+1)
    """
    L = len(qlm)-1
    if L != (len(efms)-1):
        raise ValueError('Mis-matched multipole degree, l')
    qNew = np.zeros([L+1, 2*L+1], dtype='complex')
    qNew[:, L] = np.dot(efms[0], qlm[:, L])
    for m in range(1, L+1):
        qNew[m:, L+m] = np.dot(efms[m], qlm[m:, L+m])
        qNew[m:, L-m] = np.dot(efms[m], qlm[m:, L-m])
    return qNew


def get_bnm(n, m):
    """
    B-recursion function for translation matrices.
    """
    if np.abs(m) > n:
        bval = 0
    else:
        bval = np.sqrt((n-m-1)*(n-m)/((2*n-1)*(2*n+1)))*np.sign(n)
    return bval


def get_anm(n, m):
    """
    A-recursion function for translation matrices.
    """
    am = abs(m)
    if am > n:
        aval = 0
    else:
        aval = np.sqrt((n+am+1)*(n-am+1)/((2*n+1)*(2*n+3)))
    return aval


def translate_qlm(qlm, rvec):
    lmax = len(qlm) - 1
    r = np.sqrt(rvec[0]**2 + rvec[1]**2 + rvec[2]**2)
    rrms = transl_newt_z_RR(lmax, r)
    if r == 0:
        qlmp = np.copy(qlm)
    elif r == rvec[2]:
        phi, theta = 0, 0
        qlmp = apply_trans_mat(qlm, rrms)
    else:
        phi = np.arctan2(rvec[1], rvec[0])
        theta = np.arccos(rvec[2]/r)
        Ds = rot.wignerDl(lmax, -phi, -theta, -phi)
        qm1r = rot.rotate_qlm_Ds(qlm, Ds)
        qlmp = apply_trans_mat(qm1r, rrms)
        qlmp = rot.rotate_qlm_Ds(qlmp, Ds, True)
    return qlmp


def translate_Qlmb(Qlm, rvec):
    lmax = len(Qlm) - 1
    r = np.sqrt(rvec[0]**2 + rvec[1]**2 + rvec[2]**2)
    ssms = transl_newt_z_SS(lmax, r)
    if r == 0:
        Qlmp = np.copy(Qlm)
    elif r == rvec[2]:
        phi, theta = 0, 0
        Qlmp = apply_trans_mat(Qlm, ssms)
    else:
        phi = np.arctan2(rvec[1], rvec[0])
        theta = np.arccos(rvec[2]/r)
        Ds = rot.wignerDl(lmax, -phi, -theta, -phi)
        qm1r = rot.rotate_qlm_Ds(Qlm, Ds)
        Qlmp = apply_trans_mat(qm1r, ssms)
        Qlmp = rot.rotate_qlm_Ds(Qlmp, Ds, True)
    return Qlmp


def translate_q2Q(qlm, rvec):
    lmax = len(qlm) - 1
    r = np.sqrt(rvec[0]**2 + rvec[1]**2 + rvec[2]**2)
    srms = transl_newt_z_SR(lmax, r)
    if r == 0:
        Qlmp = np.copy(qlm)
    elif r == rvec[2]:
        phi, theta = 0, 0
        Qlmp = apply_trans_mat(qlm, srms)
    else:
        phi = np.arctan2(rvec[1], rvec[0])
        theta = np.arccos(rvec[2]/r)
        Ds = rot.wignerDl(lmax, -phi, -theta, -phi)
        qm1r = rot.rotate_qlm_Ds(qlm, Ds)
        Qlmp = apply_trans_mat(qm1r, srms)
        Qlmp = rot.rotate_qlm_Ds(Qlmp, Ds, True)
    return Qlmp
