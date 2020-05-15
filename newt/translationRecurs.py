# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 08:16:19 2020

@author: John Greendeer Lee
"""
import numpy as np
import scipy.special as sp


def transl_yuk_z_SR_recurs(L, k, dr):
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
    ls = np.arange(L)
    sr = np.zeros([L+1, L+1], dtype=complex)
    sr[:, 0] = (-1)**(ls)*np.sqrt(2*ls+1)*sp.spherical_kn(ls, k*dr)
    for l in np.arange(L):
        for m in range(l):
            blmm = bnm(l, m-1)
            blmp = bnm(l+1, m)
            bmm = bnm(m+1, -m-1)
            print(blmm, blmp, bmm)
            sr[l, m+1] = (sr[l-1, m]*blmm - sr[l+1, m]*blmp)/bmm
    return sr


def transl_yuk_z_RR_recurs(L, k, dr):
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
    ls = np.arange(L+1)
    rr = np.zeros([L+1, L+1], dtype=complex)
    rr[:, 0] = (-1)**(ls)*np.sqrt(2*ls+1)*sp.spherical_in(ls, k*dr)
    for l in np.arange(L):
        for m in range(l):
            blmm = bnm(l, m-1)
            blmp = bnm(l+1, m)
            bmm = bnm(m+1, -m-1)
            print(blmm, blmp, bmm)
            rr[l, m+1] = (rr[l-1, m]*blmm - rr[l+1, m]*blmp)/bmm
    return rr


def transl_newt_z_SR_recurs(L, dr):
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
    ls = np.arange(L)
    sr = np.zeros([L+1, L+1], dtype=complex)
    sr[:, 0] = (-1)**(ls)*np.sqrt(2*ls+1)/dr**(ls+1)
    for l in np.arange(L):
        for m in range(l):
            blmm = bnm(l, m-1)
            blmp = bnm(l+1, m)
            bmm = bnm(m+1, -m-1)
            print(blmm, blmp, bmm)
            sr[l, m+1] = (sr[l-1, m]*blmm - sr[l+1, m]*blmp)/bmm
    return sr


def transl_newt_z_RR_recurs(L, k, dr):
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
    ls = np.arange(L+1)
    rr = np.zeros([L+1, L+1], dtype=complex)
    rr[:, 0] = np.sqrt(2*ls+1)*(-dr)**(ls)
    for l in np.arange(L):
        for m in range(l):
            blmm = bnm(l, m-1)
            blmp = bnm(l+1, m)
            bmm = bnm(m+1, -m-1)
            print(blmm, blmp, bmm)
            rr[l, m+1] = (rr[l-1, m]*blmm - rr[l+1, m]*blmp)/bmm
    return rr


def bnm(n, m):
    bval = np.sqrt((n-m-1)*(n-m)/((2*n-1)*(2*n+1)))*np.sign(n)
    if np.abs(m) > n:
        bval = 0
    return bval
