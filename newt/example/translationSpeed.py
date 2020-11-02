# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 11:28:19 2020

@author: John Greendeer Lee
"""
import numpy as np
import time
import newt.pg2Multi as pgm
import newt.glib as glb
import newt.translations as trs
import newt.rotations as rot
import newt.translationRecurs as trr

"""
Script for comparison of translation methods. Fast rotation + z-translation
vs regular translation.
"""

lmax = 20
# Create 2 points with mass1, at x=+/-1
d = 1
m = 1
m1 = np.array([[m, d, 0, 0], [m, -d, 0, 0]])
# Find inner moments around origin
qm1 = pgm.qmoments(lmax, m1)
# Find inner moments if translated by [.1, 0, 0]
rvec = [0.1, 0, 0]
qm1p = pgm.qmoments(lmax, glb.translate_point_array(m1, rvec))

# Find moments translated by [.1, 0, 0] using d'Urso, Adelberger
tic1 = time.perf_counter()
qm1p2 = trs.translate_qlm(qm1, rvec)
toc1 = time.perf_counter()

# Time the recursive method combined with recursive translation
tic2 = time.perf_counter()
r = np.sqrt(rvec[0]**2 + rvec[1]**2 + rvec[2]**2)
if r == 0 or r == rvec[2]:
    phi, theta = 0, 0
else:
    phi = np.arctan2(rvec[1], rvec[0])
    theta = np.arccos(rvec[2]/r)
rrms = trr.transl_newt_z_RR(lmax, r)
qm1r = rot.rotate_qlm(qm1, 0, -theta, -phi)
qlmp = trr.apply_trans_mat(qm1r, rrms)
qm1r2 = rot.rotate_qlm(qlmp, phi, theta, 0)
toc2 = time.perf_counter()

# Check that the translations match the predicted
assert (np.abs(qm1p - qm1r2) < 300*np.finfo(float).eps).all()
assert (np.abs(qm1p2 - qm1r2) < 300*np.finfo(float).eps).all()

# Compare the time for calculation at l=20
print("recursive (l<=20)[s]\t|\texplicit (l=20)[s]")
print(toc2-tic2, "\t|\t", toc1-tic1)
