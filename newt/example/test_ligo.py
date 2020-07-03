# -*- coding: utf-8 -*-
"""
Created on Mon May 25 14:10:29 2020

@author: John Greendeer Lee
"""
import numpy as np
import matplotlib.pyplot as plt
import newt.multipoleLib as mplb
import newt.qlm as qlm
import newt.translations as trs
import newt.rotations as rot

rc = .03683/2
hc = .0508
mc = 1.0558
r3 = .10478
r2 = .06033
mtm = 39.658
htm = .1998
rtm = .34021/2
dx = -.72419
dy = -.92365
LMax = 10

# Create some rotating test-masses
cyl = qlm.cylinder(LMax, mc, hc, rc)
cyl3i = trs.translate_qlm(cyl, [r3, 0, 0], LMax)
cyl2i = trs.translate_qlm(cyl, [r2, 0, 0], LMax)
cyltot = np.copy(cyl3i)
cyltot += rot.rotate_qlm(cyl3i, 2*np.pi/3, 0, 0)
cyltot += rot.rotate_qlm(cyl3i, 4*np.pi/3, 0, 0)
cyltot += cyl2i
cyltot += rot.rotate_qlm(cyl2i, np.pi, 0, 0)

# create a test-mirror
tm = qlm.cylinder(LMax, mtm, htm, rtm)
tm = rot.rotate_qlm(tm, np.pi/2, np.pi/2, -np.pi/2)
tm = trs.translate_q2Q(tm, [dx, dy, 0], LMax)

# Figure out the force at different rotor angles
nAng = 120
forces = np.zeros([nAng, 3], dtype='complex')
nlm, nc, ns = mplb.torque_lm(LMax, cyltot, tm)
dphi = 2*np.pi/nAng
# Now rotate the cylindrical test-masses through various angles and calculate
# the forces
for k in range(nAng):
    cylk = rot.rotate_qlm(cyltot, k*dphi, 0, 0)
    forces[k] = mplb.multipole_force(LMax, cylk, tm, 0, 0, 0)

# plot some forces against angle
fig, ax = plt.subplots(1, 1)
ax.scatter(np.arange(nAng)*dphi, np.real(forces[:, 0]))
