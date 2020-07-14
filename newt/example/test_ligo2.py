# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 16:30:41 2020

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

densAl = 2700   # kg/m^3
mh = densAl*np.pi*hc*rc**2
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
# Create some rotating holes with negative mass
cylh = qlm.cylinder(LMax, -mh, hc, rc)
cylh3i = trs.translate_qlm(cylh, [r3, 0, 0], LMax)
cylh2i = trs.translate_qlm(cylh, [r2, 0, 0], LMax)
cylhtot = np.copy(cylh3i)
cylhtot += rot.rotate_qlm(cylh3i, np.pi/3, 0, 0)
cylhtot += rot.rotate_qlm(cylh3i, 2*np.pi/3, 0, 0)
cylhtot += rot.rotate_qlm(cylh3i, np.pi, 0, 0)
cylhtot += rot.rotate_qlm(cylh3i, 4*np.pi/3, 0, 0)
cylhtot += rot.rotate_qlm(cylh3i, 5*np.pi/3, 0, 0)
cylhtot += cylh2i
cylhtot += rot.rotate_qlm(cylh2i, np.pi/2, 0, 0)
cylhtot += rot.rotate_qlm(cylh2i, np.pi, 0, 0)
cylhtot += rot.rotate_qlm(cylh2i, 3*np.pi/2, 0, 0)
#cyltot += cylhtot

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
# seems like last rotation matrix may have error, compute extra to be safe
ds = rot.wignerDl(LMax+1, dphi, 0, 0)
for k in range(nAng):
    print('Angle = ', np.round((k+1)*dphi*180/np.pi, 2), ' degrees')
    cyltot = rot.rotate_qlm_Ds(cyltot, ds[:-1])
    forces[k] = mplb.multipole_force(LMax, cyltot, tm, 0, 0, 0)

# plot some forces against angle
fig, ax = plt.subplots(1, 1)
ax.scatter(np.arange(1, nAng+1)*dphi, np.real(forces[:, 0]))
