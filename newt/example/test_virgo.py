# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 17:59:22 2020

@author: John Greendeer Lee
"""
import numpy as np
import matplotlib.pyplot as plt
import newt.multipoleLib as mplb
import newt.qlm as qlm
import newt.translations as trs
import newt.rotations as rot

irc = .055
orc = .190
hc = .052
beta = np.pi/8
dens = 2800  # kg/m^3, 7075 T6 aluminum
mc = dens*2*beta*(orc**2-irc**2)*hc
r3 = .10478
r2 = .06033
mtm = 39.658
htm = .20    # 20cm thickness
rtm = .35/2  # 35cm diam
d = 1.32    # m
phi = 0.241 # radian
dx = d*np.cos(phi)
dy = d*np.sin(phi)
LMax = 10

# Create some rotating test-masses
arc = qlm.annulus(LMax, mc/2, hc, irc, orc, 0, beta)
arc2 = rot.rotate_qlm(arc, np.pi, 0, 0)
arctot = arc + arc2

# create a test-mirror
tm = qlm.cylinder(LMax, mtm, htm, rtm)
tm = rot.rotate_qlm(tm, np.pi/2, np.pi/2, -np.pi/2)
tm = trs.translate_q2Q(tm, [dx, dy, 0], LMax)

# Figure out the force at different rotor angles
nAng = 120
forces = np.zeros([nAng, 3], dtype='complex')
nlm, nc, ns = mplb.torque_lm(LMax, arctot, tm)
dphi = 2*np.pi/nAng
# Now rotate the cylindrical test-masses through various angles and calculate
# the forces
for k in range(nAng):
    cylk = rot.rotate_qlm(arctot, k*dphi, 0, 0)
    forces[k] = mplb.multipole_force(LMax, cylk, tm, 0, 0, 0)

# plot some forces against angle
fig, ax = plt.subplots(1, 1)
ax.scatter(np.arange(nAng)*dphi, np.real(forces[:, 0]))
