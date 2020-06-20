# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 10:30:09 2020

@author: John
"""
import numpy as np
import matplotlib.pyplot as plt
import newt.rotations as rot
import time

# Example script
# Recursive rotation matrix calculation
beta = np.pi/4
tic = time.perf_counter()
Hs = rot.rotate_H_recurs(100, beta)
epsmmp = rot.epsm(-np.arange(-40, 41))  # sign factor #1
epsmm = rot.epsm(np.arange(-40, 41))    # sign factor #2
D40r = np.outer(epsmm, epsmmp)*Hs[40]
toc = time.perf_counter()

# Classic calcultion
tic2 = time.perf_counter()
H40 = rot.Dl(40, 0, beta, 0)  # transposed relative to Hs
toc2 = time.perf_counter()

# Show that these two things match (visually)
fig, ax = plt.subplots(1, 2, sharey=True)
ax[0].imshow(D40r, extent=[-40, 40, -40, 40])
ax[1].imshow(np.real(H40.T), extent=[-40, 40, -40, 40])
ax[0].set_title('Recursive')
ax[1].set_title('Explicit')
ax[0].set_ylabel("m'")
ax[0].set_xlabel('m')
ax[1].set_xlabel('m')
fig.suptitle(r'$L=40$ small-d Wigner matrix for $\beta=\pi/4$')

# Compare the calculation times
print("recursive (l<=100)\t|\texplicit (l=40)")
print(toc-tic, "\t|\t", toc2-tic2)
