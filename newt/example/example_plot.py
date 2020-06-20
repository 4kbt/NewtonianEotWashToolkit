# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 11:09:40 2020

@author: John Greendeer Lee
"""
import numpy as np
import newt.glib as glb
import newt.glibShapes as gshp

# Create a cylinder
cyl = gshp.annulus(1, 0, 1, 1, 10, 10)
# Inner cylinders on radius of 1m
cyl1 = glb.translate_point_array(cyl, [5, 0, 0])
# Outer cylinders on radius of 5m
cyl2 = glb.translate_point_array(cyl, [20, 0, 0])
# Combination of three inner cylinders
m1 = np.concatenate([cyl1, glb.rotate_point_array(cyl1, 2*np.pi/3, [0, 0, 1]),
                     glb.rotate_point_array(cyl1, -2*np.pi/3, [0, 0, 1])])
# Combination of three outer cylinders
m2 = np.concatenate([cyl2, glb.rotate_point_array(cyl2, 2*np.pi/3, [0, 0, 1]),
                     glb.rotate_point_array(cyl2, -2*np.pi/3, [0, 0, 1])])
fig, ax = glb.display_points(m1, m2)
ax.set_zlim([-20, 20])
