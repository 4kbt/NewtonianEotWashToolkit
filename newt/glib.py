# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 10:04:34 2015

@author: Charlie Hagedorn
Scribe: John Greendeer Lee
"""
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

BIG_G = 6.67428e-11


def gmmr2_array(mass1, mass2):
    """
    Computes the gravitational force of all mass2 points on mass1

    Inputs
    ------
    mass1 : ndarray
        numpy array row, 1x4, of form [mass, x, y, z]
    mass2 : ndarray
        numpy array, Nx4

    Returns
    -------
    force : ndarray
        numpy array, Nx4, of force on mass1 by all mass2 elements
    """
    if np.ndim(mass2) == 1:
        # Which way does the force act
        rvec = mass2[1:]-mass1[1:]
        # Pythagoras for modulus
        r = np.sqrt(np.sum(rvec**2))
        # Compute force
        force = rvec.T.dot(BIG_G*mass1[0]*mass2[0]/r**3)
    else:
        # Which way does the force act
        rvec = mass2[:, 1:]-mass1[1:]
        # Pythagoras for modulus
        r = np.sqrt(np.sum(rvec**2, 1))
        # compute force
        force = rvec.T.dot(BIG_G*mass1[0]*mass2[:, 0]/r**3)

    return force


def yukawa_array(mass1, mass2, alpha, lmbd):
    """
    Computes the yukawa force of all mass2 points on mass1

    Inputs
    ------
    mass1 : ndarray
        numpy array row, 1x4, of form [mass, x, y, z]
    mass2 : ndarray
        numpy array, Nx4
    alpha : float
        coupling strength of Yukawa interaction scaled by BIG_G
    lmbd : float
        Yukawa interaction length scale

    Returns
    -------
    force : ndarray
        numpy array, Nx4, of force on mass1 by all mass2 elements
    """
    coef = alpha*BIG_G*mass1[0]
    if np.ndim(mass2) == 1:
        # Which way does the force act
        rvec = mass2[1:]-mass1[1:]
        # Pythagoras for modulus
        r = np.sqrt(np.sum(rvec**2))
        # Compute force
        fac = (1/r**3 + 1/(r**2*lmbd))
        force = rvec.T.dot(coef*mass2[0]*np.exp(-r/lmbd)*fac)
    else:
        # Which way does the force act
        rvec = mass2[:, 1:]-mass1[1:]
        # Pythagoras for modulus
        r = np.sqrt(np.sum(rvec**2, 1))
        # compute force
        fac = (1/r**3 + 1/(r**2*lmbd))
        force = rvec.T.dot(coef*mass2[:, 0]*np.exp(-r/lmbd)*fac)

    return force


def point_matrix_gravity(mass1, mass2):
    """
    Computes the force and 3-axis torque about the origin on array1 by array2
    from a gravitational potential.

    Inputs
    ------
    mass1 : ndarray
        Mx4, of form [mass_i, x_i, y_i, z_i]
    mass2 : ndarray
        Nx4, of form [mass_i, x_i, y_i, z_i]

    Returns
    -------
    force : ndarray
        1x3 numpy array [f_x, f_y, f_z]
    torque : ndarray
        1x3 numpy array, [T_x, T_y, T_z]
    """
    force = np.zeros(3)
    torque = np.zeros(3)

    if np.ndim(mass1) == 1:
        force = gmmr2_array(mass1, mass2)
        torque = np.cross(mass1[1:], force)
    else:
        for k in range(len(mass1)):
            forceK = gmmr2_array(mass1[k, :], mass2)
            torqueK = np.cross(mass1[k, 1:], forceK)

            force += forceK
            torque += torqueK

    return force, torque


def point_matrix_yukawa(mass1, mass2, alpha, lmbd):
    """
    Computes the force and 3-axis torque about the origin on array1 by array2
    from a Yukawa potential with length scale lmbd and gravitational strength.

    Inputs
    ------
    mass1 : ndarray
        Mx4, of form [mass_i, x_i, y_i, z_i]
    mass2 : ndarray
        Nx4, of form [mass_i, x_i, y_i, z_i]
    alpha : float
        coupling strength of Yukawa interaction scaled by BIG_G
    lmbd : float
        Yukawa interaction length scale

    Returns
    -------
    force : ndarray
        1x3 numpy array [f_x, f_y, f_z]
    torque : ndarray
        1x3 numpy array, [T_x, T_y, T_z]
    """
    force = np.zeros(3)
    torque = np.zeros(3)

    if np.ndim(mass1) == 1:
        force = yukawa_array(mass1, mass2, alpha, lmbd)
        torque = np.cross(mass1[1:], force)
    else:
        for k in range(len(mass1)):
            forceK = yukawa_array(mass1[k, :], mass2, alpha, lmbd)
            torqueK = np.cross(mass1[k, 1:], forceK)

            force += forceK
            torque += torqueK

    return force, torque


def translate_point_array(pointMass, transVec):
    """
    Translates point mass by transVec (a three vector)

    Inputs
    ------
    pointMass : ndarray
        Mx4 array of form [mass_i, x_i, y_i, z_i]
    transVec : ndarray
        1x3 array

    Returns
    -------
    transArray : ndarray
        Mx4 translated array
    """
    if np.ndim(pointMass) == 1:
        transArray = np.zeros(4)
        transArray[0] = pointMass[0]
        transArray[1:] = pointMass[1:]+transVec
    else:
        transArray = np.zeros([len(pointMass), 4])
        transArray[:, 0] = pointMass[:, 0]
        transArray[:, 1:] = pointMass[:, 1:]+transVec

    return transArray


def rotate_point_array(pointMass, theta, rotVec):
    """
    Rotates pointMass by angle (in radians) about vector from origin,
    using Rodrigues' Formula:
    http://mathworld.wolfram.com/RodriguesRotationFormula.html

    Inputs
    Returns
    """
    norm = np.sqrt(np.dot(rotVec, rotVec))
    unit = rotVec/norm

    W = np.array([[0, -unit[2], unit[1]],
                  [unit[2], 0, -unit[0]],
                  [-unit[1], unit[0], 0]])
    R = np.identity(3)+np.sin(theta)*W+2*(np.sin(theta/2.)**2)*W.dot(W)

    if np.ndim(pointMass) == 1:
        rotArray = np.zeros(4)
        rotArray[0] = pointMass[0]
        rotArray[1:] = R.dot(pointMass[1:])
    else:
        rotArray = np.zeros([len(pointMass), 4])
        rotArray[:, 0] = pointMass[:, 0]
        for k in range(len(pointMass)):
            rotArray[k, 1:] = R.dot(pointMass[k, 1:])

    return rotArray


def display_points(pm1, pm2, scale_mass=False):
    """
    Creates a 3-dimensional plot of the two point-mass arrays pm1 and pm2.

    Inputs
    ------
    pm1 : ndarray
        N1x4 array of first set of point masses [m, x, y, z]
    pm2 : ndarray
        N2x4 array of second set of point masses [m, x, y, z]
    scale_mass : bool
        Determines whether points are scaled based on their masses. Useful for legendre-gauss spacing.

    Returns
    -------
    fig : matplotlib.pyplot.figure object
        Figure object
    ax : matplotlib.pyplot.axes object
        Axes object
    """
    fig = plt.figure()
    ax = Axes3D(fig)
    s1 = 50
    s2 = 50
    if scale_mass:
        pts1 = len(pm1)
        pts2 = len(pm2)
        s1 = 100*pts1*pm1[:,0]
        s2 = 100*pts1*pm2[:,0]
    ax.scatter(pm1[:, 1], pm1[:, 2], pm1[:, 3], label='mass1', s=s1, alpha=.5)
    ax.scatter(pm2[:, 1], pm2[:, 2], pm2[:, 3], label='mass2', s=s2,
               marker='s', alpha=.5)
    ax.legend()
    return fig, ax
