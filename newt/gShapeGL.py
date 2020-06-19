# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 14:55:03 2020

@author: John
"""
import numpy as np


def rectangle(mass, x, y, z, dx, dy, dz):
    """
    Creates point masses distributed in an rectangular solid of mass m.

    Inputs
    ------
    mass : float
        mass in kg
    x : float
        x-length of brick in m
    y : float
        y-length of brick in m
    z : float
        z-length of brick in m
    dx : float
        number of Gauss-Legendre points distributed in x
    dy : float
        number of Gauss-Legendre points distributed in y
    dz : float
        number of Gauss-Legendre points distributed in z

    Returns
    -------
    pointArray : ndarray
        nx*ny*nz x 4 point mass array of format [m, x, y, z]
    """
    pointArray = np.zeros([dx*dy*dz, 4])
    # In some sense we are integrating uniform density over rectangular prism.
    # Sum over all weights times constant dens should equal integral which
    # is the mass. Sum of weights is 8 (2 in each direction)
    pointArray[:, 0] = mass/8
    sx, wx = np.polynomial.legendre.leggauss(dx)
    sy, wy = np.polynomial.legendre.leggauss(dy)
    sz, wz = np.polynomial.legendre.leggauss(dz)
    for k in range(dz):
        for l in range(dy):
            for m in range(dx):
                pointArray[k*dy*dx+l*dx+m, 1] = sx[m]*x/2
                pointArray[k*dy*dx+l*dx+m, 2] = sy[l]*y/2
                pointArray[k*dy*dx+l*dx+m, 3] = sz[k]*z/2
                pointArray[k*dy*dx+l*dx+m, 0] *= wx[m]*wy[l]*wz[k]
    return pointArray


def sphere(mass, r, dd):
    """
    Creates point masses distributed in an spherical solid of mass m.

    A single point-mass will suffice for the gravitational effect of any
    spherical shell or sphere. This function is generally not needed for
    calculations of Newtonian gravity. Use only when required.

    Inputs
    ------
    mass : float
        mass in kg
    x : float
        x-length of brick in m
    y : float
        y-length of brick in m
    z : float
        z-length of brick in m
    dd : float
        order of Gauss-Legendre polynomials distributed in box of side length 2*r

    Returns
    -------
    pointArray : ndarray
        nx*ny*nz x 4 point mass array of format [m, x, y, z]
    """
    pointArray = np.zeros([dd**3, 4])
    # Factor of 8 since sum of weights is 2 in each direction
    sx, wx = np.polynomial.legendre.leggauss(dd)
    sy, wy = np.polynomial.legendre.leggauss(dd)
    sz, wz = np.polynomial.legendre.leggauss(dd)

    #Determining mass-normalization
    density = mass / ( 4.0/3.0 * np.pi * r**3 )
    massnorm = density * r**3 #(this is (2*r)**3/2**3 )

    for k in range(dd):
        for l in range(dd):
            for m in range(dd):
                pointArray[k*dd*dd+l*dd+m, 1] = sx[m]*r
                pointArray[k*dd*dd+l*dd+m, 2] = sy[l]*r
                pointArray[k*dd*dd+l*dd+m, 3] = sz[k]*r
                pointArray[k*dd*dd+l*dd+m, 0] = wx[m]*wy[l]*wz[k] * massnorm

    pointArray = np.array([pointArray[k] for k in range(dd**3) if
                           np.sum(pointArray[k, 1:]**2) <= r**2])
    # Total mass should check out as if integrating over points
    pointArray[:, 0] *= mass/np.sum(pointArray[:, 0])
    return pointArray


def annulus(mass, iR, oR, t, dd, dz):
    """
    Creates point masses distributed in an annulus of mass m.

    Inputs
    ------
    m : float
        mass in kg
    iR : float
        inner radius of annulus in m
    oR : float
        outer radius of annulus in m
    t : float
        thickness of annulus in m
    dd : float
        number of Gauss-Legendre points distributed in x,y
    dz : float
        number of Gauss-Legendre points distributed in z

    Returns
    -------
    pointArray : ndarray
        point mass array of format [m, x, y, z]
    """
    pointArray = np.zeros([dz*dd**2, 4])
    sx, wx = np.polynomial.legendre.leggauss(dd)
    sy, wy = np.polynomial.legendre.leggauss(dd)
    sz, wz = np.polynomial.legendre.leggauss(dd)
    for k in range(dz):
        for l in range(dd):
            for m in range(dd):
                pointArray[k*dd*dd+l*dd+m, 1] = sx[m]*oR
                pointArray[k*dd*dd+l*dd+m, 2] = sy[l]*oR
                pointArray[k*dd*dd+l*dd+m, 3] = sz[k]*t
                pointArray[k*dd*dd+l*dd+m, 0] = wx[m]*wy[l]*wz[k]

    pointArray = np.array([pointArray[k] for k in range(dz*dd**2) if
                           pointArray[k, 1]**2+pointArray[k, 2]**2 >= iR**2 and
                           pointArray[k, 1]**2+pointArray[k, 2]**2 <= oR**2])

    # Correct the masses of the points
    pointArray[:, 0] *= mass/np.sum(pointArray[:, 0])

    return pointArray


def cone(mass, R, H, dd, dz):
    """
    Creates point masses distributed in a cone of mass m.

    Inputs
    ------
    m : float
        mass in kg
    R : float
        outer radius of cone in m
    H : float
        height of cone in m
    dd : float
        number of Gauss-Legendre points distributed in x,y
    dz : float
        number of Gauss-Legendre points distributed in z

    Returns
    -------
    pointArray : ndarray
        point mass array of format [m, x, y, z]
    """
    pointArray = np.zeros([dz*dd**2, 4])
    tanTheta = H/R
    sx, wx = np.polynomial.legendre.leggauss(dd)
    sy, wy = np.polynomial.legendre.leggauss(dd)
    sz, wz = np.polynomial.legendre.leggauss(dd)
    for k in range(dz):
        for l in range(dd):
            for m in range(dd):
                pointArray[k*dd*dd+l*dd+m, 1] = sx[m]*R
                pointArray[k*dd*dd+l*dd+m, 2] = sy[l]*R
                pointArray[k*dd*dd+l*dd+m, 3] = sz[k]*H
                pointArray[k*dd*dd+l*dd+m, 0] = wx[m]*wy[l]*wz[k]

    pointArray = np.array([pointArray[k] for k in range(dz*dd**2) if
                           pointArray[k, 1]**2+pointArray[k, 2]**2 <=
                           (pointArray[k, 3]/tanTheta)**2])

    # Correct the masses of the points
    pointArray[:, 0] *= mass/np.sum(pointArray[:, 0])

    return pointArray


def wedge(mass, iR, oR, t, theta, dd, dz):
    """
    Creates point masses distributed in an annular section of mass m subtending
    an angle of 2*theta.

    Inputs
    ------
    m : float
        mass in kg
    iR : float
        inner radius of annulus in m
    oR : float
        outer radius of annulus in m
    t : float
        thickness of annulus in m
    dd : float
        number of Gauss-Legendre points distributed in x,y
    dz : float
        number of Gauss-Legendre points distributed in z

    Returns
    -------
    pointArray : ndarray
        point mass array of format [m, x, y, z]
    """
    pointArray = np.zeros([dz*dd**2, 4])
    xmax = oR
    if theta < np.pi/2:
        xmin = np.cos(theta)*iR
        ymax = np.sin(theta)*oR
    else:
        xmin = np.cos(theta)*oR
        ymax = oR
    dx, xa = (xmax-xmin)/2, (xmax+xmin)/2
    ctr = 0
    sx, wx = np.polynomial.legendre.leggauss(dd)
    sy, wy = np.polynomial.legendre.leggauss(dd)
    sz, wz = np.polynomial.legendre.leggauss(dd)
    for k in range(dz):
        for l in range(dd):
            for m in range(dd):
                x = sx[m]*dx + xa
                y = sy[l]*ymax
                z = sz[k]*t
                w = wx[m]*wy[l]*wz[k]
                r = np.sqrt(x**2 + y**2)
                q = np.arctan2(y, x)
                if np.abs(q) <= theta and r <= oR and r >= iR:
                    pointArray[ctr] = [w, x, y, z]
                    ctr += 1

    pointArray = pointArray[:ctr]

    # Correct the masses of the points
    pointArray[:, 0] *= mass/np.sum(pointArray[:, 0])

    return pointArray


def tri_prism(mass, d, y1, y2, t, dx, dy, dz):
    """
    Creates point masses distributed in a triangular prism of mass m.

    Inputs
    ------
    m : float
        mass in kg
    d : float
        Length of prism along x-axis
    y1 : float
        First vertex (d, y1) in y-axis of triangular-prism in m
    y2 : float
        Second vertex (d, y2) in y-axis of triangular-prism in m
    t : float
        Thickness of triangular-prism in m
    dx : float
        number of Gauss-Legendre points distributed in x
    dy : float
        number of Gauss-Legendre points distributed in y
    dz : float
        number of Gauss-Legendre points distributed in z

    Returns
    -------
    pointArray : ndarray
        point mass array of format [m, x, y, z]
    """
    if y2 < y1:
        print('Require y2 > y1')
        return []
    ymin = np.min([0, y1])
    ymax = np.max([0, y2])
    base, yave = (ymax-ymin)/2, (ymax+ymin)/2
    pointArray = np.zeros([dx*dy*dz, 4])
    sx, wx = np.polynomial.legendre.leggauss(dx)
    sy, wy = np.polynomial.legendre.leggauss(dy)
    sz, wz = np.polynomial.legendre.leggauss(dz)
    for k in range(dz):
        for l in range(dy):
            for m in range(dx):
                pointArray[k*dy*dx+l*dx+m, 1] = sx[m]*d/2 + d/2
                pointArray[k*dy*dx+l*dx+m, 2] = sy[l]*base/2 + yave
                pointArray[k*dy*dx+l*dx+m, 3] = sz[k]*t/2
                pointArray[k*dy*dx+l*dx+m, 0] = wx[m]*wy[l]*wz[k]

    pointArray = np.array([pointArray[k] for k in range(dx*dy*dz) if
                           pointArray[k, 1]*y1 <= pointArray[k, 2]*d and
                           pointArray[k, 1]*y2 >= pointArray[k, 2]*d])

    # Correct the masses of the points
    pointArray[:, 0] *= mass/np.sum(pointArray[:, 0])

    return pointArray
