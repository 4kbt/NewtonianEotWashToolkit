# -*- coding: utf-8 -*-
"""
Created on Sun Nov 01 22:30:22 2015

@author: Charlie Hagedorn
port-to-python: John Greendeer Lee
"""

import numpy as np
import numpy.random as rand


def cart_2_cyl(XYZ):
    """
    Converts XYZ cartesian coordinate data to RQZ cylindrical data.

    Inputs
    ------
    XYZ : ndarray
        n x 3 array [x,y,z]

    Returns
    -------
    RQZ : ndarray
        n x 3 array [r, q, z] where q is given in radians between [0,2*pi)
    """
    RQZ = np.zeros(np.shape(XYZ))
    RQZ[:, 0] = np.sqrt((XYZ[:, 0]**2 + XYZ[:, 1]**2))
    RQZ[:, 1] = (np.arctan2(XYZ[:, 1], XYZ[:, 0])+2.*np.pi) % (2.*np.pi)
    RQZ[:, 2] = XYZ[:, 2]

    return RQZ


def cyl_2_cart(RQZ):
    """
    Converts RQZ cylindrical coordinate data to XYZ cartesian data.

    Inputs
    ------
    RQZ : ndarray
        n x 3 array [r, q, z] where q is given in radians

    Returns
    -------
    XYZ : ndarray
        n x 3 array [x,y,z]
    """
    XYZ = np.zeros(np.shape(RQZ))
    XYZ[:, 0] = RQZ[:, 0]*np.cos(RQZ[:, 1])
    XYZ[:, 1] = RQZ[:, 0]*np.sin(RQZ[:, 1])
    XYZ[:, 2] = RQZ[:, 2]

    return XYZ


def rectangle(mass, x, y, z, nx, ny, nz):
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
    nx : float
        number of points distributed in x
    nz : float
        number of points distributed in y
    nz : float
        number of points distributed in z

    Returns
    -------
    pointArray : ndarray
        nx*ny*nz x 4 point mass array of format [m, x, y, z]
    """
    pointArray = np.zeros([nx*ny*nz, 4])
    pointArray[:, 0] = mass/float(nx*ny*nz)

    for k in range(nz):
        for l in range(ny):
            for m in range(nx):
                pointArray[k*ny*nx+l*nx+m, 1] = m*x/float(nx-1)-x/2.
                pointArray[k*ny*nx+l*nx+m, 2] = l*y/float(ny-1)-y/2.
                pointArray[k*ny*nx+l*nx+m, 3] = k*z/float(nz-1)-z/2.

    return pointArray


def annulus(mass, iR, oR, t, nx, nz):
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
    nx : float
        number of points distributed in x,y
    nz : float
        number of points distributed in z

    Returns
    -------
    pointArray : ndarray
        point mass array of format [m, x, y, z]
    """
    zgrid = t/float(nz-1)
    xgrid = oR*2./float(nx)
    ygrid = xgrid

    density = mass/(np.pi*(oR**2-iR**2)*t)
    pointMass = density*xgrid*ygrid*zgrid

    pointArray = np.zeros([nx*nx*nz, 4])
    pointArray[:, 0] = pointMass
    for k in range(nz):
        for l in range(nx):
            for m in range(nx):
                pointArray[k*nx*nx+l*nx+m, 1] = m*2*oR/float(nx)-oR
                pointArray[k*nx*nx+l*nx+m, 2] = l*2*oR/float(nx)-oR
                pointArray[k*nx*nx+l*nx+m, 3] = k*t/float(nz-1)-t/2.

    pointArray = np.array([pointArray[k] for k in range(nx*nx*nz) if
                           pointArray[k, 1]**2+pointArray[k, 2]**2 > iR**2 and
                           pointArray[k, 1]**2+pointArray[k, 2]**2 < oR**2])

    # Correct the masses of the points
    pointArray[:, 0] *= mass/np.sum(pointArray[:, 0])

    return pointArray


def cone(mass, R, H, nx, nz):
    """
    Creates point masses distributed in an annulus of mass m.

    Inputs
    ------
    m : float
        mass in kg
    R : float
        outer radius of cone in m
    H : float
        height of cone in m
    nx : float
        number of points distributed in x,y
    nz : float
        number of points distributed in z

    Returns
    -------
    pointArray : ndarray
        point mass array of format [m, x, y, z]
    """
    zgrid = H/float(nz)
    xgrid = R*2./float(nx)
    ygrid = xgrid

    density = mass/(np.pi*H*(R**2)/3.)
    pointMass = density*xgrid*ygrid*zgrid
    tanTheta = H/R

    pointArray = np.zeros([nx*nx*nz, 4])
    pointArray[:, 0] = pointMass
    for k in range(nz):
        for l in range(nx):
            for m in range(nx):
                pointArray[k*nx*nx+l*nx+m, 1] = m*2*R/float(nx)-R
                pointArray[k*nx*nx+l*nx+m, 2] = l*2*R/float(nx)-R
                pointArray[k*nx*nx+l*nx+m, 3] = k*H/float(nz-1)

    pointArray = np.array([pointArray[k] for k in range(nx*nx*nz) if
                           pointArray[k, 1]**2+pointArray[k, 2]**2 <=
                           (pointArray[k, 3]/tanTheta)**2])

    # Correct the masses of the points
    pointArray[:, 0] *= mass/np.sum(pointArray[:, 0])

    return pointArray


def annulus2(mass, iR, oR, t, nr, nq, nz):
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
    nr : float
        number of points distributed in r
    nq : float
        number of points distributed in theta (angle)
    nz : float
        number of points distributed in z

    Returns
    -------
    pointArray : ndarray
        point mass array of format [m, x, y, z]
    """
    Ntot = nr*nq*nz

    pointArray = np.zeros([Ntot, 4])
    pointArray[:, 0] = mass/float(Ntot)
    for k in range(nz):
        for l in range(nq):
            for m in range(nr):
                pointArray[k*nq*nr+l*nr+m, 1] = m*(oR-iR)/float(nr-1) + iR
                pointArray[k*nq*nr+l*nr+m, 2] = l*2*np.pi/float(nq)
                pointArray[k*nq*nr+l*nr+m, 3] = k*t/float(nz-1)-t/2.

    # Convert RQZ to XYZ
    pointArray[:, 1:] = cyl_2_cart(pointArray[:, 1:])

    return pointArray


def spherical_random_shell(mass, r, N):
    """
    Generate a spherical shell with mass m, radius r, and about N points.

    Inputs
    ------
    mass : float
        mass in kg
    r : float
        radius in m
    N : int
        number of points

    Returns
    -------
    pointArray : ndarray
        point mass array of format [m, x, y, z]

    References
    ----------
    Marsaglia (1972)
        http://mathworld.wolfram.com/SpherePointPicking.html
    """
    # generate pairs of variables ~U(-1,1)
    x = rand.rand(int(np.floor(N*4./np.pi)), 2)*2 - 1

    # cut points with x1**2+x2**2 >= 1.
    sx2 = np.sum(x**2, 1)
    x = np.array([x[k] for k in range(len(x)) if sx2[k] < 1.])
    sx2 = np.array([sx2[k] for k in range(len(sx2)) if sx2[k] < 1.])
    sqx2 = np.sqrt(1-sx2)

    nNew = len(x)
    pointArray = np.zeros([nNew, 4])
    pointArray[:, 0] = mass/nNew
    pointArray[:, 1] = 2.*x[:, 0]*sqx2[:]*r
    pointArray[:, 2] = 2.*x[:, 1]*sqx2[:]*r
    pointArray[:, 3] = (1.-2.*sx2[:])*r

    return pointArray


def wedge(mass, iR, oR, t, theta, nx, nz):
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
    nx : float
        number of points distributed in x,y
    nz : float
        number of points distributed in z

    Returns
    -------
    pointArray : ndarray
        point mass array of format [m, x, y, z]
    """
    zgrid = t/nz
    xgrid = oR*2./float(nx)
    ygrid = xgrid

    density = mass/(np.pi*(oR**2-iR**2)*t)
    pointMass = density*xgrid*ygrid*zgrid

    pointArray = np.zeros([nx*nx*nz, 4])
    pointArray[:, 0] = pointMass
    ctr = 0
    for k in range(nz):
        for l in range(nx):
            for m in range(nx):
                # idx = k*nx*nx+l*nx+m
                x = m*2*oR/float(nx)-oR
                y = l*2*oR/float(nx)-oR
                z = k*t/float(nz)-t/2.
                r = np.sqrt(x**2 + y**2)
                q = np.arctan2(y, x)
                if np.abs(q) < theta and r <= oR and r >= iR:
                    pointArray[ctr, 1:] = [x, y, z]
                    ctr += 1

    pointArray = pointArray[:ctr]

    # Correct the masses of the points
    pointArray[:, 0] *= mass/np.sum(pointArray[:, 0])

    return pointArray


def wedge2(mass, iR, oR, t, theta, nr, nq, nz):
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
    theta : float
        Angle subtended by the wedge in radians
    nr : float
        number of points distributed in r
    nq : float
        number of points distributed in theta (angle)
    nz : float
        number of points distributed in z

    Returns
    -------
    pointArray : ndarray
        point mass array of format [m, x, y, z]
    """
    if theta == 2*np.pi:
        pointArray = annulus2(mass, iR, oR, t, nr, nq, nz)
    else:
        Ntot = nr*nq*nz

        pointArray = np.zeros([Ntot, 4])
        pointArray[:, 0] = mass/float(Ntot)
        for k in range(nz):
            for l in range(nq):
                for m in range(nr):
                    idx = k*nq*nr+l*nr+m
                    pointArray[idx, 1] = m*(oR-iR)/float(nr-1) + iR
                    pointArray[idx, 2] = l*theta/float(nq-1) - theta/2.0
                    pointArray[idx, 3] = k*t/float(nz-1)-t/2.

        # Convert RQZ to XYZ
        pointArray[:, 1:] = cyl_2_cart(pointArray[:, 1:])

    return pointArray
