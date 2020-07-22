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
    zgrid = t/float(nz)
    xgrid = oR*2./float(nx)
    ygrid = xgrid

    density = mass/(np.pi*(oR**2-iR**2)*t)
    pointMass = density*xgrid*ygrid*zgrid

    pointArray = np.zeros([nx*nx*nz, 4])
    pointArray[:, 0] = pointMass
    loopCounter = 0
    for k in range(nz):
        for l in range(nx):
            for m in range(nx):
                pointArray[loopCounter, 1] = (m-(nx-1)/2)*xgrid
                pointArray[loopCounter, 2] = (l-(nx-1)/2)*ygrid
                pointArray[loopCounter, 3] = (k-(nz-1)/2)*zgrid
                loopCounter += 1

    pointArray = np.array([pointArray[k] for k in range(nx*nx*nz) if
                           pointArray[k, 1]**2+pointArray[k, 2]**2 >= iR**2 and
                           pointArray[k, 1]**2+pointArray[k, 2]**2 <= oR**2])

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
    zgrid = H/(nz-1)
    # not sure why this works better as 2*R/nx
    # annulus suggests should be 2*R/(nx-1) but test_cone fails
    xgrid = R*2./nx
    ygrid = xgrid

    density = mass/(np.pi*H*(R**2)/3.)
    pointMass = density*xgrid*ygrid*zgrid
    tanTheta = H/R

    pointArray = np.zeros([nx*nx*nz, 4])
    pointArray[:, 0] = pointMass
    for k in range(nz):
        for l in range(nx):
            for m in range(nx):
                pointArray[k*nx*nx+l*nx+m, 1] = m*xgrid-R
                pointArray[k*nx*nx+l*nx+m, 2] = l*ygrid-R
                pointArray[k*nx*nx+l*nx+m, 3] = k*zgrid

    pointArray = np.array([pointArray[k] for k in range(nx*nx*nz) if
                           pointArray[k, 1]**2+pointArray[k, 2]**2 <=
                           (pointArray[k, 3]/tanTheta)**2])

    # Correct the masses of the points
    pointArray[:, 0] *= mass/np.sum(pointArray[:, 0])

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


def wedge(mass, iR, oR, t, beta, nx, nz):
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
    beta : float
        half of the subtended angle in radians
    nx : float
        number of points distributed in x,y
    nz : float
        number of points distributed in z

    Returns
    -------
    pointArray : ndarray
        point mass array of format [m, x, y, z]
    """
    xmax = oR
    if beta < np.pi/2:
        xmin = np.cos(beta)*iR
        ymax = np.sin(beta)*oR
    else:
        xmin = np.cos(beta)*oR
        ymax = oR
    zgrid = t/(nz-1)
    xgrid = (xmax-xmin)/(nx-1)
    ygrid = 2*ymax/(nx-1)

    density = mass/(np.pi*(oR**2-iR**2)*t)
    pointMass = density*xgrid*ygrid*zgrid

    pointArray = np.zeros([nx*nx*nz, 4])
    pointArray[:, 0] = pointMass
    ctr = 0
    for k in range(nz):
        for l in range(nx):
            for m in range(nx):
                # idx = k*nx*nx+l*nx+m
                x = m*xgrid + xmin
                y = l*ygrid - ymax
                z = k*zgrid - t/2
                r = np.sqrt(x**2 + y**2)
                q = np.arctan2(y, x)
                if np.abs(q) <= beta and r <= oR and r >= iR:
                    pointArray[ctr, 1:] = [x, y, z]
                    ctr += 1

    pointArray = pointArray[:ctr]

    # Correct the masses of the points
    pointArray[:, 0] *= mass/np.sum(pointArray[:, 0])

    return pointArray


def trapezoid(mass, iR, oR, t, beta, nx, nz):
    """
    Creates point masses distributed in a trapezoid of mass m. Centered
    vertically about the xy-plane and displaced along the x-axis so that the
    closest point to the origin is at iR*cos(beta).

    Inputs
    ------
    m : float
        mass in kg
    iR : float
        inner radius of trapezoid in m
    oR : float
        outer radius of trapezoid in m
    t : float
        thickness of trapezoid in m
    beta : float
        half of the subtended angle in radians, must be less than pi/2
    nx : float
        number of points distributed in x,y
    nz : float
        number of points distributed in z

    Returns
    -------
    pointArray : ndarray
        point mass array of format [m, x, y, z]
    """
    xclose = iR*np.cos(beta)
    xfar = oR*np.cos(beta)
    d = oR*np.sin(beta)
    zgrid = t/(nz-1)
    xgrid = (xfar-xclose)/(nx-1)
    ygrid = 2*d/(nx-1)

    density = mass/(np.pi*(oR**2-iR**2)*t)
    pointMass = density*xgrid*ygrid*zgrid

    pointArray = np.zeros([nx*nx*nz, 4])
    if beta >= np.pi/2:
        return pointArray
    pointArray[:, 0] = pointMass
    ctr = 0
    for k in range(nz):
        for l in range(nx):
            for m in range(nx):
                # idx = k*nx*nx+l*nx+m
                x = m*xgrid + xclose
                y = l*ygrid - d
                z = k*zgrid - t/2
                q = np.arctan2(y, x)
                if np.abs(q) <= beta:
                    pointArray[ctr, 1:] = [x, y, z]
                    ctr += 1

    pointArray = pointArray[:ctr]

    # Correct the masses of the points
    pointArray[:, 0] *= mass/np.sum(pointArray[:, 0])

    return pointArray


def outer_cone(mass, iR, oR, H, beta, nx, nz):
    """
    Creates point masses distributed in an annular cone of mass m. If iR=0, it
    should be identical to cone. The sloped edge reaches a height H at iR and
    is 0 at oR.

    Inputs
    ------
    m : float
        mass in kg
    iR : float
        inner radius of cone arc
    oR : float
        outer radius of cone arc in m
    H : float
        height of cone in m
    beta : float
        Half-subtended angle of annular cone segment
    nx : float
        number of points distributed in x,y
    nz : float
        number of points distributed in z

    Returns
    -------
    pointArray : ndarray
        point mass array of format [m, x, y, z]
    """
    xmax = oR
    if beta < np.pi/2:
        xmin = np.cos(beta)*iR
        ymax = np.sin(beta)*oR
    else:
        xmin = np.cos(beta)*oR
        ymax = oR
    zgrid = H/(nz-1)
    # not sure why this works better as 2*R/nx
    # annulus suggests should be 2*R/(nx-1) but test_cone fails
    xgrid = (xmax-xmin)/(nx-1)
    ygrid = 2*ymax/(nx-1)
    Hp = H*oR/(oR-iR)
    vol = beta*(Hp*oR**2/3-H*iR**2-(Hp-H)*iR**2/3)

    density = mass/vol
    pointMass = density*xgrid*ygrid*zgrid
    tanPhi = H/(oR-iR)

    pointArray = np.zeros([nx*nx*nz, 4])
    pointArray[:, 0] = pointMass
    ctr = 0
    for k in range(nz):
        for l in range(nx):
            for m in range(nx):
                x = m*xgrid+xmin
                y = l*ygrid-ymax
                z = k*zgrid
                q = np.arctan2(y, x)
                r = np.sqrt(x**2 + y**2)
                if (np.abs(q) <= beta) and (r <= oR-z/tanPhi) and (iR <= r):
                    pointArray[ctr, 1:] = [x, y, z]
                    ctr += 1

    pointArray = pointArray[:ctr]

    # Correct the masses of the points
    pointArray[:, 0] *= mass/np.sum(pointArray[:, 0])

    return pointArray


def tri_prism(mass, d, y1, y2, t, nx, ny, nz):
    """
    Creates point masses distributed in a triangular prism of mass m.

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
    if y2 < y1:
        print('Require y2 > y1')
        return []
    base = np.max([0, y2])-np.min([0, y1])
    zgrid = t/(nz-1)
    xgrid = d/(nx-1)
    ygrid = base/(ny-1)
    print(base, xgrid, ygrid, zgrid)

    density = mass/(t*(y2-y1)*d/2)
    pointMass = density*(xgrid*ygrid*zgrid/2)

    pointArray = np.zeros([nx*ny*nz, 4])
    pointArray[:, 0] = pointMass
    for k in range(nz):
        for l in range(ny):
            for m in range(nx):
                pointArray[k*nx*ny+l*nx+m, 1] = m*xgrid
                pointArray[k*nx*ny+l*nx+m, 2] = l*ygrid
                pointArray[k*nx*ny+l*nx+m, 3] = k*zgrid-t/2.

    pointArray = np.array([pointArray[k] for k in range(nx*ny*nz) if
                           pointArray[k, 1]*y1 <= pointArray[k, 2]*d and
                           pointArray[k, 1]*y2 >= pointArray[k, 2]*d])

    # Correct the masses of the points
    pointArray[:, 0] *= mass/np.sum(pointArray[:, 0])

    return pointArray
