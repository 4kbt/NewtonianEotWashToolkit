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


def sphere(mass, R, N):
    """
    Creates point masses distributed in a sphere of mass m. This should only be
    used for visualization, otherwise a sphere should be treated as a point
    gravitationally.

    Inputs
    ------
    mass : float
        mass in kg
    R : float
        x-length of brick in m
    N : float
        number of points distributed in each dimension

    Returns
    -------
    pointArray : ndarray
        nx*ny*nz x 4 point mass array of format [m, x, y, z]
    """
    pointArray = np.zeros([N**3, 4])
    pointArray[:, 0] = mass/N**3
    grid = R/N

    ctr = 0
    for k in range(N):
        for l in range(N):
            for m in range(N):
                x = (m-(N-1)/2)*grid
                y = (l-(N-1)/2)*grid
                z = (k-(N-1)/2)*grid
                r = np.sqrt(x**2 + y**2 + z**2)
                if r <= R:
                    pointArray[ctr, 1:] = [x, y, z]
                    ctr += 1

    pointArray = pointArray[:ctr]

    # Correct the masses of the points
    pointArray[:, 0] *= mass/np.sum(pointArray[:, 0])

    return pointArray


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
    zgrid = z/nz
    xgrid = x/nx
    ygrid = y/ny

    for k in range(nz):
        for l in range(ny):
            for m in range(nx):
                pointArray[k*ny*nx+l*nx+m, 1] = (m-(nx-1)/2)*xgrid
                pointArray[k*ny*nx+l*nx+m, 2] = (l-(nx-1)/2)*ygrid
                pointArray[k*ny*nx+l*nx+m, 3] = (k-(nz-1)/2)*zgrid

    return pointArray


def annulus(mass, r1, r2, t, nx, nz):
    """
    Creates point masses distributed in an annulus of mass m.

    Inputs
    ------
    mass : float
        mass in kg
    r1 : float
        inner radius of annulus in m
    r2 : float
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
    xgrid = r2*2./float(nx)
    ygrid = xgrid

    density = mass/(np.pi*(r2**2-r1**2)*t)
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
                           pointArray[k, 1]**2+pointArray[k, 2]**2 >= r1**2 and
                           pointArray[k, 1]**2+pointArray[k, 2]**2 <= r2**2])

    # Correct the masses of the points
    pointArray[:, 0] *= mass/np.sum(pointArray[:, 0])

    return pointArray


def cone(mass, R, H, beta, nx, nz):
    """
    Creates point masses distributed as a section of a cone of with apex at z=H
    and base radius of R.

    Inputs
    ------
    mass : float
        mass in kg
    R : float
        outer radius of cone in m
    H : float
        height of cone in m
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
    xmax = R
    if beta < np.pi/2:
        xmin = 0
        ymax = np.sin(beta)*R
    else:
        xmin = np.cos(beta)*R
        ymax = R
    zgrid = H/nz
    xgrid = (xmax-xmin)/nx
    xave = (xmax+xmin)/2
    ygrid = 2*ymax/nx

    density = mass/(np.pi*H*(R**2)/3.)
    pointMass = density*xgrid*ygrid*zgrid

    pointArray = np.zeros([nx*nx*nz, 4])
    pointArray[:, 0] = pointMass
    ctr = 0
    for k in range(nz):
        for l in range(nx):
            for m in range(nx):
                x = (m-(nx-1)/2)*xgrid + xave
                y = (l-(nx-1)/2)*ygrid
                z = (k-(nz-1)/2)*zgrid + H/2
                r = np.sqrt(x**2 + y**2)
                q = np.arctan2(y, x)
                if np.abs(q) <= beta and r**2 <= R**2*(1-z/H)**2:
                    pointArray[ctr, 1:] = [x, y, z]
                    ctr += 1

    pointArray = pointArray[:ctr]

    # Correct the masses of the points
    pointArray[:, 0] *= mass/np.sum(pointArray[:, 0])

    return pointArray


def spherical_random_shell(mass, r, N):
    """
    Generate a spherical shell with radius r and about N points.

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


def wedge(mass, r1, r2, t, beta, nx, nz):
    """
    Creates point masses distributed as a section of an annulus of inner and
    outer radius r1 and r2, thickness t, and half-subtended angle beta.

    Inputs
    ------
    m : float
        mass in kg
    r1 : float
        inner radius of annulus in m
    r2 : float
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
    xmax = r2
    if beta < np.pi/2:
        xmin = np.cos(beta)*r1
        ymax = np.sin(beta)*r2
    else:
        xmin = np.cos(beta)*r2
        ymax = r2
    zgrid = t/nz
    xgrid = (xmax-xmin)/nx
    xave = (xmax+xmin)/2
    ygrid = 2*ymax/nx

    density = mass/(np.pi*(r2**2-r1**2)*t)
    pointMass = density*xgrid*ygrid*zgrid

    pointArray = np.zeros([nx*nx*nz, 4])
    pointArray[:, 0] = pointMass
    ctr = 0
    for k in range(nz):
        for l in range(nx):
            for m in range(nx):
                # idx = k*nx*nx+l*nx+m
                x = (m-(nx-1)/2)*xgrid + xave
                y = (l-(nx-1)/2)*ygrid
                z = (k-(nz-1)/2)*zgrid
                r = np.sqrt(x**2 + y**2)
                q = np.arctan2(y, x)
                if np.abs(q) <= beta and r <= r2 and r >= r1:
                    pointArray[ctr, 1:] = [x, y, z]
                    ctr += 1

    pointArray = pointArray[:ctr]

    # Correct the masses of the points
    pointArray[:, 0] *= mass/np.sum(pointArray[:, 0])

    return pointArray


def trapezoid(mass, r1, r2, t, beta, nx, nz):
    """
    Creates point masses distributed in a trapezoid of mass m. Centered
    vertically about the xy-plane and displaced along the x-axis so that the
    closest point to the origin is at r1*cos(beta).

    Inputs
    ------
    m : float
        mass in kg
    r1 : float
        inner radius of trapezoid in m
    r2 : float
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
    xclose = r1*np.cos(beta)
    xfar = r2*np.cos(beta)
    d = r2*np.sin(beta)
    zgrid = t/nz
    xgrid = (xfar-xclose)/nx
    xave = (xfar+xclose)/2
    ygrid = 2*d/nx

    density = mass/(np.pi*(r2**2-r1**2)*t)
    pointMass = density*xgrid*ygrid*zgrid

    pointArray = np.zeros([nx*nx*nz, 4])
    if beta >= np.pi/2:
        return pointArray
    pointArray[:, 0] = pointMass
    ctr = 0
    for k in range(nz):
        for l in range(nx):
            for m in range(nx):
                x = (m-(nx-1)/2)*xgrid + xave
                y = (l-(nx-1)/2)*ygrid
                z = (k-(nz-1)/2)*zgrid
                q = np.arctan2(y, x)
                if np.abs(q) <= beta:
                    pointArray[ctr, 1:] = [x, y, z]
                    ctr += 1

    pointArray = pointArray[:ctr]

    # Correct the masses of the points
    pointArray[:, 0] *= mass/np.sum(pointArray[:, 0])

    return pointArray


def outer_cone(mass, r1, r2, H, beta, nx, nz):
    """
    Creates point masses distributed in an annular cone of mass m. If r1=0, it
    should be identical to cone. The sloped edge reaches a height H at r1 and
    is 0 at r2.

    Inputs
    ------
    m : float
        mass in kg
    r1 : float
        inner radius of cone arc
    r2 : float
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
    xmax = r2
    if beta < np.pi/2:
        xmin = np.cos(beta)*r1
        ymax = np.sin(beta)*r2
    else:
        xmin = np.cos(beta)*r2
        ymax = r2
    zgrid = H/nz
    xgrid = (xmax-xmin)/nx
    xave = (xmax+xmin)/2
    ygrid = 2*ymax/nx
    Hp = H*r2/(r2-r1)
    vol = beta*(Hp*r2**2/3-H*r1**2-(Hp-H)*r1**2/3)

    density = mass/vol
    pointMass = density*xgrid*ygrid*zgrid
    tanPhi = H/(r2-r1)

    pointArray = np.zeros([nx*nx*nz, 4])
    pointArray[:, 0] = pointMass
    ctr = 0
    for k in range(nz):
        for l in range(nx):
            for m in range(nx):
                x = (m-(nx-1)/2)*xgrid + xave
                y = (l-(nx-1)/2)*ygrid
                z = (k-(nz-1)/2)*zgrid + H/2
                q = np.arctan2(y, x)
                r = np.sqrt(x**2 + y**2)
                if (np.abs(q) <= beta) and (r <= r2-z/tanPhi) and (r1 <= r):
                    pointArray[ctr, 1:] = [x, y, z]
                    ctr += 1

    pointArray = pointArray[:ctr]

    # Correct the masses of the points
    pointArray[:, 0] *= mass/np.sum(pointArray[:, 0])

    return pointArray


def tri_prism(mass, d, y1, y2, t, nx, ny, nz):
    """
    Creates point masses distributed in a triangular prism with vertices at
    (x, y, z) = (0, 0, +/-t/2), (d, y1, +/-t/2), (d, y2, +/-t/2) with y2 > y1.

    Inputs
    ------
    m : float
        mass in kg
    r1 : float
        inner radius of annulus in m
    r2 : float
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
    yave = base/2
    zgrid = t/nz
    xgrid = d/nx
    ygrid = base/ny

    density = mass/(t*(y2-y1)*d/2)
    pointMass = density*(xgrid*ygrid*zgrid/2)

    pointArray = np.zeros([nx*ny*nz, 4])
    pointArray[:, 0] = pointMass
    for k in range(nz):
        for l in range(ny):
            for m in range(nx):
                pointArray[k*nx*ny+l*nx+m, 1] = (m-(nx-1)/2)*xgrid + d/2
                pointArray[k*nx*ny+l*nx+m, 2] = (l-(ny-1)/2)*ygrid + yave
                pointArray[k*nx*ny+l*nx+m, 3] = (k-(nz-1)/2)*zgrid

    pointArray = np.array([pointArray[k] for k in range(nx*ny*nz) if
                           pointArray[k, 1]*y1 <= pointArray[k, 2]*d and
                           pointArray[k, 1]*y2 >= pointArray[k, 2]*d])

    # Correct the masses of the points
    pointArray[:, 0] *= mass/np.sum(pointArray[:, 0])

    return pointArray


def tetrahedron(mass, x, y1, y2, z, nx, ny, nz):
    """
    A tetrahedron with vertices at (x,y,z) = (x,y1,0), (x,y2,0), (0,0,0), and
    (0,0,z).

    Inputs
    ------
    m : float
        mass in kg
    x : float
        X-position of first and second vertices
    y1 : float
        Y-position of first vertex
    y2 : float
        Y-position of second vertex
    z : float
        Distance to vertex along z-axis
    nx : float
        number of points distributed in x
    ny : float
        number of points distributed in y
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
    yave = base/2
    zgrid = z/nz
    xgrid = x/nx
    ygrid = base/ny
    print(xgrid, ygrid)

    density = mass/(z*(y2-y1)*x/6)
    pointMass = density*(xgrid*ygrid*zgrid/2)

    pointArray = np.zeros([nx*ny*nz, 4])
    pointArray[:, 0] = pointMass
    ctr = 0
    for k in range(nz):
        for l in range(ny):
            for m in range(nx):
                xval = (m-(nx-1)/2)*xgrid + x/2
                yval = (l-(ny-1)/2)*ygrid + yave
                zval = (k-(nz-1)/2)*zgrid + z/2
                yx = yval*x
                xr = xval/x
                if (xval*y1 <= yx) and (xval*y2 >= yx) and (zval <= z*(1-xr)):
                    pointArray[ctr, 1:] = xval, yval, zval
                    ctr += 1

    pointArray = pointArray[:ctr]

    # Correct the masses of the points
    pointArray[:, 0] *= mass/np.sum(pointArray[:, 0])

    return pointArray


def ngon_prism(mass, H, a, N, nx, nz):
    """
    Creates point masses distributed in a regular N-gon prism of thickness H
    and with sides of length a.

    Inputs
    ------
    mass : float
        Mass of the prism
    H : float
        Total height of the prism
    a : float
        Length of sides of prism
    N : int
        Number of sides to regular prism
    nx : float
        number of points distributed in x,y
    nz : float
        number of points distributed in z

    Returns
    -------
    pointArray : ndarray
        point mass array of format [m, x, y, z]
    """
    phih = np.pi/N
    b = a/(2*np.tan(phih))
    d = a/(2*np.sin(phih))
    zgrid = H/nz
    xgrid = 2*d/nx
    ygrid = xgrid

    density = mass/(N*H*a*b/2)
    pointMass = density*(xgrid*ygrid*zgrid/2)

    pointArray = np.zeros([nx*nx*nz, 4])
    pointArray[:, 0] = pointMass
    ctr = 0
    for k in range(nz):
        for l in range(nx):
            for m in range(nx):
                x = (m-(nx-1)/2)*xgrid
                y = (l-(nx-1)/2)*ygrid
                z = (k-(nz-1)/2)*zgrid
                qsect = np.arctan2(y, x) % (2*phih)
                if qsect > phih:
                    qsect -= 2*phih
                r = np.sqrt(x**2 + y**2)
                xsect = r*np.cos(qsect)
                if xsect <= b:
                    pointArray[ctr, 1:] = x, y, z
                    ctr += 1

    pointArray = pointArray[:ctr]

    # Correct the masses of the points
    pointArray[:, 0] *= mass/np.sum(pointArray[:, 0])

    return pointArray


def pyramid(mass, x, y, z, nx, ny, nz):
    """
    A rectangular pyramid extending above the xy-plane by a height z. The
    rectangular base of the pyramid has vertices at (x,y) = (x, y), (x, -y),
    (-x, y), and (-x, -y).

    Inputs
    ------
    mass : float
        mass in kg
    x : float
        Half-length of rectangular base of pyramid
    y : float
        Half-width of rectangular base of pyramid
    z : float
        Height of pyramid
    nx : float
        number of points distributed in x
    nx : float
        number of points distributed in x,y
    nz : float
        number of points distributed in z

    Returns
    -------
    pointArray : ndarray
        point mass array of format [m, x, y, z]
    """
    zgrid = z/nz
    xgrid = x/nx
    ygrid = y/ny
    print(xgrid, ygrid)

    density = mass/(z*y*x/3)
    pointMass = density*(xgrid*ygrid*zgrid/3)

    pointArray = np.zeros([nx*ny*nz, 4])
    pointArray[:, 0] = pointMass
    ctr = 0
    for k in range(nz):
        for l in range(ny):
            for m in range(nx):
                xval = (m-(nx-1)/2)*xgrid
                yval = (l-(ny-1)/2)*ygrid
                zval = (k-(nz-1)/2)*zgrid + z/2
                yx = yval*x/2
                xy = xval*y/2
                xr = 2*xval/x
                yr = 2*yval/y
                # upper quad
                if (xy <= yx) and (xy >= -yx) and (zval <= z*(1-yr)):
                    pointArray[ctr, 1:] = xval, yval, zval
                    ctr += 1
                # lower quad
                elif (xy >= yx) and (xy <= -yx) and (zval <= z*(1+yr)):
                    pointArray[ctr, 1:] = xval, yval, zval
                    ctr += 1
                # left quad
                elif (xy <= yx) and (xy <= -yx) and (zval <= z*(1+xr)):
                    pointArray[ctr, 1:] = xval, yval, zval
                    ctr += 1
                # right quad
                elif (xy >= yx) and (xy >= -yx) and (zval <= z*(1-xr)):
                    pointArray[ctr, 1:] = xval, yval, zval
                    ctr += 1

    pointArray = pointArray[:ctr]

    # Correct the masses of the points
    pointArray[:, 0] *= mass/np.sum(pointArray[:, 0])

    return pointArray


def cylhole(mass, r, R, nx, nz):
    """
    The shape consists of the volume that would be removed by drilling a hole
    of radius r into a cylinder of radius R. The symmetry axis of the hole is
    along zhat, and the cylinder has its symmetry axis along yhat.

    Inputs
    ------
    m : float
        mass in kg
    r : float
        smaller radius extended along z
    R : float
        larger radius extended along x
    nx : float
        number of points distributed in x,y
    nz : float
        number of points distributed in z

    Returns
    -------
    pointArray : ndarray
        point mass array of format [m, x, y, z]
    """
    zgrid = 2*R/nz
    xgrid = 2*r/nx
    ygrid = xgrid

    # guess at volume, will correct for later
    density = mass/(16*R*r**2/3)
    pointMass = density*xgrid*ygrid*zgrid

    pointArray = np.zeros([nx*nx*nz, 4])
    pointArray[:, 0] = pointMass
    ctr = 0
    for k in range(nz):
        for l in range(nx):
            for m in range(nx):
                x = (m-(nx-1)/2)*xgrid
                y = (l-(nx-1)/2)*ygrid
                z = (k-(nz-1)/2)*zgrid
                rz = np.sqrt(x**2 + y**2)
                rx = np.sqrt(y**2 + z**2)
                if rz <= r and rx <= R:
                    pointArray[ctr, 1:] = x, y, z
                    ctr += 1

    pointArray = pointArray[:ctr]

    # Correct the masses of the points
    pointArray[:, 0] *= mass/np.sum(pointArray[:, 0])

    return pointArray


def platehole(mass, t, r, theta, nx, ny, nz):
    """
    This shape consists of the volume that would be removed by drilling a hole
    of radius r through a parallel-sided plate of thickness t. The plate is
    centered on the xy-plane. The hole axis, which passes through the origin,
    lies in the xz-plane at an angle theta measured from zhat, where -pi/2 <
    theta < pi/2.

    Inputs
    ------
    m : float
        mass in kg
    t : float
        Thickness of rectangular plate, centered on xy-plane
    r : float
        Radius of cylindrical hole
    theta : float
        Angle of hole relative to z axis, tilted toward x axis.
    nx : float
        number of points distributed in x,y
    nz : float
        number of points distributed in z

    Returns
    -------
    pointArray : ndarray
        point mass array of format [m, x, y, z]
    """
    s = np.tan(theta)
    x = 2*(r + t*s/2)
    zgrid = t/nz
    xgrid = 2*x/nx
    ygrid = 2*r/ny

    # guess at volume, will correct for later
    density = mass/(np.pi*np.sqrt(1+s**2)*t*r**2)
    pointMass = density*xgrid*ygrid*zgrid

    pointArray = np.zeros([nx*ny*nz, 4])
    pointArray[:, 0] = pointMass
    ctr = 0
    for k in range(nz):
        for l in range(ny):
            for m in range(nx):
                x = (m-(nx-1)/2)*xgrid
                y = (l-(ny-1)/2)*ygrid
                z = (k-(nz-1)/2)*zgrid
                xz = z*s
                if (y**2 + (x-xz)**2) <= r**2:
                    pointArray[ctr, 1:] = x, y, z
                    ctr += 1

    pointArray = pointArray[:ctr]

    # Correct the masses of the points
    pointArray[:, 0] *= mass/np.sum(pointArray[:, 0])

    return pointArray
