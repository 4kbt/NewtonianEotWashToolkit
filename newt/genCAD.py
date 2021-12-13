import numpy as np
import stl
import cadquery as cq

global enabled
enabled= False

def sphere(pmass, R):
    """
    Create sphere with madcad at origin.

    Inputs
    ------
    pmass : bool
        bool that indicates whether mesh has positive mass
    R : float
        Radius of the cylinder

    Returns
    -------
    result : CadQuery object
    """
    result = cq.Workplane("XY").sphere(R)
    result.pmass = pmass
    return result


def cylinder(pmass, H, R):
    """
    The cylinder has a height H and extends above and below the xy-plane by
    H/2.

    Inputs
    ------
    pmass : bool
        bool that indicates whether mesh has positive mass
        Total height of the cylinder
    R : float
        Radius of the cylinder

    Returns
    -------
    result : CadQuery object
    """
    result = cq.Workplane("XY").cylinder(H,R)
    result.pmass = pmass
    return result


def annulus(pmass, H, Ri, Ro, phic, phih):
    """
    The solid has a height H and extends above and below the xy-plane
    by H/2.

    Inputs
    ------
    pmass : bool
        bool that indicates whether mesh has positive mass
    H : float
        Total height of the annular section
    Ri : float
        Inner radius of the annular section
    Ro : float
        Outer radius of the annular section
    phic : float
        Average angle of annular section, in radians
    phih : float
        Half of the total angular span of the annular section, in radians

    Returns
    -------
    result : CadQuery object
    """
    result = (cq.Workplane("XZ", origin=((Ro+Ri)/2,0,0))
        .rect(Ri-Ro, H)
        .revolve(2*phih*180/np.pi,(-(Ro+Ri)/2,0,0),(-(Ro+Ri)/2,1,0))
        .rotate((0,0,0), (0,0,1), (phic-phih)*180/np.pi)
    )
    result.pmass = pmass
    return result


def cone(pmass, H, r1, r2, phic, phih):
    """
    Cone with apex at z=P and base radius of R.

    Inputs
    ------
    pmass : bool
        bool that indicates whether mesh has positive mass
    H : float
        Total height of the cone section, extends from the xy-plane up to z=P.
    r1 : float
        Radius of top of the cone section
    r2 : fload
        Radius at bottom of the cone section
    phic : float
        Average angle of cone section, in radians
    phih : float
        Half of the total angular span of the cone section, in radians

    Returns
    -------
    result : CadQuery object
    """
    if r1 == 0:        
        result = (cq.Workplane("XZ")
            .lineTo(r2, 0).lineTo(0, H)
            .close().revolve(2*phih*180/np.pi,(0,0,0),(0,1,0))
        )
    else:
        result = (cq.Workplane("XZ")
            .lineTo(r2,0).lineTo(r1,H).lineTo(0,H)
            .close().revolve(2*phih*180/np.pi,(0,0,0),(0,1,0))
        )
    result.pmass = pmass
    result = rotate_mesh(result, phic-phih, 0, 0)
    return result


def tri_iso_prism(pmass, H, a, d, phic):
    """
    The isosceles triangular prism has a height H and extends above and below
    the xy-plane by H/2. The triangular faces have vertices at (x,y)=(0,0),
    (d,a/2), and (d,-a/2) when phic=0.

    XXX: Check if d<0 matches expectation

    Inputs
    pmass : bool
        bool that indicates whether mesh has positive mass
    ------
    L : int
        Maximum order for multipole expansion
    mass : float
        Mass of the prism
    H : float
        Total height of the prism
    a : float
        Length of side opposite origin
    d : float
        Distance to the side opposite the origin
    phic : float
        Average angle of prism, in radians

    Returns
    -------
    result : CadQuery object
    """
    result = (cq.Workplane("XY")
        .lineTo(d,-a/2).lineTo(d,a/2)
        .close().extrude(H/2,both=True)
    )
    result.pmass = pmass
    result = rotate_mesh(result, phic, 0, 0)
    return result


def tri_iso_prism2(pmass, H, R, phic, phih):
    """
    The isosceles triangular prism has a height H and extends above and below
    the xy-plane by H/2. The triangular faces span an angle from phic-phih to
    phic + phih where the equal length sides have length R.

    Inputs
    pmass : bool
        bool that indicates whether mesh has positive mass
    ------
    H : float
        Total height of the prism
    R : float
        Length of equal length sides of triangular face
    phic : float
        Average angle of prism, in radians
    phih : float
        Half of the total angular span of the prism, in radians

    Returns
    -------
    qlm : ndarray, complex
        (L+1)x(2L+1) array of complex moment values
    """
    Rx = R*np.cos(phih)
    Ry = R*np.sin(phih)
    result = (cq.Workplane("XY")
        .lineTo(Rx, Ry).lineTo(Rx, -Ry)
        .close().extrude(H/2,both=True)
    )
    result.pmass = pmass
    result = rotate_mesh(result, phic, 0, 0)
    return result


def tri_prism(pmass, H, d, y1, y2):
    """
    The triangular prism has a height H and extends above and below the
    xy-plane by H/2. The triangular faces have vertices at (x,y)=(0,0),
    (d,y1), and (d,y2).

    Inputs
    ------
    pmass : bool
        bool that indicates whether mesh has positive mass
    H : float
        Total height of the prism
    d : float
        X-position of first and second vertices
    y1 : float
        Y-position of first vertex
    y2 : float
        Y-position of second vertex

    Returns
    -------
    result : CadQuery object
    """
    result = (cq.Workplane("XY")
        .lineTo(d,y1).lineTo(d,y2)
        .close().extrude(H/2,both=True)
    )
    result.pmass = pmass
    return result


def rect_prism(pmass, H, a, b, phic):
    """
    Rectangular prism centered on the origin with height H and sides of length
    a and b extending along the x and y axes respectively when phic=0.

    Inputs
    ------
    pmass : bool
        bool that indicates whether mesh has positive mass
    H : float
        Total height of the prism
    a : float
        Length of prism
    b : float
        Width of prism
    phic : float
        Average angle of prism, in radians

    Returns
    -------
    result : CadQuery object
    """
    result = cq.Workplane("XY").box(a, b, H)
    result.pmass = pmass
    result = rotate_mesh(result, phic, 0, 0)
    return result


def trapezoid(pmass, t, w1, w2, h):
    """
    Trapezoidal prism with longer face centered at origin.

    Inputs
    ------
    pmass : bool
        bool that indicates whether mesh has positive mass
    t : float
        thickness along z axis
    w1 : float
        width of -y side
    w2 : float
        width of +y side
    h : float
        height of trapezoid along x axis
    thickness : float
        thickness

    Returns
    -------
    result : CadQuery object
    """
    result = (cq.Workplane("XY")
        .rect(w1, thickness)
        .workplane(offset=H)
        .rect(w2, thickness)
        .loft()
    )
    result.pmass = pmass
    return result


def ngon_prism(pmass, H, a, phic, N):
    """
    Regular N-sided prism centered on the origin with height H with sides of
    length a. When phic=0, the first side is oriented parallel to the y-axis.

    Inputs
    ------
    pmass : bool
        bool that indicates whether mesh has positive mass
    L : int
        Maximum order for multipole expansion
    mass : float
        Mass of the prism
    H : float
        Total height of the prism
    a : float
        Length of sides of prism
    phic : float
        Average angle of prism, in radians
    N : int
        Number of sides to regular prism

    Returns
    -------
    qlmesh : ndarray, complex
        (L+1)x(2L+1) array of complex moment values
    """
    ang = 2*np.pi/N
    R = a/np.sqrt(2*(1-np.cos(ang)))
    result = cq.Workplane("XY").polygon(N, R).extrude(H/2, both=True)
    result.pmass = pmass
    result = rotate_mesh(result, ang/2, 0, 0)
    return result


def tetrahedron(pmass, x, y, z):
    """
    A tetrahedron having three mutually perpendicular triangular faces that
    meet at the origin. The fourth triangular face is defined by points at
    coordinates x, y, and z along the xhat, yhat, and zhat axes respectively.

    Inputs
    ------
    pmass : bool
        bool that indicates whether mesh has positive mass
    x : float
        Distance to vertex along x-axis
    y : float
        Distance to vertex along y-axis
    z : float
        Distance to vertex along z-axis

    Returns
    -------
    result : CadQuery object
    """
    vertices = [[0,0,0], [x,0,0], [0,y,0], [0,0,z]]
    faces_ixs = [[0, 1, 2, 0], [1, 0, 3, 1], [2, 3, 0, 2], [3, 2, 1, 3]]

    faces = []
    for ixs in faces_ixs:
        lines = []
        for v1, v2 in zip(ixs, ixs[1:]):
            lines.append(
                cq.Edge.makeLine(cq.Vector(*vertices[v1]), cq.Vector(*vertices[v2]))
            )
        wire = cq.Wire.combine(lines)
        faces.append(cq.Face.makeFromWires(wire[0]))

    shell = cq.Shell.makeShell(faces)
    solid = cq.Solid.makeSolid(shell)
    result = cq.CQ(solid)
    result.pmass = pmass
    return result


def tetrahedron2(pmass, x, y1, y2, z):
    """
    A tetrahedron with vertices at (x,y,z) = (x,y1,0), (x,y2,0), (0,0,0), and
    (0,0,z).

    Inputs
    ------
    pmass : bool
        bool that indicates whether mesh has positive mass
    x : float
        x position of first and second vertex
    y1 : float
        y position of first vertex
    y2 : float
        y position of second vertex
    z : float
        Distance to vertex along z-axis

    Returns
    -------
    result : CadQuery object
    """
    vertices = [[0,0,0], [x,y1,0], [x,y2,0], [0,0,z]]
    faces_ixs = [[0, 1, 2, 0], [1, 0, 3, 1], [2, 3, 0, 2], [3, 2, 1, 3]]

    faces = []
    for ixs in faces_ixs:
        lines = []
        for v1, v2 in zip(ixs, ixs[1:]):
            lines.append(
                cq.Edge.makeLine(cq.Vector(*vertices[v1]), cq.Vector(*vertices[v2]))
            )
        wire = cq.Wire.combine(lines)
        faces.append(cq.Face.makeFromWires(wire[0]))

    shell = cq.Shell.makeShell(faces)
    solid = cq.Solid.makeSolid(shell)
    result = cq.CQ(solid)
    result.pmass = pmass
    return result


def pyramid(pmass, x, y, z):
    """
    A rectangular pyramid extending above the xy-plane by a height z. The
    rectangular base of the pyramid has vertices at (x,y) = (x, y), (x, -y),
    (-x, y), and (-x, -y).

    Inputs
    ------
    pmass : bool
        bool that indicates whether mesh has positive mass
    x : float
        Half-length of rectangular base of pyramid
    y : float
        Half-width of rectangular base of pyramid
    z : float
        Height of pyramid

    Returns
    -------
    result : CadQuery object
    """
    vertices = [[x/2,y/2,0], [x/2,-y/2,0], [-x/2,-y/2,0], [-x/2,y/2,0] [0,0,z]]
    faces_ixs = [[0,1,2,3,0], [0,1,4,0], [1,2,4,1], [2,3,4,2], [3,0,4,3]]

    faces = []
    for ixs in faces_ixs:
        lines = []
        for v1, v2 in zip(ixs, ixs[1:]):
            lines.append(
                cq.Edge.makeLine(cq.Vector(*vertices[v1]), cq.Vector(*vertices[v2]))
            )
        wire = cq.Wire.combine(lines)
        faces.append(cq.Face.makeFromWires(wire[0]))

    shell = cq.Shell.makeShell(faces)
    solid = cq.Solid.makeSolid(shell)
    result = cq.CQ(solid)
    result.pmass = pmass
    return result


def cylhole(pmass, r, R):
    """
    Intersection of small cylinder through side of larger cylinder

    Inputs
    ------
    pmass : bool
        bool that indicates whether mesh has positive mass
    r : float, radius of hole
    R : float, radius of larger cylinder

    Returns
    -------
    result : CadQuery object
    """
    mr = cylinder(0, r, 2*R)
    mR = cylinder(0, R, 2*r)
    mR = mR.rotate((0,0,0), (1,0,0), 90)
    result = mr.intersect(mR)
    result.pmass = pmass
    return result


def platehole(pmass, t, r, theta):
    """
    This shape consists of the volume that would be removed by drilling a hole
    of radius r through a parallel-sided plate of thickness t. The plate is
    centered on the xy-plane. The hole axis, which passes through the origin,
    lies in the xz-plane at an angle theta measured from zhat, where -pi/2 <
    theta < pi/2. Values are only known up to LMax=5. The density is given by
    rho.

    Inputs
    ------
    LMax : int
        Maximum order of inner multipole moments. Only known to LMax=5.
    pmass : bool
        keeps track of positive/negative density objects
    t : float
        Thickness of rectangular plate, centered on xy-plane
    r : float
        Radius of cylindrical hole
    theta : float
        Angle of hole relative to z axis, tilted toward x axis.

    Returns
    -------
    qlm : ndarray
        (LMax+1)x(2LMax+1) complex array of inner moments
    """
    rect = rect_prism(pmass, t, 2*r, 2*r)
    cyl = cylinder(0, r, 2*t)
    cyl = rotate_mesh(cyl, 0, theta, 0)
    result = rect.intersect(cyl)
    result.pmass = pmass
    return result



def translate_mesh(mesh, rPrime):
    """
    Applies translation to madcad mesh
    Inputs
    ------
    result : CadQuery object
    rPrime : list
        x, y, and z coordinates to translate
    Returns
    -------
    newresult : CadQuery object
    """
    pmass = mesh.pmass
    result = mesh.translate(rPrime)
    result.pmass = pmass
    return result


def rotate_mesh(mesh, alpha, beta, gamma):
    """
    Applies an arbitrary rotation given as Euler angles in z-y-z convention to
    a madcad mesh.

    Inputs
    ------
    result : CadQuery object
    alpha : float
        Angle in radians about z-axis
    beta : float
        Angle in radians about y-axis
    gamma : float
        Angle in radians about z-axis

    Returns
    -------
    result: CadQuery object
    """
    pmass = mesh.pmass
    result = mesh.rotate((0,0,0), (0,0,1), alpha*180/np.pi)
    result = result.rotate((0,0,0), (0,1,0), beta*180/np.pi)
    result = result.rotate((0,0,0), (0,0,1), gamma*180/np.pi)
    result.pmass = pmass
    return result


def sum_mesh(mesh):
    #Assumes you add positive before negative test masses
    tmpmesh = mesh[0]
        # add up all positive masses
    for i in range(1, len(mesh)):
        if mesh[i].pmass:
            tmpmesh = tmpmesh.union(mesh[i])
        else:
            tmpmesh = tmpmesh.cut(mesh[i])
    tmpmesh.pmass = True #Total sum should be positive
    return tmpmesh


def save_stl(mesh, name):
    """
    Saves madcad mesh as stl file

    Inputs
    ------
    result : CadQuery object
    name : name of output file
    Returns
    -------
    None. Outputs file
    """
    if hasattr(mesh, '__iter__'):
        tmpmesh = sum_mesh(mesh)
        cq.exporters.export(tmpmesh, f"{name}.stl",tolerance=.01)

    else:
        cq.exporters.export(mesh, f"{name}.stl",tolerance=.01)
