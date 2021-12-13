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
    mesh : madcad mesh
    """
    result = cq.Workplane("XY").sphere(R)
    result.pmass = pmass
    # mesh = mc.icosphere(mc.O, R)
    # mesh.options["pmass"] = pmass
    return result


def cylinder(pmass, H, R, res=None):
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
    mesh : madcad mesh
    """
    result = cq.Workplane("XY").cylinder(H,R)
    result.pmass = pmass
    # vRb = mc.vec3(R, 0, -H/2)
    # vRt = mc.vec3(R, 0, H/2)
    # vOb = mc.vec3(0, 0, -H/2)
    # vOt = mc.vec3(0, 0, H/2)
    # s = mc.flatsurface(mc.web([mc.Segment(vOb,vRb), mc.Segment(vRb,vRt), mc.Segment(vRt,vOt), mc.Segment(vOt,vOb)]))
    # mesh = mc.revolution(mc.radians(360),(mc.O,mc.Z),s,resolution=res)
    # mesh.options["pmass"] = pmass
    return result


def annulus(pmass, H, Ri, Ro, phic, phih, res=None):
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
    mesh : madcad mesh
    """
    result = (cq.Workplane("XZ", origin=((Ro+Ri)/2,0,0))
        .rect(Ri-Ro, H)
        .revolve(2*phih*180/np.pi,(-(Ro+Ri)/2,0,0),(-(Ro+Ri)/2,1,0))
        .rotate((0,0,0), (0,0,1), (phic-phih)*180/np.pi)
    )
    result.pmass = pmass
    # Rix = Ri*np.cos(-phih)
    # Riy = Ri*np.sin(-phih)
    # Rox = Ro*np.cos(-phih)
    # Roy = Ro*np.sin(-phih)
    # vr1b = mc.vec3(Rix, Riy, -H/2)
    # vr1t = mc.vec3(Rix, Riy, H/2)
    # vr2b = mc.vec3(Rox, Roy, -H/2)
    # vr2t = mc.vec3(Rox, Roy, H/2)
    # side1 = mc.Segment(vr1b, vr1t)
    # side2 = mc.Segment(vr1t, vr2t)
    # side3 = mc.Segment(vr2t, vr2b)
    # side4 = mc.Segment(vr2b, vr1b)
    # surf = mc.flatsurface(mc.web([side1, side2, side3, side4]))
    # mesh = mc.revolution(2*phih, (mc.O, mc.Z), surf, resolution=res)
    # mesh = rotate_mesh(mesh, phic, 0, 0)
    # mesh.options["pmass"] = pmass
    return result


def cone(pmass, P, R, phic, phih):
    """
    Cone with apex at z=P and base radius of R.

    Inputs
    ------
    pmass : bool
        bool that indicates whether mesh has positive mass
    P : float
        Total height of the cone section, extends from the xy-plane up to z=P.
    R : float
        Radius of the base of the cone section
    phic : float
        Average angle of cone section, in radians
    phih : float
        Half of the total angular span of the cone section, in radians

    Returns
    -------
    mesh : madcad mesh
    """
    result = (cq.Workplane("XZ")
        .lineTo(R, 0).lineTo(0, P)
        .close().revolve(2*phih,(0,0,0),(0,1,0))
    )
    result.pmass = pmass
    result = rotate_mesh(result, phic-phih)
    # Rx = R*np.cos(phic-phih)
    # Ry = R*np.sin(phic-phih)
    # vlowerR = mc.vec3(Rx, Ry, 0)
    # upperO = mc.vec3(0, 0, P)
    # s1 = mc.Segment(mc.O, vlowerR)
    # s2 = mc.Segment(vlowerR, upperO)
    # s3 = mc.Segment(upperO, mc.O)
    # w = mc.web([s1, s2, s3])
    # mesh = mc.revolution(mc.radians(2*phih), (mc.O, mc.Z), w)
    # mesh.options["pmass"] = pmass
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
    mesh : madcad mesh
    """
    result = (cq.Workplane("XY")
        .lineTo(d,-a/2).lineTo(d,a/2)
        .close().extrude(H/2,both=True)
    )
    result.pmass = pmass
    result = rotate_mesh(result, phic, 0, 0)
    # va1 = mc.vec3(d, a/2, 0)
    # va2 = mc.vec3(d, -a/2, 0)
    # soa1 = mc.Segment(mc.O, va1)
    # sa1a2 = mc.Segment(va1, va2)
    # sa2o = mc.Segment(va2, mc.O)
    # mesh = mc.flatsurface(mc.web([soa1, sa1a2, sa2o]))
    # mesh = mc.thicken(mesh, H, 0.5)
    # mesh = rotate_mesh(mesh, phic, 0, 0)
    # mesh.options["pmass"] = pmass
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
    # Rxp = R*np.cos(phic+phih)
    # Ryp = R*np.sin(phic+phih)
    # Rxm = R*np.cos(phic-phih)
    # Rym = R*np.sin(phic-phih)
    # vR1 = mc.vec3(Rxp, Ryp, 0)
    # vR2 = mc.vec3(Rxm, Rym, 0)
    # soR1 = mc.Segment(mc.O, vR1)
    # sR1R2 = mc.Segment(vR1, vR2)
    # sR2o = mc.Segment(vR2, mc.O)
    # mesh = mc.flatsurface(mc.web([soR1, sR1R2, sR2o]))
    # mesh = mc.thicken(mesh, H, 0.5)
    # mesh.options["pmass"] = pmass
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
    mesh : madcad mesh
    """
    result = (cq.Workplane("XY")
        .lineTo(d,y1).lineTo(d,y2)
        .close().extrude(H/2,both=True)
    )
    result.pmass = pmass
    # vdy1 = mc.vec3(d, y1, 0)
    # vdy2 = mc.vec3(d, y2, 0)
    # sody1 = mc.Segment(mc.O, vdy1)
    # sdy1dy2 = mc.Segment(vdy1, vdy2)
    # sdy2o = mc.Segment(vdy2, mc.O)
    # mesh = mc.flatsurface(mc.web([sody1, sdy1dy2, sdy2o]))
    # mesh = mc.thicken(mesh, H, 0.5)
    # mesh.options["pmass"] = pmass
    return result


def rect_prism(pmass, H, a, b):
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
    mesh : madcad mesh
    """
    result = cq.Workplane("XY").box(a, b, H)
    result.pmass = pmass
    # mesh = mc.brick(width=mc.vec3(a, b, H))
    # mesh.options["pmass"] = pmass
    return result


def trapezoid(pmass, w1, w2, H, thickness):
    """
    Trapezoidal prism with longer face centered at origin.

    Inputs
    ------
    pmass : bool
        bool that indicates whether mesh has positive mass
    w1 : float
        width of bottom
    w2 : float
        width of top
    H : float
        height
    thickness : float
        thickness

    Returns
    -------
    mesh : madcad mesh
    """
    result = (cq.Workplane("XY")
        .rect(w1, thickness)
        .workplane(offset=H)
        .rect(w2,thickness)
        .loft()
    )
    result.pmass = pmass
    # v1 = mc.vec3(w1/2, 0, 0)
    # v2 = mc.vec3(w2/2, H, 0)
    # v3 = mc.vec3(-w2/2, H, 0)
    # v4 = mc.vec3(-w1/2, 0, 0)
    # s1 = mc.Segment(v1, v2)
    # s2 = mc.Segment(v2, v3)
    # s3 = mc.Segment(v3, v4)
    # s4 = mc.Segment(v4, v1)
    # mesh = mc.flatsurface(mc.web([s1, s2, s3, s4]))
    # mesh = mc.thicken(mesh, thickness, 0.5)
    # mesh.options["pmass"] = pmass
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
    # points = []
    # for i in range(N):
    #     rx = r*np.cos(phic-ang/2+i*ang)
    #     ry = r*np.sin(phic-ang/2+i*ang)
    #     points.append(mc.vec3(rx, ry, 0))
    # segments = []
    # for i in range(len(points)-1):
    #     segments.append(mc.Segment(points[i], points[i+1]))
    # segments.append(mc.Segment(points[-1],points[0]))
    # mesh = mc.flatsurface(mc.web(segments))
    # mesh = mc.thicken(mesh, h, 0.5)
    # mesh.options["pmass"] = pmass
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
    mesh : madcad mesh
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
    # vx = mc.vec3(x, 0, 0)
    # vy = mc.vec3(0, y, 0)
    # vz = mc.vec3(0, 0, z)
    # f1 = mc.web([mc.Segment(mc.O, vx), mc.Segment(vx, vy), mc.Segment(vy, mc.O)])
    # f2 = mc.web([mc.Segment(mc.O, vx), mc.Segment(vx, vz), mc.Segment(vz, mc.O)])
    # f3 = mc.web([mc.Segment(mc.O, vz), mc.Segment(vz, vy), mc.Segment(vy, mc.O)])
    # f4 = mc.web([mc.Segment(vy, vx), mc.Segment(vx, vz), mc.Segment(vz, vy)])
    # side1 = mc.flatsurface(f1)
    # side2 = mc.flatsurface(f2)
    # side3 = mc.flatsurface(f3)
    # side4 = mc.flatsurface(f4)
    # mesh = side1+side2+side3+side4
    # mesh.options["pmass"] = pmass
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
    mesh : madcad mesh
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
    # v1 = mc.vec3(x, y1, 0)
    # v2 = mc.vec3(x, y2, 0)
    # vz = mc.vec3(0, 0, z0)
    # f1 = mc.web([mc.Segment(mc.O, v1), mc.Segment(v1, v2), mc.Segment(v2, mc.O)])
    # f2 = mc.web([mc.Segment(mc.O, v1), mc.Segment(v1, vz), mc.Segment(vz, mc.O)])
    # f3 = mc.web([mc.Segment(mc.O, vz), mc.Segment(vz, v2), mc.Segment(v2, mc.O)])
    # f4 = mc.web([mc.Segment(v2, v1), mc.Segment(v1, vz), mc.Segment(vz, v2)])
    # side1 = mc.flatsurface(f1)
    # side2 = mc.flatsurface(f2)
    # side3 = mc.flatsurface(f3)
    # side4 = mc.flatsurface(f4)
    # mesh = side1+side2+side3+side4
    # mesh.options["pmass"] = pmass
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
    mesh : madcad mesh
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
    # vb1 = mc.vec3(x/2, y/2)
    # vb2 = mc.vec3(-x/2, y/2)
    # vb3 = mc.vec3(-x/2, -y/2)
    # vb4 = mc.vec3(x/2, -y/2)
    # vt = mc.vec3(0, 0, z)
    # fb = mc.web([mc.Segment(vb1, vb2), mc.Segment(vb2, vb3), mc.Segment(vb3, vb4), mc.Segment(vb4, vb1)])
    # f1 = mc.web([mc.Segment(vb1, vb2), mc.Segment(vb2, vt), mc.Segment(vt, vb1)])
    # f2 = mc.web([mc.Segment(vb2, vb3), mc.Segment(vb3, vt), mc.Segment(vt, vb2)])
    # f3 = mc.web([mc.Segment(vb3, vb4), mc.Segment(vb4, vt), mc.Segment(vt, vb3)])
    # f4 = mc.web([mc.Segment(vb4, vb1), mc.Segment(vb1, vt), mc.Segment(vt, vb4)])
    # b = mc.flatsurface(fb)
    # side1 = mc.flatsurface(f1)
    # side2 = mc.flatsurface(f2)
    # side3 = mc.flatsurface(f3)
    # side4 = mc.flatsurface(f4)
    # mesh = b+side1+side2+side3+side4
    # mesh.options["pmass"] = pmass
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
    mesh : madcad mesh
    """
    mr = cylinder(0, r, 2*R)
    mR = cylinder(0, R, 2*r)
    mR = mR.rotate((0,0,0), (1,0,0), 90)
    result = mr.intersect(mR)
    result.pmass = pmass
    # mR = mR.transform(rotatearound(mc.radians(90), mc.O, mc.X))
    # mesh = intersection(mr, mR)
    # mesh.options["pmass"] = pmass
    return result


def translate_mesh(mesh, rPrime):
    """
    Applies translation to madcad mesh
    Inputs
    ------
    mesh : madcad mesh
    rPrime : list
        x, y, and z coordinates to translate
    Returns
    -------
    newmesh : madcad mesh
    """
    pmass = mesh.pmass
    result = mesh.translate(rPrime)
    result.pmass = pmass
    # newmesh = mesh.transform(mc.vec3(rPrime))
    return result


def rotate_mesh(mesh, alpha, beta, gamma):
    """
    Applies an arbitrary rotation given as Euler angles in z-y-z convention to
    a madcad mesh.

    Inputs
    ------
    mesh : madcad mesh
    alpha : float
        Angle in radians about z-axis
    beta : float
        Angle in radians about y-axis
    gamma : float
        Angle in radians about z-axis

    Returns
    -------
    newmesh: madcad mesh
    """
    pmass = mesh.pmass
    result = mesh.rotate((0,0,0), (0,0,1), alpha*180/np.pi)
    result = result.rotate((0,0,0), (0,1,0), beta*180/np.pi)
    result = result.rotate((0,0,0), (0,0,1), gamma*180/np.pi)
    result.pmass = pmass
    # newmesh = mesh.transform(mc.rotatearound(alpha, mc.O, mc.Z))
    # newmesh = newmesh.transform(mc.rotatearound(beta, mc.O, mc.Y))
    # newmesh = newmesh.transform(mc.rotatearound(gamma, mc.O, mc.Z))
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
    mesh : madcad mesh
    name : name of output file
    Returns
    -------
    None. Ourputs file
    """
    if hasattr(mesh, '__iter__'):
        tmpmesh = sum_mesh(mesh)
        cq.exporters.export(tmpmesh, f"{name}.stl")
        # vertices = np.array(tmpmesh.points)
        # faces = np.array(tmpmesh.faces)
        # out = stl.mesh.Mesh(np.zeros(faces.shape[0], dtype=stl.mesh.Mesh.dtype))
        # for i, f in enumerate(faces):
        #     for j in range(3):
        #         out.vectors[i][j] = vertices[f[j], :]

    else:
        # vertices = np.array(mesh.points)
        # faces = np.array(mesh.faces)
        # out = stl.mesh.Mesh(np.zeros(faces.shape[0], dtype=stl.mesh.Mesh.dtype))
        # for i, f in enumerate(faces):
        #     for j in range(3):
        #         out.vectors[i][j] = vertices[f[j], :]
        cq.exporters.export(mesh, f"{name}.stl")
