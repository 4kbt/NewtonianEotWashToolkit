# Shapes with known inner multipole moments

With our methods, we have calculated analytical formulas for the inner moments of various primitive shapes at the origin. We can
then rotate, translate, and add together shapes of various types to create complex solids. A list of our shape primitives with a
description of the shape are given below.

## Cylinder
A section of an annulus of height H with its symmetry axis along z and vertically centered on the xy-plane. The annular section
has inner radius IR and outer radius OR and subtends an azimuthal angle from phic - phih to phic + phih. If IR = 0 and phih = pi,
this is a cylinder of height H and radius OR.

## Cone
An azimuthal section of a cone of height H with its symmetry axis along z extending vertically above the xy-plane. The base of the
cone has a radius R and comes to its apex on the z-axis at z=H. The cone section subtends an azimuthal angle from phic - phih to
phic + phih.

## Isosceles Triangular Prism
The isosceles triangular prism has a height H and extends above and below the xy-plane by H/2. The triangular faces have vertices
at (x,y)=(0,0),(d,a/2), and (d,-a/2) when phic=0 as projected onto the xy-plane.

## Triangular Prism
The triangular prism has a height H and extends above and below the xy-plane by H/2. The triangular faces have vertices at
(x,y)=(0,0),(d,y1), and (d,y2) as projected onto the xy-plane.

## Rectangular Prism
Rectangular prism centered on the origin with height H and sides of length a and b extending along the x and y axes respectively
when phic=0.

## N-gon Prism
Regular N-sided prism centered on the origin with height H with sides of length a. When phic=0, the first side is oriented parallel
to the y-axis.

## Trirectangular Tetrahedron
This shape consists of a tetrahedron having three mutually perpendicular triangular faces that meet at the origin. The fourth
triangular face is defined by points at corrdinates x, y, and z along the xhat, yhat, and zhat axes respectively.

## Tetrahedron
This shape consists of a tetrahedron with vertices at (x,y,z) = (0,0,0),(x,y1,0),(x,y2,0), and (0,0,z).

## Rectangular Pyramid
A rectangular pyramid extending above the xy-plane by a height z. The rectangular base of the pyramid has vertices at (x,y) = (x, y),
(x, -y), (-x, y), and (-x, -y).

## Steinmetz Solid (aka Cylhole) (L < 6)
The shape consists of the volume that would be removed by drilling a hole of radius r into a cylinder of radius R. The symmetry axis
of the hole is along zhat, and the cylinder has its symmetry axis along yhat.

## Plate Hole (L < 6)
This shape consists of the volume that would be removed by drilling a hole of radius r through a parallel-sided plate of thickness t.
The plate is centered on the xy-plane. The hole axis, which passes through the origin, lies in the xz-plane at an angle theta
measured from zhat, where -pi/2 < theta < pi/2.