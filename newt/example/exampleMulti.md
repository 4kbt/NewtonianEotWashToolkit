# Multipole calculations

We will eventually provide some use cases of the multipole functionality to
this library. Hopefully it will illustrate the effectiveness and speed of the
multipole calculations over the PointGravity version.

### What are multipoles
The multipoles are complex values given for in pairs of indicies (l, m), where 
l is the degree and m the order. These coefficients act similar to a fourier
series expansion, but in 3D.

The lowest degree inner multipole is the (0, 0), which essentially says how
heavy is the object. The l=1 multipoles look similar to a hemisphere oriented 
either vertically (1, 0) or horizontally (1, 1), and tell how asymmetric the 
object is. The l=2 multipoles look like peanuts, and higher degree multipoles 
appear as objects with more lobes.

## Moments of homogeneous solids

The beauty of this method relies on having analytically calculated the moments 
of various shapes centered around the origin. Combining these known moments
with translation and rotation formulae allows us to construct intricate
massive bodies from their combinations.

## Rotations

Rotations are performed via the Wiger-D matrices using a z-y-z Euler angle representation.
For the z-axis rotations, they just amount to a multiplication by a complex 
exponential of the rotation angle for the given order m. However, the y-axis
 rotation is given by the Wigner small-d matrices.

## Translations