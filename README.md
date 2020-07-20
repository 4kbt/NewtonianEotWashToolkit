# NEWT (Newtonian E&ouml;t-Wash Toolkit)
[![Build Status](https://travis-ci.com/JGLee6/PointGravity.svg?branch=master)](https://travis-ci.com/JGLee6/PointGravity)
## A package for calculating forces and torques with Newtonian gravity from the E&ouml;t-Wash group
### Author (Octave): Charlie Hagedorn
### Author (Python): John G. Lee


## Introduction

This package allows one to calculate forces and torques from extended bodies 
due to a gravitational or Yukawa interaction. It does so in the most simple 
(although not necessarily most efficient) manner imaginable. We simply
discretize our shapes with a 3-dimensional array of points. We can then 
compute the force or torque between each unique pair of points and sum up the 
total for the entire body. Gravity depends only linearly on the source masses, 
so many complicated shapes can be modeled by just concatenating lists of 
individual simple components.

We can also simply visualize the point mass arrays using a 3-d plot.

![Example: Three cylinders comprising object 1(blue), and object 2(orange)](/newt/example/glibEx.png)

### Python implementation

The python implementation of PointGravity is a nearly identical framework of
the work of Dr. Charlie Hagedorn. For instance, to generate the figure shown
above:


```python
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
```

We have also implemented some shapes with grid-spacing and masses weighted
according to a Gauss-Legendre quadrature. We retain the functionality of simple 
translations and rotations while potentially improving convergence rates.


However, the second portion of this package
includes the implementation of a multipole analysis for the same calculations.
It is well known that the gravitational potential can be decomposed into
interactions of multipole moments allowing for accurate and fast calculations
from only a relatively small number of low order moments. We can compute
several low order moments of basic shapes ([ref][6], [ref][7]) or estimate the moments from a
point-mass array. We can rotate these moments to large orders ([ref][4]) and 
translate moments in several ways ([ref][2], [ref][3]). This allows us to
compute the interactions in an entirely different and often useful perspective.
Not only that, but it's WAY FASTER! This portion is based largely on the
private program, MULTIN, of Prof. Adelberger.

Example calculations are explained [here](newt/example/exampleMulti.md)

## References
1. [https://github.com/4kbt/PointGravity][10]
1. [A Sub-Millimeter Parallel-Plate Test of Gravity][1]
1. [Translation of multipoles for a 1/r potential][2]
1. [Interaction potential between extended bodies][3]
1. [Recursive computation of spherical harmonic rotation coefficients of large degree][4]
1. [Multipole calculation of gravitational forces][5]
1. [Analytic expressions for gravitational inner multipole moments of elementary solids and for the force between two rectangular solids][6]
1. [Closed form expressions for gravitational multipole moments of elementary solids][7]
1. [Comparison of the Efficiency of Translation Operators Used in the Fast Multipole Method for the 3D Laplace Equation][8]
1. [Recursions for the computation of multipole translation and rotation coefficients for the 3-D Helmholtz equation][9]

[1]: https://digital.lib.washington.edu/researchworks/handle/1773/34135
[2]: https://journals.aps.org/prd/abstract/10.1103/PhysRevD.55.7970
[3]: https://journals.aps.org/prd/abstract/10.1103/PhysRevD.60.107501
[4]: https://arxiv.org/abs/1403.7698
[5]: https://journals.aps.org/prd/abstract/10.1103/PhysRevD.95.124059
[6]: https://iopscience.iop.org/article/10.1088/0264-9381/23/17/C02
[7]: https://journals.aps.org/prd/abstract/10.1103/PhysRevD.100.124053
[8]: https://drum.lib.umd.edu/handle/1903/3023
[9]: http://users.umiacs.umd.edu/~ramani/pubs/GumerovDuraiswamiSISC03.pdf
[10]: https://github.com/4kbt/PointGravity

## Installation
Clone or download this repository from [here][10], and navigate to the directory
containing the setup.py file. Then simply install using pip.

```python
pip install .
```

## Citing
We ask that anyone using this package cite this [reference][2]

### To Do
- [X] Multipole rotations
- [X] Multipole torques
- [X] Multipole forces
- [X] Multipole shapes
- [X] Multipole shapes adelberger
- [X] multipole shape comparisons
- [X] Multipole translation
- [ ] Recursive translation matrices
- [X] More tests
- [ ] Always more tests!
- [ ] pip package
- [ ] More Doc Strings
- [ ] Implement tests against MULTIN outputs
- [X] Pull request to Charlie's Octave version &#8594; Collaborator
- [X] Outer Multipoles from point-mass
- [X] Example visualization
- [X] Example calculations