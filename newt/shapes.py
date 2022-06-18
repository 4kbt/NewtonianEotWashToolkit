# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 20:28:18 2020

@author: jgl6
"""
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as mp3d
import newt.qlm as qlm
import newt.qlmNum as qlmn
import newt.bigQlm as bqlm
import newt.bigQlmNum as bqlmn
import newt.genCAD as gcad
import newt.glib as glb
import newt.translations as trs
import newt.rotations as rot
import newt.multipoleLib as mplb


class Shape:
    """
    A Shape object is an inheritable class for shape primitives with abilities
    to both calculate multipole moments and visualize with cadQuery.
    """
    def __init__(self, lmax, inner=True):
        self.inner = bool(inner)
        self.lmax = lmax
        self.qlm = np.zeros([lmax+1, 2*lmax+1], dtype=complex)
        self.com = np.zeros(3)
        self.mass = 0
        self.mesh = None

    def save_shape(self, filename):
        #fig = plt.figure()
        #ax = mp3d.Axes3D(fig)

        # Load the STL files and add the vectors to the plot
        #self.mesh = gcad.stl.mesh.Mesh.from_file('tests/stl_binary/HalfDonut.stl')
        #ax.add_collection3d(mp3d.art3d.Poly3DCollection(self.mesh.vectors))

        # Auto scale to the mesh size
        #scale = self.mesh.points.flatten()
        #ax.auto_scale_xyz(scale, scale, scale)

        # Show the plot to the screen
        #plt.show()
        gcad.save_stl(self.mesh, filename)
        header = 'Moment file\n'
        if self.inner:
            header += 'Inner moments\n'
        else:
            header += 'Outer moments\n'
        header += 'Mass: ' + str(self.mass) + '\n'
        header += 'Center of Mass: ' + str(self.com) + '\n'
        np.savetxt(filename+'.txt', self.qlm, header=header)

    def rotate(self, alpha, beta, gamma):
        self.qlm = rot.rotate_qlm(self.qlm, alpha, beta, gamma)
        self.mesh = gcad.rotate_mesh(self.mesh, alpha, beta, gamma)

        # Rotate center of mass using point-gravity tools
        rotcom = np.concatenate([[self.mass], self.com])
        rotcom = glb.rotate_point_array(rotcom, gamma, [0, 0, 1])
        rotcom = glb.rotate_point_array(rotcom, beta, [0, 1, 0])
        rotcom = glb.rotate_point_array(rotcom, alpha, [0, 0, 1])
        self.com = rotcom[1:4]

    def translate(self, tvec, to_inner):
        self.com += tvec
        self.mesh = gcad.translate_mesh(self.mesh, tvec)
        if self.inner and to_inner:
            self.qlm = trs.translate_qlm(self.qlm, tvec)
        elif self.inner and not to_inner:
            self.inner = False
            self.qlm = trs.translate_q2Q(self.qlm, tvec)
        elif not self.inner and not to_inner:
            self.qlm = trs.translate_Qlmb(self.qlm, tvec)
        else:
            raise ValueError('Outer to inner translations not allowed.')

    def add(self, shapes):
        """
        Add Shape objects.

        Parameters
        ----------
        shapes : Shape or List
            Shape object or List of Shape objects.

        Raises
        ------
        TypeError
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if isinstance(shapes, list):
            shape_list = [self] + shapes
        elif isinstance(shapes, Shape):
            shape_list = [self] + [shapes]
        else:
            raise TypeError('Shapes should be a Shape object or list of Shapes')
        inner = [shape.inner for shape in shape_list]
        outer = [not shape.inner for shape in shape_list]
        lmax = max([shape.lmax for shape in shape_list])
        if all(inner) or all(outer):
            com, mtot, qlmtot = 0, 0, 0
            for k in range(len(shape_list)):
                shape = shape_list[k]
                com += shape.mass*shape.com
                mtot += shape.mass
                if shape.lmax == lmax:
                    qlmtot += shape.qlm
                else:
                    qlmtot += mplb.embed_qlm(shape.qlm, lmax)
            self.mass = mtot
            self.com = com/mtot
            self.qlm = qlmtot
            self.lmax = lmax
            self.mesh = gcad.sum_mesh([shape.mesh for shape in shape_list])
        else:
            raise TypeError('Trying to combine inner and outer moments')



class Annulus(Shape):
    """
    Annulus with multipoles and cadquery
    """
    def __init__(self, lmax, inner, mass, H, Ri, Ro, phic, phih):
        Shape.__init__(self, lmax, inner)
        self.mass = mass
        if self.inner:
            self.qlm = qlm.annulus(lmax, mass, H, Ri, Ro, phic, phih)
        else:
            dens = mass/(2*phih*(Ro**2 - Ri**2)*H)
            self.qlm = bqlm.annulus(lmax, dens, H/2, Ri, Ro, phic, phih)
            self.qlm += bqlm.annulus(lmax, dens, -H/2, Ri, Ro, phic, phih)
        self.mesh = gcad.annulus(mass > 0, H, Ri, Ro, phic, phih)


class RectPrism(Shape):
    """
    Rectangular prism with multipoles and cadquery.
    """
    def __init__(self, lmax, inner, mass, H, a, b, phic):
        Shape.__init__(self, lmax, inner)
        self.mass = mass
        if self.inner:
            self.qlm = qlm.rect_prism(lmax, mass, H, a, b, phic)
        else:
            print('Outer rectangular prism shape not defined')
        self.mesh = gcad.rect_prism(mass > 0, H, a, b, phic)


class TriPrism(Shape):
    """
    Triangular prism with multipoles and cadquery.
    """
    def __init__(self, lmax, inner, mass, H, d, y1, y2):
        Shape.__init__(self, lmax, inner)
        self.mass = mass
        if self.inner:
            self.qlm = qlm.tri_prism(lmax, mass, H, d, y1, y2)
        else:
            print('Outer triangular prism shape not defined')
        self.mesh = gcad.rect_prism(mass > 0, H, d, y1, y2)


class Cone(Shape):
    """
    Cone with multipoles and cadquery.
    """
    def __init__(self, lmax, inner, mass, P, R, phic, phih):
        Shape.__init__(self, lmax, inner)
        self.mass = mass
        if self.inner:
            self.qlm = qlm.tri_prism(lmax, mass, P, R, phic, phih)
        else:
            print('Outer cone shape not defined')
        self.mesh = gcad.cone(mass > 0, P, 0, R, phic, phih)


class Sphere(Shape):
    """
    Sphere with multipoles and cadquery
    """
    def __init__(self, lmax, inner, mass, R, center):
        Shape.__init__(self, lmax, inner)
        self.mass = mass
        if self.inner:
            self.qlm = qlm.sphere(lmax, mass, R)
            self.translate(center, True)
        else:
            dens = mass/(4*np.pi*R**3/3)
            bqlm.sphere(lmax, dens, R, center[0], center[1], center[2])
        self.mesh = gcad.sphere(mass > 0, R)


class NGon(Shape):
    """
    N-gon with multipoles and cadquery
    """
    def __init__(self, N, lmax, inner, mass, H, a, phic, Ns):
        Shape.__init__(self, lmax, inner)
        self.mass = mass
        if self.inner:
            self.qlm = qlm.ngon_prism(lmax, mass, H, a, phic, Ns)
        else:
            print('Outer N-gon shape not defined.')
        self.mesh = gcad.ngon_prism(mass > 0, H, a, phic, Ns)


class Tetrahedron(Shape):
    """
    Tetrahedron with multipoles and cadquery
    """
    def __init__(self, lmax, inner, mass, x, y1, y2, z):
        Shape.__init__(self, lmax, inner)
        self.mass = mass
        if self.inner:
            self.qlm = qlm.tetrahedron2(lmax, mass, x, y1, y2, z)
        else:
            print('Outer Tetrahedron shape not defined.')
        self.mesh = gcad.tetrahedron2(mass > 0, x, y1, y2, z)


class Pyramid(Shape):
    """
    Pyramid with multipoles and cadquery
    """
    def __init__(self, lmax, inner, mass, x, y, z):
        Shape.__init__(self, lmax, inner)
        self.mass = mass
        if self.inner:
            self.qlm = qlm.pyramid(lmax, mass, x, y, z)
        else:
            print('Outer pyramid shape not defined.')
        self.mesh = gcad.pyramid(mass > 0, x, y, z)


class OuterCone(Shape):
    """
    OuterCone with multipoles and cadquery
    """
    def __init__(self, lmax, inner, mass, H, IR, OR, phih):
        Shape.__init__(self, lmax, inner)
        self.mass = mass
        if self.inner:
            print('Inner cone shape should come from Cone.')
        else:
            dens = mass/(2*phih*H*(OR**2-IR**2)/3)
            bqlmn.outer_cone(lmax, dens, H, IR, OR, phih)
        self.mesh = gcad.outercone(mass, H, IR, OR, phih)


class Cylhole(Shape):
    """
    Cylhole with multipoles and pointgravity
    """
    def __init__(self, lmax, inner, mass, r, R):
        Shape.__init__(self, lmax, inner)
        self.mass = mass
        if self.inner:
            dens = 1
            self.qlm = qlmn.steinmetz(lmax, dens, r, R)
            self.qlm *= mass/np.real(qlmn[0, lmax])*np.sqrt(4*np.pi)
        else:
            print('Outer cylindrical hole shape not defined.')
        self.mesh = gcad.cylhole(mass > 0, r, R)


class Platehole(Shape):
    """
    Platehole with multipoles and pointgravity
    """
    def __init__(self, N, lmax, inner, mass, t, r, theta):
        Shape.__init__(self, N, lmax, inner)
        self.mass = mass
        if self.inner:
            dens = 1
            self.qlm = qlmn.platehole(lmax, dens, t, r, theta)
            self.qlm *= mass/np.real(qlmn[0, lmax])*np.sqrt(4*np.pi)
        else:
            print('Outer plate hole shape not defined.')
        self.mesh = gcad.platehole(mass > 0, t, r, theta)
