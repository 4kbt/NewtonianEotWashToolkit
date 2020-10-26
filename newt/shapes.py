# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 20:28:18 2020

@author: jgl6
"""
import numpy as np
import newt.qlm as qlm
import newt.qlmNum as qlmn
import newt.bigQlm as bqlm
import newt.bigQlmNum as bqlmn
import newt.glibShapes as gshp
import newt.glib as glb
import newt.translations as trs
import newt.rotations as rot


class Shape:
    """
    A Shape object is an inheritable class for shape primitives with abilities
    to both calculate multipole moments and crudely visualize using
    PointGravity.
    """
    def __init__(self, N, lmax, inner=True):
        self.inner = bool(inner)
        self.lmax = lmax
        self.qlm = np.zeros([lmax+1, 2*lmax+1], dtype=complex)
        self.pointmass = np.zeros([N, 4])
        self.com = np.zeros(3)
        self.mass = 0

    def display_shape(self):
        pmp = self.pointmass[np.where(self.pointmass[:, 0] > 0)[0]]
        pmn = self.pointmass[np.where(self.pointmass[:, 0] < 0)[0]]
        glb.display_points(pmp, pmn)

    def rotate(self, alpha, beta, gamma):
        self.qlm = rot.rotate_qlm(self.qlm, alpha, beta, gamma)
        self.pointmass = glb.rotate_point_array(self.pointmass, gamma,
                                                [0, 0, 1])
        self.pointmass = glb.rotate_point_array(self.pointmass, beta,
                                                [0, 1, 0])
        self.pointmass = glb.rotate_point_array(self.pointmass, alpha,
                                                [0, 0, 1])
        # Rotate center of mass using point-gravity tools
        rotcom = np.concatenate([[self.mass], self.com])
        rotcom = glb.rotate_point_array(rotcom, gamma, [0, 0, 1])
        rotcom = glb.rotate_point_array(rotcom, beta, [0, 1, 0])
        rotcom = glb.rotate_point_array(rotcom, alpha, [0, 0, 1])
        self.com = rotcom[1:4]

    def translate(self, tvec, inner):
        self.pointmass = glb.translate_point_array(self.pointmass, tvec)
        self.com += tvec
        if self.inner and inner:
            self.qlm = trs.translate_qlm(self.qlm, tvec)
        elif self.inner and not inner:
            self.inner = False
            self.qlm = trs.translate_q2Q(self.qlm, tvec)
        else:
            self.qlm = trs.translate_Qlmb(self.qlm, tvec)

    def add(self, shape2):
        if (self.inner is shape2.inner) and (self.lmax == shape2.lmax):
            m1, m2 = self.mass, shape2.mass
            self.com = (m1*self.com + m2*shape2.com)/(m1+m2)
            self.mass = m1 + m2
            self.qlm += shape2.qlm
            self.pointmass = np.concatenate([self.pointmass, shape2.pointmass])
        else:
            raise TypeError('Trying to combine inner and outer moments')


class Annulus(Shape):
    """
    Annulus with multipoles and pointgravity
    """
    def __init__(self, N, lmax, inner, mass, H, Ri, Ro, phic, phih):
        Shape.__init__(self, N, lmax, inner)
        self.mass = mass
        if self.inner:
            self.qlm = qlm.annulus(lmax, mass, H, Ri, Ro, phic, phih)
        else:
            dens = mass/(2*phih*(Ro**2 - Ri**2)*H)
            self.qlm = bqlm.annulus(lmax, dens, H/2, Ri, Ro, phic, phih)
            self.qlm += bqlm.annulus(lmax, dens, -H/2, Ri, Ro, phic, phih)
        self.pointmass = gshp.wedge(mass, Ri, Ro, H, phih, N, N)
        self.pointmass = glb.rotate_point_array(self.pointmass, phic,
                                                [0, 0, 1])


class RectPrism(Shape):
    """
    Rectangular prism with multipoles and pointgravity
    """
    def __init__(self, N, lmax, inner, mass, H, a, b, phic):
        Shape.__init__(self, N, lmax, inner)
        self.mass = mass
        if self.inner:
            self.qlm = qlm.rect_prism(lmax, mass, H, a, b, phic)
        else:
            print('Outer rectangular prism shape not defined')
        self.pointmass = gshp.wedge(mass, a, b, H, N, N, N)
        self.pointmass = glb.rotate_point_array(self.pointmass, phic,
                                                [0, 0, 1])


class TriPrism(Shape):
    """
    Triangular prism with multipoles and pointgravity
    """
    def __init__(self, N, lmax, inner, mass, H, d, y1, y2):
        Shape.__init__(self, N, lmax, inner)
        self.mass = mass
        if self.inner:
            self.qlm = qlm.tri_prism(lmax, mass, H, d, y1, y2)
        else:
            print('Outer triangular prism shape not defined')
        self.pointmass = gshp.tri_prism(mass, d, y1, y2, H, N, N)


class Cone(Shape):
    """
    Cone with multipoles and pointgravity
    """
    def __init__(self, N, lmax, inner, mass, P, R, phic, phih):
        Shape.__init__(self, N, lmax, inner)
        self.mass = mass
        if self.inner:
            self.qlm = qlm.cone(lmax, mass, P, R, phic, phih)
        else:
            print('Outer cone shape not defined. See OuterCone')
        self.pointmass = gshp.cone(mass, R, P, phih, N, N)
        self.pointmass = glb.rotate_point_array(self.pointmass, phic,
                                                [0, 0, 1])


class Sphere(Shape):
    """
    Sphere with multipoles and pointgravity
    """
    def __init__(self, N, lmax, inner, mass, R):
        Shape.__init__(self, N, lmax, inner)
        self.mass = mass
        if self.inner:
            self.qlm = qlm.sphere(lmax, mass, R)
        else:
            print('Outer sphere shape not defined.')
        self.pointmass = gshp.sphere(mass, R, N)


class NGon(Shape):
    """
    N-gon with multipoles and pointgravity
    """
    def __init__(self, N, lmax, inner, mass, H, a, phic, Ns):
        Shape.__init__(self, N, lmax, inner)
        self.mass = mass
        if self.inner:
            self.qlm = qlm.ngon_prism(lmax, mass, H, a, phic, Ns)
        else:
            print('Outer Tetrahedron shape not defined.')
        self.pointmass = gshp.ngon_prism(mass, H, a, Ns, N, N)
        self.pointmass = glb.rotate_point_array(self.pointmass, phic,
                                                [0, 0, 1])


class Tetrahedron(Shape):
    """
    Tetrahedron with multipoles and pointgravity
    """
    def __init__(self, N, lmax, inner, mass, x, y1, y2, z):
        Shape.__init__(self, N, lmax, inner)
        self.mass = mass
        if self.inner:
            self.qlm = qlm.tetrahedron2(lmax, mass, x, y1, y2, z)
        else:
            print('Outer Tetrahedron shape not defined.')
        self.pointmass = gshp.tetrahedron(mass, x, y1, y2, z, N, N, N)


class Pyramid(Shape):
    """
    Pyramid with multipoles and pointgravity
    """
    def __init__(self, N, lmax, inner, mass, x, y, z):
        Shape.__init__(self, N, lmax, inner)
        self.mass = mass
        if self.inner:
            self.qlm = qlm.pyramid(lmax, mass, x, y, z)
        else:
            print('Outer pyramid shape not defined.')
        self.pointmass = gshp.pyramid(mass, x, y, z, N, N, N)


class OuterCone(Shape):
    """
    OuterCone with multipoles and pointgravity
    """
    def __init__(self, N, lmax, inner, mass, IR, OR, H, phih):
        Shape.__init__(self, N, lmax, inner)
        self.mass = mass
        if self.inner:
            print('Inner cone shape should come from Cone.')
        else:
            dens = mass/(2*phih*H*(OR**2-IR**2)/3)
            bqlmn.outer_cone(lmax, dens, H, IR, OR, phih)
        self.pointmass = gshp.outer_cone(mass, IR, OR, H, phih, N, N)


class Cylhole(Shape):
    """
    Cylhole with multipoles and pointgravity
    """
    def __init__(self, N, lmax, inner, mass, r, R):
        Shape.__init__(self, N, lmax, inner)
        self.mass = mass
        if self.inner:
            dens = 1
            self.qlm = qlmn.steinmetz(lmax, dens, r, R)
            self.qlm *= mass/np.real(qlmn[0, lmax])*np.sqrt(4*np.pi)
        else:
            print('Outer cylindrical hole shape not defined.')
        self.pointmass = gshp.cylhole(mass, r, R, N, N)


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
            print('Outer cylindrical hole shape not defined.')
        self.pointmass = gshp.platehole(mass, t, r, theta, N, N, N)
