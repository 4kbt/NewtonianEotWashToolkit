# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 13:06:38 2020

@author: John Greendeer Lee
"""
import numpy as np
import newt.qlm as qlm
import newt.qlmACH as qlmA
import newt.qlmNum as qlmN
import newt.bigQlm as bqlm
import newt.bigQlmNum as bqlmn
import newt.rotations as rot
import newt.translations as trs


def read_gsq(filename, filepath='C:\\mom\\'):
    """
    Read an inner multipole moment file generated with MULTIN. Takes an
    optional filepath assumed to be 'C:\\mom\\' as typical for MULTIN.

    Inputs
    ------
    filename : str
        Name of .gsq file (with or without extension)
    filepath : str, optional
        Directory containing .gsq file, assumed 'C:\\mom\\' in MULTIN

    Returns
    -------
    qlm : ndarray, complex
        Complex (LMax+1)x(2LMax + 1) array of inner multipole coefficients
    """
    if not filename.endswith('.gsq'):
        filename += '.gsq'
    with open(filepath+filename) as f:
        lines = f.readlines()
    title = lines[0]
    print(title)
    LMax = int(lines[1].split()[0])
    l, m = 0, 0
    qlm = np.zeros([LMax+1, 2*LMax+1], dtype='complex')
    # moments in cgs so to convert to mks multiply by 1e-3*(.01)**l
    for line in lines[9:]:
        lsplit = line.split()
        lr, li = float(lsplit[0]), float(lsplit[1])
        qlm[l, LMax+m] = (lr+1j*li)*1e-3*(0.01)**l
        m += 1
        if m > l:
            l += 1
            m = 0

    # Moments always satisfy Q(l, -m) = (-1)^m Q(l, m)*
    ms = np.arange(-LMax, LMax+1)
    mfac = (-1)**(np.abs(ms))
    qlm += np.conj(np.fliplr(qlm))*mfac
    qlm[:, LMax] /= 2
    return qlm


def read_gbq(filename, filepath='C:\\mom\\'):
    """
    Read an outer multipole moment file generated with MULTIN. Takes an
    optional filepath assumed to be 'C:\\mom\\' as typical for MULTIN.

    Inputs
    ------
    filename : str
        Name of .gbq file (with or without extension)
    filepath : str, optional
        Directory containing .gbq file, assumed 'C:\\mom\\' in MULTIN

    Returns
    -------
    Qlmb : ndarray, complex
        Complex (LMax+1)x(2LMax + 1) array of outer multipole coefficients
    """
    if not filename.endswith('.gbq'):
        filename += '.gbq'
    with open(filepath+filename) as f:
        lines = f.readlines()
    title = lines[0]
    print(title)
    LMax = int(lines[1].split()[0])
    l, m = 0, 0
    Qlmb = np.zeros([LMax+1, 2*LMax+1], dtype='complex')
    for line in lines[9:]:
        lsplit = line.split()
        lr, li = float(lsplit[0]), float(lsplit[1])
        Qlmb[l, LMax+m] = lr+1j*li
        m += 1
        if m > l:
            l += 1
            m = 0

    # Moments always satisfy Q(l, -m) = (-1)^m Q(l, m)*
    ms = np.arange(-LMax, LMax+1)
    mfac = (-1)**(np.abs(ms))
    Qlmb += np.conj(np.fliplr(Qlmb))*mfac
    Qlmb[:, LMax] /= 2
    return Qlmb


def read_mpc(LMax, filename, filepath='C:\\mpc\\'):
    """
    Attempts to open .mpc files in the same manner as MULTIN by having a
    'working register' and a 'total register'. That is, there are two sets of
    moments stored in memory. One is the sum of all the previous moments
    (total) and the other is the current shape (working) being manipulated. The
    working shape can be rotated, translated, and added to total sequentially
    many times. This function does not allow the use of 'recall', 'store',
    'rescale', or 'save' statements currently. Takes an optional filepath which
    is assumed to be the standard filepath for MULTIN, 'C:\\mpc\\'.

    Inputs
    ------
    LMax : int
        Highest degree inner multipole moments to compute
    filename : str
        Name of .mpc file (with or without extension)
    filepath : str, optional
        Directory containing .mpc file, assumed 'C:\\mpc\\' in MULTIN

    Returns
    -------
    qlmTot : ndarray, complex
        Complex (LMax+1)x(2LMax + 1) array of inner multipole coefficients
    """
    if not filename.endswith('.mpc'):
        filename += '.mpc'
    with open(filepath+filename) as f:
        lines = f.readlines()
    title = lines[0]
    print(title)
    # We don't actually use n integration in this
    nsteps = 25
    # Assume units are centimeters unless told otherwise
    fac = 1e-2
    nlines = len(lines[1:])
    qlmTot = np.zeros([LMax+1, 2*LMax+1], dtype='complex')
    qlmWrk = np.zeros([LMax+1, 2*LMax+1], dtype='complex')
    k = 0
    while k < nlines:
        line = lines[1+k]
        # Get rid of stuff after a comment
        line = line.split('%')[0]
        if 'create' in line:
            print(line)
            shape = line.split('create')[1].split()[0]
            if shape == 'cylinder':
                line2 = [float(val) for val in lines[2+k].split(',')]
                Ri, Ro, H, phi0, phi1 = line2
                Ri, Ro, H = Ri*fac, Ro*fac, H*fac
                phic = (phi1+phi0)*np.pi/180/2
                phih = (phi1-phi0)*np.pi/180/2
                dens = float(lines[3+k].split(',')[0])*1000
                mass = dens*phih*H*(Ro**2-Ri**2)
                print(dens, line2)
                qlmWrk = qlm.annulus(LMax, mass, H, Ri, Ro, phic, phih)
                pos = np.array(lines[4+k].split(','), dtype=float)*fac
                k += 3
                if (pos == 0).all() and ('add' in lines[2+k]):
                    k += 1
                    print('added ', shape)
                    qlmTot += qlmWrk
                else:
                    print('translated ', shape)
                    qlmWrk = trs.translate_qlm(qlmWrk, pos)
            elif shape == 'sphere':
                line2 = [float(val)*fac for val in lines[2+k].split(',')]
                r = line2[0]
                dens = float(lines[3+k].split(',')[0])*1000
                mass = dens*4/3*np.pi*r**3
                qlmWrk = qlm.sphere(LMax, mass, r)
                pos = np.array(lines[4+k].split(','), dtype=float)*fac
                k += 3
                if (pos == 0).all() and ('add' in lines[2+k]):
                    k += 1
                    print('added ', shape)
                    qlmTot += qlmWrk
                else:
                    print('translated ', shape)
                    qlmWrk = trs.translate_qlm(qlmWrk, pos)
            elif shape == 'cone':
                line2 = [float(val)*fac for val in lines[2+k].split(',')]
                LR, UR, H = line2
                dens = float(lines[3+k].split(',')[0])*1000
                mass = dens*np.pi*H*LR**2/3
                print(LR, UR, H, dens, mass)
                qlmWrk = qlm.cone(LMax, mass, H, LR, 0, np.pi)
                pos = np.array(lines[4+k].split(','), dtype=float)*fac
                k += 3
                if (pos == 0).all() and ('add' in lines[2+k]):
                    k += 1
                    print('added ', shape)
                    qlmTot += qlmWrk
                else:
                    print('translated ', shape)
                    qlmWrk = trs.translate_qlm(qlmWrk, pos)
            elif shape == 'triangle':
                line2 = [float(val)*fac for val in lines[2+k].split(',')]
                d, y1, y2, t = line2
                dens = float(lines[3+k].split(',')[0])*1000
                mass = dens*t*d*abs(y2-y1)/2
                qlmWrk = qlm.tri_prism(LMax, mass, t, d, y1, y2)
                pos = np.array(lines[4+k].split(','), dtype=float)*fac
                k += 3
                if (pos == 0).all() and ('add' in lines[2+k]):
                    k += 1
                    print('added ', shape)
                    qlmTot += qlmWrk
                else:
                    print('translated ', shape)
                    qlmWrk = trs.translate_qlm(qlmWrk, pos)
            elif shape == 'trapezoid':
                line2 = [float(val)*fac for val in lines[2+k].split(',')]
                w1, w2, h, t = line2
                if w2 < w1:
                    w3, w4 = w2, w1
                    flip = True
                else:
                    w3, w4 = w1, w2
                    flip = False
                dens = float(lines[3+k].split(',')[0])*1000
                y1, y2 = w3/2, w4/2
                hs = w3*h/(w4-w3)
                hb = h + hs
                massTrib = dens*t*w4*hb/2
                massTris = dens*t*w3*hs/2
                print(hs, hb, w3, w4)
                qlmWrk = qlm.tri_iso_prism(LMax, massTrib, t, w4, hb, 0)
                qlmWrk -= qlm.tri_iso_prism(LMax, massTris, t, w3, hs, 0)
                qlmWrk = trs.translate_qlm(qlmWrk, [-hs, 0, 0])
                if flip:
                    qlmWrk = trs.translate_qlm(qlmWrk, [-(hb-hs), 0, 0])
                    qlmWrk = rot.rotate_qlm(qlmWrk, 0, 0, np.pi)
                pos = np.array(lines[4+k].split(','), dtype=float)*fac
                k += 3
                if (pos == 0).all() and ('add' in lines[2+k]):
                    k += 1
                    print('added ', shape)
                    qlmTot += qlmWrk
                else:
                    print('translated ', shape)
                    qlmWrk = trs.translate_qlm(qlmWrk, pos)
            elif shape == 'partcylinder':
                line2 = [float(val)*fac for val in lines[2+k].split(',')]
                r, d, h = line2
                dens = float(lines[3+k].split(',')[0])*1000
                phih = np.arccos(d/r)
                massCyl = dens*h*phih*r**2
                massTri = dens*h*d*r*np.sin(phih)
                qlmWrk = qlm.annulus(LMax, massCyl, h, 0, r, 0, phih)
                qlmWrk -= qlm.tri_iso_prism2(LMax, massTri, h, r, 0, phih)
                pos = np.array(lines[4+k].split(','), dtype=float)*fac
                k += 3
                if (pos == 0).all() and ('add' in lines[2+k]):
                    k += 1
                    print('added ', shape)
                    qlmTot += qlmWrk
                else:
                    print('translated ', shape)
                    qlmWrk = trs.translate_qlm(qlmWrk, pos)
            elif shape == 'tetrahedron':
                line2 = [float(val)*fac for val in lines[2+k].split(',')]
                x, y, z = line2
                dens = float(lines[3+k].split(',')[0])*1000
                mass = dens*x*y*z/6
                qlmWrk = qlm.tetrahedron(LMax, mass, x, y, z)
                pos = np.array(lines[4+k].split(','), dtype=float)*fac
                k += 3
                if (pos == 0).all() and ('add' in lines[2+k]):
                    k += 1
                    print('added ', shape)
                    qlmTot += qlmWrk
                else:
                    print('translated ', shape)
                    qlmWrk = trs.translate_qlm(qlmWrk, pos)
            elif shape == 'platehole':
                line2 = [float(val) for val in lines[2+k].split(',')]
                r, t, theta = line2
                t, r = t*fac, r*fac
                theta *= np.pi/180
                dens = float(lines[3+k].split(',')[0])*1000
                qlmWrk = qlmA.platehole(LMax, dens, t, r, theta)
                # qlmWrk = qlmN.platehole(LMax, dens, t, r, theta)
                # qlmWrk = rot.rotate_qlm(qlmWrk, 0, theta, 0)
                pos = np.array(lines[4+k].split(','), dtype=float)*fac
                k += 3
                if (pos == 0).all() and ('add' in lines[2+k]):
                    k += 1
                    print('added ', shape)
                    qlmTot += qlmWrk
                else:
                    print('translated ', shape)
                    qlmWrk = trs.translate_qlm(qlmWrk, pos)
            elif shape == 'cylhole':
                line2 = [float(val)*fac for val in lines[2+k].split(',')]
                r, R = line2
                dens = float(lines[3+k].split(',')[0])*1000
                qlmWrk = qlmN.steinmetz(LMax, dens, r, R)
                qlmWrk = rot.rotate_qlm(qlmWrk, np.pi/2, 0, 0)
                pos = np.array(lines[4+k].split(','), dtype=float)*fac
                k += 3
                if (pos == 0).all() and ('add' in lines[2+k]):
                    k += 1
                    print('added ', shape)
                    qlmTot += qlmWrk
                else:
                    print('translated ', shape)
                    qlmWrk = trs.translate_qlm(qlmWrk, pos)
            elif shape == 'pyramid':
                line2 = [float(val)*fac for val in lines[2+k].split(',')]
                x, y, z = line2
                dens = float(lines[3+k].split(',')[0])*1000
                mass = dens*x*y*z/3
                qlmWrk = qlm.pyramid(LMax, mass, x/2, y/2, z)
                pos = np.array(lines[4+k].split(','), dtype=float)*fac
                k += 3
                if (pos == 0).all() and ('add' in lines[2+k]):
                    k += 1
                    print('added ', shape)
                    qlmTot += qlmWrk
                else:
                    print('translated ', shape)
                    qlmWrk = trs.translate_qlm(qlmWrk, pos)
            elif shape == 'rectangle':
                line2 = [float(val)*fac for val in lines[2+k].split(',')]
                x, y, z = line2
                dens = float(lines[3+k].split(',')[0])*1000
                mass = dens*x*y*z
                qlmWrk = qlm.rect_prism(LMax, mass, z, x, y, 0)
                pos = np.array(lines[4+k].split(','), dtype=float)*fac
                k += 3
                if (pos == 0).all() and ('add' in lines[2+k]):
                    k += 1
                    print('added ', shape)
                    qlmTot += qlmWrk
                else:
                    print('translated ', shape)
                    qlmWrk = trs.translate_qlm(qlmWrk, pos)
            else:
                print(shape, ' does not have a known set of moments')
                for n in range(nlines-k):
                    line2 = lines[1+k+1]
                    if ('create' not in line2) and ('end' not in line2):
                        k += 1
                    else:
                        break
        elif 'rotate' in line:
            dphi = np.array(line.split('rotate')[1].split(','), dtype=float)
            # Convert to radians
            dphi *= np.pi/180
            print('rotated ', shape)
            qlmWrk = rot.rotate_qlm(qlmWrk, dphi[0], dphi[1], dphi[2])
        elif 'translate' in line:
            print('translated ', shape)
            dr = np.array(line.split('translate')[1].split(','), dtype=float)
            dr *= fac
            qlmWrk = trs.translate_qlm(qlmWrk, dr)
        elif 'add' in line:
            print('added ', shape)
            qlmTot += qlmWrk
        elif 'cmin' in line:
            print('parameters in centimeters')
            fac = 1e-2
        elif 'inchin' in line:
            print('parameters in inches')
            fac = 25.4e-3
        elif 'nsteps' in line:
            nsteps = line.split()[-1]
        elif 'load' in line:
            fname = line.split('load ')[1]
            fname = fname.strip()
            print(fname)
            qlmLoad = read_gsq(fname, filepath.replace('mpc', 'mom'))
            if np.shape(qlmLoad) != np.shape(qlmWrk):
                raise TypeError('Loaded part '+fname+' has wrong LMax')
            qlmWrk = qlmLoad
            shape = fname
        elif 'zeroqlm' in line:
            qlmTot *= 0
        elif 'gettotal' in line:
            qlmWrk = np.copy(qlmTot)
            shape = 'Total'
        elif 'puttotal' in line:
            qlmTot = np.copy(qlmWrk)
            shape = 'Working'
        elif 'end' in line:
            print('end')
            break
        k += 1
    return qlmTot
