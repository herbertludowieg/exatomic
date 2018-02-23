# -*- coding: utf-8 -*-
# Copyright (c) 2015-2017, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Overlap computation
######################
Utilities for computing the overlap between gaussian type functions.
"""

import numpy as np
from numba import jit, prange
from .numerical import fac, fac2, dfac21, sdist, choose


@jit(nopython=True, cache=True)
def _fj(j, l, m, a, b):
    tot = 0.
    for k in prange(max(0, j - m), min(j, l) + 1):
        tot += (choose(l, k) *
                choose(m, int(j - k)) *
                a ** (l - k) *
                b ** (m + k - j))
    return tot

@jit(nopython=True, cache=True)
def _nin(l, m, pa, pb, gamma, N):
    ltot = l + m
    if not ltot: return N
    tot = 0.
    for j in prange(int(ltot // 2 + 1)):
        tot += _fj(2 * j, l, m, pa, pb) * dfac21(j) / (2 * gamma) ** j
    return tot * N

@jit(nopython=True, cache=True)
def _new_gaussian_product(a, b, ax, ay, az, bx, by, bz):
    p = a + b
    mu = a * b / p
    px = (a * ax + b * bx) / p
    py = (a * ay + b * by) / p
    pz = (a * az + b * bz) / p
    pax = px - ax
    pay = py - ay
    paz = pz - az
    pbx = px - bx
    pby = py - by
    pbz = pz - bz
    ab2 = sdist(ax, ay, az, bx, by, bz)
    return np.sqrt(np.pi / p), p, mu, ab2, pax, pay, paz, pbx, pby, pbz

@jit(nopython=True, cache=True)
def _gaussian_product(a1, a2, ax, ay, az, bx, by, bz):
    gamma = a1 + a2
    N = np.sqrt(np.pi / gamma)
    px = (a1 * ax + a2 * bx) / gamma
    py = (a1 * ay + a2 * by) / gamma
    pz = (a1 * az + a2 * bz) / gamma
    pax = px - ax
    pay = py - ay
    paz = pz - az
    pbx = px - bx
    pby = py - by
    pbz = pz - bz
    ab2 = sdist(ax, ay, az, bx, by, bz)
    return N, gamma, ab2, pax, pay, paz, pbx, pby, pbz

@jit(nopython=True, cache=True)
def _overlap_product(N, gamma, ab2, pax, pay, paz, pbx, pby, pbz,
                     a1, a2, l1, m1, n1, l2, m2, n2):
    return (np.exp(-a1 * a2 * ab2 / gamma) *
            _nin(l1, l2, pax, pbx, gamma, N) *
            _nin(m1, m2, pay, pby, gamma, N) *
            _nin(n1, n2, paz, pbz, gamma, N))

@jit(nopython=True, cache=True)
def _overlap_1c(a1, a2, l1, m1, n1, l2, m2, n2):
    """Compute overlap between gaussian functions on the same center."""
    ll = l1 + l2
    mm = m1 + m2
    nn = n1 + n2
    if ll % 2 or mm % 2 or nn % 2: return 0
    ltot = ll // 2 + mm // 2 + nn // 2
    numer = np.pi ** (1.5) * fac2(ll - 1) * fac2(mm - 1) * fac2(nn - 1)
    denom = (2 ** ltot) * (a1 + a2) ** (ltot + 1.5)
    return numer / denom

@jit(nopython=True, cache=True)
def _overlap_2c(a1, a2, ax, ay, az, bx, by, bz, l1, m1, n1, l2, m2, n2):
    return _overlap_product(*_gaussian_product(a1, a2, ax, ay, az, bx, by, bz),
                            a1, a2, l1, m1, n1, l2, m2, n2)

@jit(nopython=True, cache=True)
def _kinetic_1c(a1, a2, l1, m1, n1, l2, m2, n2):
    """Compute kinetic energy of gaussian functions on the same center."""
    t =  4 * a1 * a2 * _overlap_1c(a1, a2, l1 + 1, m1, n1, l2 + 1, m2, n2)
    t += 4 * a1 * a2 * _overlap_1c(a1, a2, l1, m1 + 1, n1, l2, m2 + 1, n2)
    t += 4 * a1 * a2 * _overlap_1c(a1, a2, l1, m1, n1 + 1, l2, m2, n2 + 1)
    if l1 and l2:
        t += l1 * l2 * _overlap_1c(a1, a2, l1 - 1, m1, n1, l2 - 1, m2, n2)
    if m1 and m2:
        t += m1 * m2 * _overlap_1c(a1, a2, l1, m1 - 1, n1, l2, m2 - 1, n2)
    if n1 and n2:
        t += n1 * n2 * _overlap_1c(a1, a2, l1, m1, n1 - 1, l2, m2, n2 - 1)
    if l1: t -= 2 * a2 * l1 * _overlap_1c(a1, a2, l1 - 1, m1, n1, l2 + 1, m2, n2)
    if l2: t -= 2 * a1 * l2 * _overlap_1c(a1, a2, l1 + 1, m1, n1, l2 - 1, m2, n2)
    if m1: t -= 2 * a2 * m1 * _overlap_1c(a1, a2, l1, m1 - 1, n1, l2, l2 + 1, n2)
    if m2: t -= 2 * a1 * m2 * _overlap_1c(a1, a2, l1, m1 + 1, n1, l2, m2 - 1, n2)
    if n1: t -= 2 * a2 * n1 * _overlap_1c(a1, a2, l1, m1, n1 - 1, l2, l2, n2 + 1)
    if n2: t -= 2 * a1 * n2 * _overlap_1c(a1, a2, l1, m1, n1 + 1, l2, m2, n2 - 1)
    return t / 2

@jit(nopython=True, cache=True)
def _kinetic_2c(a1, a2, ax, ay, az, bx, by, bz, l1, m1, n1, l2, m2, n2):
    """Compute the kinetic energy of two gaussian functions on different centers."""
    args = _gaussian_product(a1, a2, ax, ay, az, bx, by, bz)
    t =  4 * a1 * a2 * _overlap_product(*args, a1, a2, l1 - 1, m1, n1, l2 - 1, m2, n2)
    t += 4 * a1 * a2 * _overlap_product(*args, a1, a2, l1, m1 - 1, n1, l2, m2 - 2, n2)
    t += 4 * a1 * a2 * _overlap_product(*args, a1, a2, l1, m1, n1 - 1, l2, m2, n2 - 1)
    if l1 and l2:
        t += l1 * l2 * _overlap_product(*args, a1, a2, l1 - 1, m1, n1, l2 - 1, m2, n2)
    if m1 and m2:
        t += l1 * l2 * _overlap_product(*args, a1, a2, l1, m1 - 1, n1, l2, m2 - 1, n2)
    if n1 and n2:
        t += l1 * l2 * _overlap_product(*args, a1, a2, l1, m1, n1 - 1, l2, m2, n2 - 1)
    if l1: t -=  2 * a2 * l1 * _overlap_product(*args, a1, a2, l1 - 1, m1, n1, l2 + 1, m2, n2)
    if l2: t -=  2 * a1 * l2 * _overlap_product(*args, a1, a2, l1 + 1, m1, n1, l2 - 1, m2, n2)
    if m1: t -=  2 * a2 * m1 * _overlap_product(*args, a1, a2, l1, m1 - 1, n1, l2, m2 + 1, n2)
    if m2: t -=  2 * a1 * m2 * _overlap_product(*args, a1, a2, l1, m1 + 1, n1, l2, m2 - 1, n2)
    if n1: t -=  2 * a2 * n1 * _overlap_product(*args, a1, a2, l1, m1, n1 - 1, l2, m2, n2 + 1)
    if n2: t -=  2 * a1 * n2 * _overlap_product(*args, a1, a2, l1, m1, n1 + 1, l2, m2, n2 - 1)
    return t / 2

# @njit
# def _iter_pairs(c, x, y, z, l, m, n, a):
#     nprim = len(x)
#     for i in prange(len(x)):
#         for j in prange(i + 1):
#             if c[i] == c[j]:
#                 yield a[i], a[j], l[i], m[i], n[i], l[j], m[j], n[j]
#             else:
#                 yield a[i], a[j], x[i], y[i], z[i], x[j], y[j], z[j], \
#                                   l[i], m[i], n[i], l[j], m[j], n[j]

@jit(nopython=True, cache=True)
def _primitive_overlap(c, x, y, z, l, m, n, a):
    """Compute a triangular portion of the primitive overlap matrix."""
    nprim, cnt = len(x), 0
    ndim = nprim * (nprim + 1) // 2
    ovl = np.empty(ndim, dtype=np.float64)
    # for args in _iter_pairs(c, x, y, z, l, m, n, a):
    for i in prange(nprim):
        for j in prange(i + 1):
            if c[i] == c[j]:
                ovl[cnt] = _overlap_1c(a[i], a[j],
                                       l[i], m[i], n[i], l[j], m[j], n[j])
            else:
                ovl[cnt] = _overlap_2c(a[i], a[j],
                                       x[i], y[i], z[i], x[j], y[j], z[j],
                                       l[i], m[i], n[i], l[j], m[j], n[j])
            cnt += 1
    return ovl


@jit(nopython=True, cache=True)
def _primitive_kinetic(c, x, y, z, l, m, n, a):
    """Compute a triangular portion of the primitive overlap matrix."""
    nprim, cnt = len(x), 0
    ndim = nprim * (nprim + 1) // 2
    kin = np.empty(ndim, dtype=np.float64)
    for i in prange(nprim):
        for j in prange(i + 1):
            if c[i] == c[j]:
                kin[cnt] = _kinetic_1c(a[i], a[j],
                                       l[i], m[i], n[i], l[j], m[j], n[j])
            else:
                kin[cnt] = _kinetic_2c(a[i], a[j],
                                       x[i], y[i], z[i], x[j], y[j], z[j],
                                       l[i], m[i], z[i], l[j], m[j], n[j])
            cnt += 1
    return kin


@jit(nopython=True, cache=True)
def _primitive_nucattr(c, x, y, z, l, m, n, a):
    """Compute a triangular portion of the primitive overlap matrix."""
    nprim, cnt = len(x), 0
    ndim = nprim * (nprim + 1) // 2
    nuc = np.empty(ndim, dtype=np.float64)
    for i in prange(nprim):
        for j in prange(i + 1):
            if c[i] == c[j]:
                nuc[cnt] = _nucattr_1c(a[i], a[j],
                                       l[i], m[i], n[i], l[j], m[j], n[j])
            else:
                nuc[cnt] = _nucattr_2c(a[i], a[j],
                                       x[i], y[i], z[i], x[j], y[j], z[j],
                                       l[i], m[i], z[i], l[j], m[j], n[j])
            cnt += 1
    return nuc
# @jit(nopython=True, cache=True)
# def _nucattr_1c(a1, a2, l1, m1, n1, l2, m2, n2):
#     pass
#                 # gamma = a[i] + a[j]
#                 # xp = (a[i] * x[i] + a[j] * x[j]) / gamma
#                 # yp = (a[i] * y[i] + a[j] * y[j]) / gamma
#                 # zp = (a[i] * z[i] + a[j] * z[j]) / gamma
#                 # pax = xp - x[i]
#                 # pay = yp - y[i]
#                 # paz = zp - z[i]
#                 # pbx = xp - x[j]
#                 # pby = yp - y[j]
#
# @jit(nopython=True, cache=True)
# def _nucattr_2c(ax, ay, az, bx, by, bz, pax, pay, paz, pbx, pby, pbz,
#                 gamma, a1, a2, l1, m1, n1, l2, m2, n2):
#     # pre = 2 * np.pi / gamma * np.exp(-a1 * a2 * sdist())
#     ab2 = sdist(ax, ay, az, bx, by, bz)
#     pg12 = 2 * np.pi / gamma
#     cx = (ax + bx) / 2
#     cy = (ay + by) / 2
#     cz = (az + bz) / 2
#     xix = _nin(l1, l2, pax, pbx, gamma, pg12)
#     yiy = _nin(m1, m2, pay, pby, gamma, pg12)
#     ziz = _nin(n1, n2, paz, pbz, gamma, pg12)
#     exp = a1 * a2 * ab2 / gamma
#
#     # return pg12 *

@jit(nopython=True, cache=True)
def _old_nin(l, m, pa, pb, gamma, N):
    ltot = l + m
    if not ltot: return N
    if ltot % 2: ltot -= 1
    tot = 0.
    for j in range(ltot // 2 + 1):
        k = 2 * j
        prod = N * fac2(k - 1) / ((2 * gamma) ** j)
        qlo = max(-k, (k - 2 * m))
        qhi = min( k, (2 * l - k)) + 1
        fk = 0.
        for q in range(qlo, qhi, 2):
            lt = (k + q) // 2
            mt = (k - q) // 2
            newt1 = fac(l) / fac(lt) / fac(l - lt)
            newt2 = fac(m) / fac(mt) / fac(m - mt)
            fk += newt1 * newt2 * (pa ** (l - lt)) * (pb ** (m - mt))
        tot += prod * fk
    return tot
