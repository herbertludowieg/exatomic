## -*- coding: utf-8 -*-
## Copyright (c) 2015-2018, Exa Analytics Development Team
## Distributed under the terms of the Apache License 2.0
#"""
#Radial Grid Spacing
######################
#This function is used by NWChem to define the radial grid on which the radial
#Hamiltonian is solved.
#"""
#import numba as nb
#import numpy as np
#
#
#@nb.jit(nopython=True, nogil=True)
#def scaled_logspace(a, b, n):
#    """
#    Scaled logarithmically spaced discrete points.
#
#    .. math::
#
#    r_i = ab^i\ \mathrm{for i=0, 1, ..., n}
#    """
#    r = np.empty((n, ), dtype=np.float64)
#    for i in range(n):
#        r[i] = b**i
#    return a*r
