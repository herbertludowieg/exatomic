# -*- coding: utf-8 -*-
'''
Diffusion Coefficients
========================
Various algorithms for computing diffusion coefficients are coded here.
'''
from exa import _np as np
from exa import _pd as pd
from atomic import Length, Time
from atomic.algorithms.displacement import absolute_msd


def einstein_relation(universe, length='cm', time='s'):
    '''
    Compute the (time dependent) diffusion coefficient using Einstein's relation.

    .. math::

        D\left(t\\right) = \\frac{1}{6Nt}\\sum_{i=1}^{N}\\left|\\mathbf{r}_{i}\left(t\\right)
            - \\mathbf{r}_{i}\\left(0\\right)\\right|^{2}

        D = \\lim_{t\\to\\infty} D\\left(t\\right)

    Args:
        universe (:class:`~atomic.Universe`): The universe object
        msd (:class:`~exa.DataFrame`): Mean squared displacement dataframe

    Returns:
        D (:class:`~exa.DataFrame`): Diffussion coefficient as a function of time

    Note:
        The asymptotic value of the returned variable is the diffusion coefficient.
        The default units of the diffusion coefficient are :math:`\\frac{cm^{2}}{s}`.
    '''
    msd = absolute_msd(universe).mean(axis=1)
    t = universe.frame['time'].values
    return msd / (6 * t)
    #return msd / (6 * t) * Length['au', length]**2 / Time['au', time]
