# -*- coding: utf-8 -*-
# Compatible with Python 2.7.xx
# Copyright (C) 2019 Oscar Gerardo Lazo Arjona
# mailto: oscar.lazoarjona@physics.ox.ac.uk
r"""This is a template."""
import numpy as np


def num_integral(t, f):
    """We integrate using the trapezium rule."""
    dt = t[1]-t[0]
    F = sum(f[1:-1])
    F += (f[1] + f[-1])*0.5
    return np.real(F*dt)


def ffftfreq(t):
    r"""Calculate the angular frequency axis for a given time axis."""
    dt = t[1]-t[0]
    nu = np.fft.fftshift(np.fft.fftfreq(t.size, dt))
    return nu


def ffftfft(f, t):
    r"""Calculate the Fourier transform."""
    dt = t[1]-t[0]
    return np.fft.fftshift(np.fft.fft(np.fft.ifftshift(f)))*dt


def iffftfft(f, nu):
    r"""Calculate the inverse Fourier transform."""
    Deltanu = nu[-1]-nu[0]
    return np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(f)))*Deltanu
