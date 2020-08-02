# -*- coding: utf-8 -*-
# Compatible with Python 2.7.xx
# Copyright (C) 2019 Oscar Gerardo Lazo Arjona
# mailto: oscar.lazoarjona@physics.ox.ac.uk
r"""This is a template."""
import numpy as np
from sympy import zeros, sqrt, Abs, GramSchmidt, Matrix
from sympy import factorial as sym_fact
from math import factorial as num_fact
from scipy.sparse import csr_matrix


bfmt = "csr"
bfmtf = csr_matrix


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


def matrix_decompose(A):
    r"""Return the eigenvalues and eigenvectors of a `A`."""
    N = A.shape[0]

    # We build the transformation matrix
    eig = A.eigenvects()
    lam = []
    S = zeros(N)
    count = 0
    for i in range(len(eig)):
        for j in range(eig[i][1]):
            vj = eig[i][2][j]
            norm = sqrt(sum([Abs(vj[k])**2 for k in range(N)]))
            vj = vj/norm

            S[:, count] = vj
            lam += [eig[i][0]]
            count += 1

    # If the matrix is normal, we make sure we return a unitary
    # transformation matrix.
    if A*A.adjoint() - A.adjoint()*A == zeros(N):
        vlist = [S[:, i] for i in range(N)]
        vlist = GramSchmidt(vlist)
        for i in range(N):
            vi = vlist[i]
            norm = sqrt(sum([Abs(vi[k])**2 for k in range(N)]))
            S[:, i] = vlist[i]/norm
    return lam, S


def matrix_function(f, A):
    r"""Return the application of function `f` to matrix `A`."""
    N = A.shape[0]

    # We build the transformation matrix
    lam, S = matrix_decompose(A)

    # We get its inverse.
    if A*A.adjoint() - A.adjoint()*A == zeros(N):
        # The matrix is normal, so the inverse is just the adjoint.
        Si = S.adjoint()
    else:
        # Calculate the inverse.
        Si = S.inv()

    # We build the result matrix.
    fA = zeros(N)
    for i in range(N):
        fA += f(lam[i])*S[:, i]*Si[i, :]

    return fA


def D_coefficients(p, j, xaxis=None, d=1, symbolic=False):
    r"""Calculate finite difference coefficients that approximate the
    derivative of order ``d`` to precision order $p$ on point $j$ of an
    arbitrary grid.

    INPUT:

    -  ``p`` - int, the precission order of the approximation.

    -  ``j`` - int, the point where the approximation is centered.

    -  ``xaxis`` - an array, the grid on which the function is represented.

    -  ``d`` - int, the order of the derivative.

    -  ``symbolic`` - a bool, whether to return symbolic coefficients.

    OUTPUT:

    An array of finite difference coefficients.

    Examples
    ========

    First order derivatives:
    >>> from sympy import pprint
    >>> pprint(D_coefficients(2, 0, symbolic=True))
    [-3/2  2  -1/2]
    >>> pprint(D_coefficients(2, 1, symbolic=True))
    [-1/2  0  1/2]
    >>> pprint(D_coefficients(2, 2, symbolic=True))
    [1/2  -2  3/2]

    Second order derivatives:
    >>> pprint(D_coefficients(2, 0, d=2, symbolic=True))
    [1  -2  1]
    >>> pprint(D_coefficients(3, 0, d=2, symbolic=True))
    [2  -5  4  -1]
    >>> pprint(D_coefficients(3, 1, d=2, symbolic=True))
    [1  -2  1  0]
    >>> pprint(D_coefficients(3, 2, d=2, symbolic=True))
    [0  1  -2  1]
    >>> pprint(D_coefficients(3, 3, d=2, symbolic=True))
    [-1  4  -5  2]

    A non uniform grid:
    >>> x = np.array([1.0, 3.0, 5.0])
    >>> print(D_coefficients(2, 1, xaxis=x))
    [-0.25  0.    0.25]

    """
    def poly_deri(x, a, n):
        if a-n >= 0:
            if symbolic:
                return sym_fact(a)/sym_fact(a-n)*x**(a-n)
            else:
                return num_fact(a)/float(num_fact(a-n))*x**(a-n)
        else:
            return 0.0

    if d > p:
        mes = "Cannot calculate a derivative of order "
        mes += "`d` larger than precision `p`."
        raise ValueError(mes)
    Nt = p+1
    if symbolic:
        arr = Matrix
    else:
        arr = np.array

    if xaxis is None:
        xaxis = arr([i for i in range(Nt)])

    zp = arr([poly_deri(xaxis[j], i, d) for i in range(Nt)])
    eqs = arr([[xaxis[ii]**jj for jj in range(Nt)] for ii in range(Nt)])

    if symbolic:
        coefficients = zp.transpose()*eqs.inv()
    else:
        coefficients = np.dot(zp.transpose(), np.linalg.inv(eqs))
    return coefficients


def derivative_operator(xaxis, p=2, symbolic=False, sparse=False):
    u"""A matrix representation of the differential operator for an arbitrary
    xaxis.

    Multiplying the returned matrix by a discretized function gives the second
    order centered finite difference for all points except the extremes, where
    a forward and backward second order finite difference is used for the
    first and last points respectively.

    Setting higher=True gives a fourth order approximation for the extremes.

    Setting symbolic=True gives a symbolic exact representation of the
    coefficients.

    INPUT:

    -  ``xaxis`` - an array, the grid on which the function is represented.

    -  ``p`` - int, the precission order of the approximation.

    -  ``symbolic`` - a bool, whether to return symbolic coefficients.

    -  ``sparse`` - a bool, whether to return a sparse matrix.

    OUTPUT:

    A 2-d array representation of the differential operator.

    Examples
    ========

    >>> from sympy import pprint
    >>> D = derivative_operator(range(5))
    >>> print(D)
    [[-1.5  2.  -0.5  0.   0. ]
     [-0.5  0.   0.5  0.   0. ]
     [ 0.  -0.5  0.   0.5  0. ]
     [ 0.   0.  -0.5  0.   0.5]
     [ 0.   0.   0.5 -2.   1.5]]

    >>> D = derivative_operator(range(5), p=4)
    >>> print(D)
    [[-2.08333333  4.         -3.          1.33333333 -0.25      ]
     [-0.25       -0.83333333  1.5        -0.5         0.08333333]
     [ 0.08333333 -0.66666667  0.          0.66666667 -0.08333333]
     [-0.08333333  0.5        -1.5         0.83333333  0.25      ]
     [ 0.25       -1.33333333  3.         -4.          2.08333333]]

    >>> D = derivative_operator(range(5), p=4, symbolic=True)
    >>> pprint(D)
    ⎡-25                           ⎤
    ⎢────    4     -3   4/3   -1/4 ⎥
    ⎢ 12                           ⎥
    ⎢                              ⎥
    ⎢-1/4   -5/6  3/2   -1/2  1/12 ⎥
    ⎢                              ⎥
    ⎢1/12   -2/3   0    2/3   -1/12⎥
    ⎢                              ⎥
    ⎢-1/12  1/2   -3/2  5/6    1/4 ⎥
    ⎢                              ⎥
    ⎢                          25  ⎥
    ⎢ 1/4   -4/3   3     -4    ──  ⎥
    ⎣                          12  ⎦

    >>> D = derivative_operator([1, 2, 4, 6, 7], p=2, symbolic=True)
    >>> pprint(D)
    ⎡-4/3  3/2   -1/6   0     0 ⎤
    ⎢                           ⎥
    ⎢-2/3  1/2   1/6    0     0 ⎥
    ⎢                           ⎥
    ⎢ 0    -1/4   0    1/4    0 ⎥
    ⎢                           ⎥
    ⎢ 0     0    -1/6  -1/2  2/3⎥
    ⎢                           ⎥
    ⎣ 0     0    1/6   -3/2  4/3⎦


    """
    def rel_dif(a, b):
        if a > b:
            return 1-b/a
        else:
            return 1-a/b
    #########################################################################
    if symbolic and sparse:
        mes = "There is no symbolic sparse implementation."
        raise NotImplementedError(mes)

    N = len(xaxis); h = xaxis[1] - xaxis[0]
    if p % 2 != 0:
        raise ValueError("The precission must be even.")
    if N < p+1:
        raise ValueError("N < p+1!")

    if symbolic:
        D = zeros(N)
    else:
        D = np.zeros((N, N))
    #########################################################################
    hlist = [xaxis[i+1] - xaxis[i] for i in range(N-1)]
    err = np.any([rel_dif(hlist[i], h) >= 1e-5 for i in range(N-1)])
    if not err:
        coefficients = [D_coefficients(p, i, symbolic=symbolic)
                        for i in range(p+1)]
        mid = int((p+1)/2)

        # We put in place the middle coefficients.
        for i in range(mid, N-mid):
            a = i-mid; b = a+p+1
            D[i, a:b] = coefficients[mid]

        # We put in place the forward coefficients.
        for i in range(mid):
            D[i, :p+1] = coefficients[i]

        # We put in place the backward coefficients.
        for i in range(N-mid, N):
            D[i, N-p-1:N] = coefficients[p+1-N+i]

        D = D/h
    else:
        # We generate a p + 1 long list for each of the N rows.
        for i in range(N):
            if i < p/2:
                a = 0
                jj = i
            elif i >= N - p/2:
                a = N - p - 1
                jj = (i - (N - p - 1))
            else:
                a = i - p/2
                jj = p/2
            b = a + p + 1
            D[i, a: b] = D_coefficients(p, jj, xaxis=xaxis[a:b],
                                        symbolic=symbolic)
    if sparse:
        return bfmtf(D)
    return D
