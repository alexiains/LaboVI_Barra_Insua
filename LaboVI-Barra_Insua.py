# Materia: Informatica II
# Integrantes: Barra, Azul - Insua, Alexia
# Laboratorio VI.1 Calculo de raices, derivadas, sistemas de ecuaciones lineales e integrales con GUI
# URL repositorio: https://github.com/alexiains/LaboVI_Barra_Insua.git

import math
import tkinter as tk
from tkinter import ttk, messagebox

# EVALUACIÓN DE POLINOMIO

def evaluar_polinomio(coefs, x):
    """Evalúa P(x) = c0 + c1*x + c2*x^2 + ... + cn*x^n"""
    res = 0.0
    for i, c in enumerate(coefs):
        res += c * (x ** i)
    return res

def evaluar_derivada_polinomio(coefs, x):
    """Evalúa la derivada analítica P'(x) = c1 + 2*c2*x + 3*c3*x^2 + ..."""
    res = 0.0
    for i in range(1, len(coefs)):
        res += i * coefs[i] * (x ** (i - 1))
    return res

def formatear_cifras_significativas(valor, cifras):
    """Formatea un valor flotante a la cantidad de cifras significativas requerida."""
    if valor == 0 or not math.isfinite(valor):
        return f"{valor}"
    try:
        cifras = max(1, int(cifras))
        return f"{valor:.{cifras}g}"
    except Exception:
        return str(valor)

# RAÍCES DE ECUACIONES

def biseccion(f, xl, xu, es, imax):
    fl = f(xl)
    fu = f(xu)
    if fl * fu > 0:
        return None, 0, 0.0, "El intervalo [xl, xu] no encierra una raíz (sin cambio de signo)."

    xr = xl
    ea = 100.0
    iter_count = 0

    for i in range(1, imax + 1):
        xrold = xr
        xr = (xl + xu) / 2.0
        iter_count = i

        if xr != 0:
            ea = abs((xr - xrold) / xr) * 100.0

        fr = f(xr)
        test = fl * fr

        if test < 0:
            xu = xr
            fu = fr
        elif test > 0:
            xl = xr
            fl = fr
        else:
            ea = 0.0

        if ea < es:
            return xr, iter_count, ea, None

    return xr, iter_count, ea, "Se alcanzó el número máximo de iteraciones."

def newton_raphson(f, df, x0, es, imax):
    x = x0
    ea = 100.0
    iter_count = 0

    for i in range(1, imax + 1):
        xold = x
        d = df(x)
        if abs(d) < 1e-14:
            return None, i, ea, f"Derivada analítica nula o casi nula en x = {x:.6f}."

        x = xold - f(xold) / d
        iter_count = i

        if x != 0:
            ea = abs((x - xold) / x) * 100.0

        if ea < es:
            return x, iter_count, ea, None

    return x, iter_count, ea, "Se alcanzó el número máximo de iteraciones."

def secante(f, x_minus1, x0, es, imax):
    x_prev = x_minus1
    x_curr = x0
    ea = 100.0
    iter_count = 0

    for i in range(1, imax + 1):
        f_prev = f(x_prev)
        f_curr = f(x_curr)

        denom = f_curr - f_prev
        if abs(denom) < 1e-14:
            return None, i, ea, "Denominador nulo en el método de la secante."

        x_next = x_curr - (f_curr * (x_prev - x_curr)) / (f_prev - f_curr)
        iter_count = i

        if x_next != 0:
            ea = abs((x_next - x_curr) / x_next) * 100.0

        x_prev = x_curr
        x_curr = x_next

        if ea < es:
            return x_curr, iter_count, ea, None

    return x_curr, iter_count, ea, "Se alcanzó el número máximo de iteraciones."    

# DIFERENCIACIÓN NUMÉRICA

def diferencia_centrada(f, x, h):
    if h == 0:
        raise ValueError("El tamaño de paso h no puede ser cero.")
    return (f(x + h) - f(x - h)) / (2.0 * h)

# SISTEMAS DE ECUACIONES LINEALES (GAUSS)

def gauss_sin_pivoteo(A_mat, b_vec):
    n = len(b_vec)
    A = [[float(A_mat[i][j]) for j in range(n)] for i in range(n)]
    b = [float(b_vec[i]) for i in range(n)]

    # Eliminacion hacia adelante
    for q in range(n - 1):
        if abs(A[q][q]) < 1e-14:
            return None, f"Pivote nulo detectado. A[{q+1}][{q+1}] = 0 en la etapa {q+1}. Gauss sin pivoteo cancelado."
        for k in range(q + 1, n):
            m = A[k][q] / A[q][q]
            for j in range(q, n):
                A[k][j] -= m * A[q][j]
            b[k] -= m * b[q]

    if abs(A[n-1][n-1]) < 1e-14:
        return None, f"Pivote nulo detectado. A[{n}][{n}] = 0 en el último elemento diagonal."

    # Sustitucion regresiva
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        suma = 0.0
        for j in range(i + 1, n):
            suma += A[i][j] * x[j]
        x[i] = (b[i] - suma) / A[i][i]

    return x, None

def gauss_con_pivoteo(A_mat, b_vec):
    n = len(b_vec)
    A = [[float(A_mat[i][j]) for j in range(n)] for i in range(n)]
    b = [float(b_vec[i]) for i in range(n)]

    # Eliminacion hacia adelante con pivoteo parcial
    for q in range(n - 1):
        p = q
        max_val = abs(A[q][q])
        for i in range(q + 1, n):
            if abs(A[i][q]) > max_val:
                max_val = abs(A[i][q])
                p = i

        if max_val < 1e-14:
            return None, "El sistema es singular o no posee solución única."

        if p != q:
            A[q], A[p] = A[p], A[q]
            b[q], b[p] = b[p], b[q]

        for k in range(q + 1, n):
            m = A[k][q] / A[q][q]
            for j in range(q, n):
                A[k][j] -= m * A[q][j]
            b[k] -= m * b[q]

    if abs(A[n-1][n-1]) < 1e-14:
        return None, "El sistema es singular (matriz no invertible)."

    # Sustitucion regresiva
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        suma = 0.0
        for j in range(i + 1, n):
            suma += A[i][j] * x[j]
        x[i] = (b[i] - suma) / A[i][i]

    return x, None

# INTEGRACION NUMERICA

def simpson_13_compuesta(f, a, b, n):
    if n % 2 != 0:
        return None, "La cantidad de segmentos 'n' debe ser un número par."
    if n <= 0:
        return None, "La cantidad de segmentos 'n' debe ser un entero positivo mayor a cero."

    h = (b - a) / n
    suma = f(a) + f(b)

    for i in range(1, n):
        x_i = a + i * h
        if i % 2 != 0:
            suma += 4.0 * f(x_i)
        else:
            suma += 2.0 * f(x_i)

    integral = (h / 3.0) * suma
    return integral, None