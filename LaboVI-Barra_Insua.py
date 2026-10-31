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
    