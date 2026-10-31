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

# INTERFAZ GRAFICA

class AplicacionCalculoNumerico(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Laboratorio VI.1 - Calculo Numerico")
        self.minsize(850, 600)

        self.style = ttk.Style()
        self.style.theme_use('clam')

        self.scrollable_frame = ttk.Frame(self, padding="10")
        self.scrollable_frame.pack(fill=tk.BOTH, expand=True)

        # COEFICIENTES DEL POLINOMIO
        self.crear_seccion_polinomio(self.scrollable_frame)

        # GRILLA DE LAS 4 OPERACIONES
        frame_operaciones = ttk.Frame(self.scrollable_frame)
        frame_operaciones.pack(fill=tk.BOTH, expand=True, pady=5)

        # RAICES
        self.crear_seccion_raices(frame_operaciones)

        # DIFERENCIACIÓN
        self.crear_seccion_derivacion(frame_operaciones)

        # GAUSS
        self.crear_seccion_gauss(frame_operaciones)

        # INTEGRACIÓN
        self.crear_seccion_integracion(frame_operaciones)

    # POLINOMIO (P(x))

    def crear_seccion_polinomio(self, parent):
        frame_poly = ttk.LabelFrame(parent, text=" Coeficientes del Polinomio P(x) = c0 + c1*x + c2*x² + ... + c6*x⁶ ", padding=10)
        frame_poly.pack(fill=tk.X, pady=5)

        self.entries_coefs = []
        for i in range(7):
            lbl_c = ttk.Label(frame_poly, text=f"c{i}:", font=('Helvetica', 9, 'bold'))
            lbl_c.grid(row=0, column=i*2, padx=2, sticky=tk.E)
            ent_c = ttk.Entry(frame_poly, width=8, justify='center')
            ent_c.grid(row=0, column=i*2+1, padx=6, pady=2)
            self.entries_coefs.append(ent_c)

    def obtener_coeficientes(self):
        coefs = []
        for ent in self.entries_coefs:
            txt = ent.get().strip()
            coefs.append(float(txt) if txt else 0.0)
        return coefs

    # RAICES DE ECUACIONES

    def crear_seccion_raices(self, parent):
        frame = ttk.LabelFrame(parent, text=" 1. Raíces de Polinomios ", padding=10)
        frame.grid(row=0, column=0, sticky=tk.NSEW, padx=5, pady=5)
        parent.columnconfigure(0, weight=1)

        ttk.Label(frame, text="Método:").grid(row=0, column=0, sticky=tk.E, padx=5, pady=2)
        self.var_metodo = tk.StringVar(value="Bisección")
        cb_metodo = ttk.Combobox(frame, textvariable=self.var_metodo, state="readonly", width=16,
                                 values=["Bisección", "Newton-Raphson", "Secante"])
        cb_metodo.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        cb_metodo.bind("<<ComboboxSelected>>", self.on_metodo_change)

        self.lbl_p1 = ttk.Label(frame, text="Límite inferior (xl):")
        self.lbl_p1.grid(row=1, column=0, sticky=tk.E, padx=5, pady=2)
        self.ent_p1 = ttk.Entry(frame, width=12)
        self.ent_p1.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)

        self.lbl_p2 = ttk.Label(frame, text="Límite superior (xu):")
        self.lbl_p2.grid(row=2, column=0, sticky=tk.E, padx=5, pady=2)
        self.ent_p2 = ttk.Entry(frame, width=12)
        self.ent_p2.grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)

        self.var_stop_type = tk.StringVar(value="es")
        rb_es = ttk.Radiobutton(frame, text="Error εs (%)", variable=self.var_stop_type, value="es", command=self.on_stop_type_change)
        rb_es.grid(row=3, column=0, sticky=tk.E, padx=5)
        rb_cifras = ttk.Radiobutton(frame, text="Cifras sig. (n)", variable=self.var_stop_type, value="cifras", command=self.on_stop_type_change)
        rb_cifras.grid(row=3, column=1, sticky=tk.W, padx=5)

        self.lbl_val_stop = ttk.Label(frame, text="Valor εs (%):")
        self.lbl_val_stop.grid(row=4, column=0, sticky=tk.E, padx=5, pady=2)
        self.ent_val_stop = ttk.Entry(frame, width=12)
        self.ent_val_stop.grid(row=4, column=1, sticky=tk.W, padx=5, pady=2)

        ttk.Label(frame, text="Máx. Iteraciones:").grid(row=5, column=0, sticky=tk.E, padx=5, pady=2)
        self.ent_imax = ttk.Entry(frame, width=12)
        self.ent_imax.grid(row=5, column=1, sticky=tk.W, padx=5, pady=2)

        btn_calc = ttk.Button(frame, text="Calcular Raíz", command=self.calcular_raiz)
        btn_calc.grid(row=6, column=0, columnspan=2, pady=8)

        frame_res = ttk.Frame(frame)
        frame_res.grid(row=7, column=0, columnspan=2, sticky=tk.W, pady=2)

        self.lbl_res_raiz = ttk.Label(frame_res, text="Raíz (xr): ---", font=('Helvetica', 9, 'bold'))
        self.lbl_res_raiz.pack(anchor=tk.W)
        self.lbl_res_iters = ttk.Label(frame_res, text="Iteraciones: ---")
        self.lbl_res_iters.pack(anchor=tk.W)
        self.lbl_res_ea = ttk.Label(frame_res, text="Error ea (%): ---")
        self.lbl_res_ea.pack(anchor=tk.W)
        self.lbl_res_fxr = ttk.Label(frame_res, text="f(xr): ---")
        self.lbl_res_fxr.pack(anchor=tk.W)

    def on_metodo_change(self, event=None):
        met = self.var_metodo.get()
        if met == "Bisección":
            self.lbl_p1.config(text="Límite inferior (xl):")
            self.lbl_p2.config(text="Límite superior (xu):")
            self.ent_p1.delete(0, tk.END)
            self.ent_p2.delete(0, tk.END)
            self.ent_p2.config(state=tk.NORMAL)
        elif met == "Newton-Raphson":
            self.lbl_p1.config(text="Valor inicial (x0):")
            self.lbl_p2.config(text="---")
            self.ent_p1.delete(0, tk.END)
            self.ent_p2.delete(0, tk.END)
            self.ent_p2.config(state=tk.DISABLED)
        elif met == "Secante":
            self.lbl_p1.config(text="Valor inicial (x-1):")
            self.lbl_p2.config(text="Valor inicial (x0):")
            self.ent_p1.delete(0, tk.END)
            self.ent_p2.config(state=tk.NORMAL)
            self.ent_p2.delete(0, tk.END)

    def on_stop_type_change(self):
        self.ent_val_stop.delete(0, tk.END)
        if self.var_stop_type.get() == "es":
            self.lbl_val_stop.config(text="Valor εs (%):")
        else:
            self.lbl_val_stop.config(text="Cifras significativas (n):")

    def calcular_raiz(self):
        try:
            coefs = self.obtener_coeficientes()
            if all(c == 0 for c in coefs):
                messagebox.showwarning("Atención", "Por favor ingrese al menos un coeficiente no nulo en P(x).")
                return

            f = lambda x: evaluar_polinomio(coefs, x)
            df = lambda x: evaluar_derivada_polinomio(coefs, x)

            txt_imax = self.ent_imax.get().strip()
            imax = int(txt_imax) if txt_imax else 100

            txt_stop = self.ent_val_stop.get().strip()
            if not txt_stop:
                messagebox.showerror("Error", "Ingrese el criterio de parada (εs o n).")
                return

            if self.var_stop_type.get() == "es":
                es = float(txt_stop)
                n_cifras = 4
            else:
                n = int(txt_stop)
                n_cifras = n
                es = 0.5 * (10.0 ** (2.0 - n))

            metodo = self.var_metodo.get()

            if metodo == "Bisección":
                if not self.ent_p1.get().strip() or not self.ent_p2.get().strip():
                    messagebox.showerror("Error", "Ingrese xl y xu para Bisección.")
                    return
                xl = float(self.ent_p1.get())
                xu = float(self.ent_p2.get())
                xr, iters, ea, err = biseccion(f, xl, xu, es, imax)
            elif metodo == "Newton-Raphson":
                if not self.ent_p1.get().strip():
                    messagebox.showerror("Error", "Ingrese x0 para Newton-Raphson.")
                    return
                x0 = float(self.ent_p1.get())
                xr, iters, ea, err = newton_raphson(f, df, x0, es, imax)
            elif metodo == "Secante":
                if not self.ent_p1.get().strip() or not self.ent_p2.get().strip():
                    messagebox.showerror("Error", "Ingrese x-1 y x0 para Secante.")
                    return
                xm1 = float(self.ent_p1.get())
                x0 = float(self.ent_p2.get())
                xr, iters, ea, err = secante(f, xm1, x0, es, imax)

            if err or xr is None:
                messagebox.showwarning("Atención", err or "No se encontró la raíz.")
                self.lbl_res_raiz.config(text="Raíz (xr): NO ENCONTRADA")
                self.lbl_res_iters.config(text=f"Iteraciones: {iters}")
                self.lbl_res_ea.config(text="Error ea (%): ---")
                self.lbl_res_fxr.config(text="f(xr): ---")
            else:
                xr_fmt = formatear_cifras_significativas(xr, n_cifras)
                self.lbl_res_raiz.config(text=f"Raíz (xr): {xr_fmt} ({xr:.6f})")
                self.lbl_res_iters.config(text=f"Iteraciones: {iters}")
                self.lbl_res_ea.config(text=f"Error ea (%): {ea:.6f} %")
                self.lbl_res_fxr.config(text=f"f(xr): {f(xr):.6e}")

        except ValueError:
            messagebox.showerror("Error de entrada", "Ingrese valores numéricos válidos.")

    #DIFERENCIACIÓN NUMÉRICA
    def crear_seccion_derivacion(self, parent):
            frame = ttk.LabelFrame(parent, text=" 2. Diferenciación Numérica ", padding=10)
            frame.grid(row=0, column=1, sticky=tk.NSEW, padx=5, pady=5)
            parent.columnconfigure(1, weight=1)

            ttk.Label(frame, text="Punto x:").grid(row=0, column=0, sticky=tk.E, padx=5, pady=5)
            self.ent_deriv_x = ttk.Entry(frame, width=12)
            self.ent_deriv_x.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)

            ttk.Label(frame, text="Paso h (dx):").grid(row=1, column=0, sticky=tk.E, padx=5, pady=5)
            self.ent_deriv_h = ttk.Entry(frame, width=12)
            self.ent_deriv_h.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)

            btn_calc_deriv = ttk.Button(frame, text="Calcular Derivada", command=self.calcular_derivada)
            btn_calc_deriv.grid(row=2, column=0, columnspan=2, pady=12)

            frame_res = ttk.Frame(frame)
            frame_res.grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=5)

            self.lbl_res_deriv = ttk.Label(frame_res, text="Derivada aprox. f'(x): ---", font=('Helvetica', 9, 'bold'))
            self.lbl_res_deriv.pack(anchor=tk.W, pady=2)
            self.lbl_res_exacta = ttk.Label(frame_res, text="Derivada exacta: ---")
            self.lbl_res_exacta.pack(anchor=tk.W, pady=2)
            self.lbl_res_err_deriv = ttk.Label(frame_res, text="Error verd. et (%): ---")
            self.lbl_res_err_deriv.pack(anchor=tk.W, pady=2)

    def calcular_derivada(self):
            try:
                if not self.ent_deriv_x.get().strip() or not self.ent_deriv_h.get().strip():
                    messagebox.showerror("Error", "Ingrese el punto x y el paso h.")
                    return

                x = float(self.ent_deriv_x.get())
                h = float(self.ent_deriv_h.get())
                if h <= 0:
                    messagebox.showerror("Error", "El paso h debe ser mayor a cero.")
                    return

                coefs = self.obtener_coeficientes()
                if all(c == 0 for c in coefs):
                    messagebox.showwarning("Atención", "Ingrese coeficientes en la sección del Polinomio P(x).")
                    return

                f = lambda x_val: evaluar_polinomio(coefs, x_val)
                df = lambda x_val: evaluar_derivada_polinomio(coefs, x_val)

                d_aprox = diferencia_centrada(f, x, h)
                d_exacta = df(x)

                et = abs((d_exacta - d_aprox) / d_exacta) * 100.0 if d_exacta != 0 else 0.0

                self.lbl_res_deriv.config(text=f"Derivada aprox. f'({x}): {d_aprox:.8f}")
                self.lbl_res_exacta.config(text=f"Derivada exacta: {d_exacta:.8f}")
                self.lbl_res_err_deriv.config(text=f"Error verd. et: {et:.6f} %")
            except ValueError:
                messagebox.showerror("Error de entrada", "Ingrese valores numéricos válidos.")

    # GAUSS
    
    def crear_seccion_gauss(self, parent):
        frame = ttk.LabelFrame(parent, text=" 3. Sistemas de Ecuaciones Lineales 4x4 (Gauss) ", padding=10)
        frame.grid(row=1, column=0, sticky=tk.NSEW, padx=5, pady=5)

        frame_grid = ttk.Frame(frame)
        frame_grid.pack(pady=2)

        self.entries_A = []
        self.entries_b = []

        for j in range(4):
            ttk.Label(frame_grid, text=f"x{j+1}", font=('Helvetica', 8, 'bold')).grid(row=0, column=j, padx=4)
        ttk.Label(frame_grid, text="= b", font=('Helvetica', 8, 'bold')).grid(row=0, column=4, padx=4)

        for i in range(4):
            row_entries = []
            for j in range(4):
                ent = ttk.Entry(frame_grid, width=6, justify='center')
                ent.grid(row=i+1, column=j, padx=2, pady=2)
                row_entries.append(ent)
            self.entries_A.append(row_entries)

            ent_b = ttk.Entry(frame_grid, width=6, justify='center')
            ent_b.grid(row=i+1, column=4, padx=4, pady=2)
            self.entries_b.append(ent_b)

        frame_opts = ttk.Frame(frame)
        frame_opts.pack(pady=2)

        self.var_pivoteo = tk.BooleanVar(value=True)
        rb_con_piv = ttk.Radiobutton(frame_opts, text="Con Pivoteo", variable=self.var_pivoteo, value=True)
        rb_con_piv.pack(side=tk.LEFT, padx=5)
        rb_sin_piv = ttk.Radiobutton(frame_opts, text="Sin Pivoteo", variable=self.var_pivoteo, value=False)
        rb_sin_piv.pack(side=tk.LEFT, padx=5)

        btn_solve = ttk.Button(frame, text="Resolver Sistema", command=self.resolver_gauss)
        btn_solve.pack(pady=6)

        self.lbl_res_gauss = ttk.Label(frame, text="x = [---, ---, ---, ---]", font=('Helvetica', 9, 'bold'))
        self.lbl_res_gauss.pack(anchor=tk.W, pady=2)

    def resolver_gauss(self):
        try:
            A = []
            for i in range(4):
                row = []
                for j in range(4):
                    txt = self.entries_A[i][j].get().strip()
                    if not txt:
                        messagebox.showerror("Error", f"Falta ingresar A[{i+1}][{j+1}].")
                        return
                    row.append(float(txt))
                A.append(row)

            b = []
            for i in range(4):
                txt = self.entries_b[i].get().strip()
                if not txt:
                    messagebox.showerror("Error", f"Falta ingresar término independiente b[{i+1}].")
                    return
                b.append(float(txt))

            if self.var_pivoteo.get():
                x, err = gauss_con_pivoteo(A, b)
            else:
                x, err = gauss_sin_pivoteo(A, b)

            if err or x is None:
                messagebox.showerror("Error en Gauss", err)
                self.lbl_res_gauss.config(text="Solución: NO DISPONIBLE")
            else:
                res_str = ", ".join([f"x{i+1}={val:.4f}" for i, val in enumerate(x)])
                self.lbl_res_gauss.config(text=f"Solución: {res_str}")
        except ValueError:
            messagebox.showerror("Error de entrada", "Ingrese valores numéricos válidos.")

    # INTEGRACIÓN NUMÉRICA

    def crear_seccion_integracion(self, parent):
        frame = ttk.LabelFrame(parent, text=" 4. Integración Numérica (Simpson 1/3) ", padding=10)
        frame.grid(row=1, column=1, sticky=tk.NSEW, padx=5, pady=5)

        ttk.Label(frame, text="Límite inferior (a):").grid(row=0, column=0, sticky=tk.E, padx=5, pady=4)
        self.ent_int_a = ttk.Entry(frame, width=12)
        self.ent_int_a.grid(row=0, column=1, sticky=tk.W, padx=5, pady=4)

        ttk.Label(frame, text="Límite superior (b):").grid(row=1, column=0, sticky=tk.E, padx=5, pady=4)
        self.ent_int_b = ttk.Entry(frame, width=12)
        self.ent_int_b.grid(row=1, column=1, sticky=tk.W, padx=5, pady=4)

        ttk.Label(frame, text="Segmentos n (par):").grid(row=2, column=0, sticky=tk.E, padx=5, pady=4)
        self.ent_int_n = ttk.Entry(frame, width=12)
        self.ent_int_n.grid(row=2, column=1, sticky=tk.W, padx=5, pady=4)

        btn_calc_int = ttk.Button(frame, text="Calcular Integral", command=self.calcular_integracion)
        btn_calc_int.grid(row=3, column=0, columnspan=2, pady=10)

        frame_res = ttk.Frame(frame)
        frame_res.grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=2)

        self.lbl_res_int = ttk.Label(frame_res, text="Integral I ≈ ---", font=('Helvetica', 9, 'bold'))
        self.lbl_res_int.pack(anchor=tk.W, pady=2)
        self.lbl_res_h = ttk.Label(frame_res, text="Ancho h = (b-a)/n: ---")
        self.lbl_res_h.pack(anchor=tk.W, pady=2)

    def calcular_integracion(self):
        try:
            if not self.ent_int_a.get().strip() or not self.ent_int_b.get().strip() or not self.ent_int_n.get().strip():
                messagebox.showerror("Error", "Ingrese los límites a, b y los segmentos n.")
                return

            a = float(self.ent_int_a.get())
            b = float(self.ent_int_b.get())
            n = int(self.ent_int_n.get())

            coefs = self.obtener_coeficientes()
            if all(c == 0 for c in coefs):
                messagebox.showwarning("Atención", "Ingrese coeficientes en la sección del Polinomio P(x).")
                return

            f = lambda x_val: evaluar_polinomio(coefs, x_val)

            integral, err = simpson_13_compuesta(f, a, b, n)

            if err:
                messagebox.showerror("Error en Integración", err)
                self.lbl_res_int.config(text="Integral I ≈ ---")
                self.lbl_res_h.config(text="Ancho h: ---")
            else:
                h = (b - a) / n
                self.lbl_res_int.config(text=f"Integral I ≈ {integral:.8f}")
                self.lbl_res_h.config(text=f"Ancho h = {h:.6f}")
        except ValueError:
            messagebox.showerror("Error de entrada", "Ingrese valores numéricos válidos.")

if __name__ == "__main__":
    app = AplicacionCalculoNumerico()
    app.mainloop()