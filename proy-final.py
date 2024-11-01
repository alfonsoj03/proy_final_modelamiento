import numpy as np
import pandas as pd

def solver_euler(condicion_inicial, h, valores_reales):
    # Condición inicial
    x0, y0 = condicion_inicial
    
    # Definir la ecuación diferencial y' = (x + y - 1)^2
    def f(x, y):
        return (x + y - 1)**2
    
    # Inicializar listas para almacenar los valores
    Xn = [x0]
    YnEuler = [y0]
    YnEulerMejorado = [y0]
    ValorReal = valores_reales
    ErrorAbsolutoEuler = []
    ErrorRelativoEuler = []
    ErrorAbsolutoEulerMejorado = []
    ErrorRelativoEulerMejorado = []
    
    # Iteraciones (4 pasos adicionales, ya que tenemos 5 valores reales)
    n = len(valores_reales)
    
    for i in range(1, n):
        # Método de Euler
        y_euler = YnEuler[-1] + h * f(Xn[-1], YnEuler[-1])
        YnEuler.append(y_euler)
        
        # Método de Euler Mejorado (Heun)
        y_predict = YnEulerMejorado[-1] + h * f(Xn[-1], YnEulerMejorado[-1])
        y_euler_mejorado = YnEulerMejorado[-1] + (h/2) * (f(Xn[-1], YnEulerMejorado[-1]) + f(Xn[-1] + h, y_predict))
        YnEulerMejorado.append(y_euler_mejorado)
        
        # Actualizar Xn
        Xn.append(Xn[-1] + h)
    
    # Calcular errores para Euler y Euler Mejorado
    for i in range(n):
        # Errores para método de Euler
        error_abs_euler = abs(ValorReal[i] - YnEuler[i])
        error_rel_euler = abs((ValorReal[i] - YnEuler[i]) / ValorReal[i]) * 100
        ErrorAbsolutoEuler.append(error_abs_euler)
        ErrorRelativoEuler.append(error_rel_euler)
        
        # Errores para método de Euler Mejorado
        error_abs_mejorado = abs(ValorReal[i] - YnEulerMejorado[i])
        error_rel_mejorado = abs((ValorReal[i] - YnEulerMejorado[i]) / ValorReal[i]) * 100
        ErrorAbsolutoEulerMejorado.append(error_abs_mejorado)
        ErrorRelativoEulerMejorado.append(error_rel_mejorado)
    
    # Crear el dataset en forma de diccionario
    data = {
        'Xn': Xn,
        'YnEuler': YnEuler,
        'YnEulerMejorado': YnEulerMejorado,
        'ValorReal': ValorReal,
        'ErrorAbsolutoEuler': ErrorAbsolutoEuler,
        '% ErrorRelativoEuler': ErrorRelativoEuler,
        'ErrorAbsolutoEulerMejorado': ErrorAbsolutoEulerMejorado,
        '% ErrorRelativoEulerMejorado': ErrorRelativoEulerMejorado
    }
    
    # Convertir a DataFrame para mejor visualización
    df = pd.DataFrame(data)
    
    return df

def runge_kutta(condicion_inicial, h, valores_reales, f):
    # Condición inicial
    x0, y0 = condicion_inicial
    
    # Calcular valores de k y devolverlos en una lista
    def calcKs(xn, yn, h):
        lista_Ks = []
        # k1
        lista_Ks.append(f(xn, yn))
        # k2
        lista_Ks.append(f(xn + h/2, yn + (lista_Ks[0] * h/2)))
        # k3
        lista_Ks.append(f(xn + h/2, yn + (lista_Ks[1] * h/2)))
        # k4
        lista_Ks.append(f(xn + h, yn + (lista_Ks[2] * h)))
                
        return lista_Ks

    # Inicializar listas para almacenar los valores
    Xn = [x0]
    Yn = [y0]
    ValorReal = valores_reales
    ErrorAbsolutoRk4 = []
    ErrorRelativoRk4 = []
    
    # Iteraciones (4 pasos adicionales, ya que tenemos 5 valores reales)
    n = len(valores_reales)
    
    for i in range(1, n):
        # Xn, Yn
        # Calcular Ks
        lista_Ks = calcKs(Xn[-1], Yn[-1], h)
        # Calcular sig Yn+1
        sig_yn = Yn[-1] + (1/6 * ((lista_Ks[0]) + (2 * lista_Ks[1]) + (2 * lista_Ks[2])+ (lista_Ks[3])) * h)
        # Actualizar Xn y Yn
        Xn.append(Xn[-1] + h)
        Yn.append(sig_yn)
        
    for i in range(n):
        # Errores para método de rk4
        error_abs = abs(ValorReal[i] - Yn[i])
        error_rel = abs((ValorReal[i] - Yn[i]) / ValorReal[i]) * 100
        ErrorAbsolutoRk4.append(error_abs)
        ErrorRelativoRk4.append(error_rel)
        
    # Crear el dataset en forma de diccionario
    data = {
        'Xn': Xn,
        'Yn': Yn,
        'ValorReal': ValorReal,
        'ErrorAbsoluto': ErrorAbsolutoRk4,
        '% ErrorRelativo': ErrorRelativoRk4,
    }
    
    # Convertir a DataFrame para mejor visualización
    df = pd.DataFrame(data)
    
    return df

def adamsBashford(condicion_inicial, h, valores_reales):
    
    # Definir la ecuación diferencial
    def f(x, y):
        return (2*x - 3*y + 1)
    
    # Predecimos los primeros valores con rk4
    runge_data = runge_kutta(condicion_inicial, h, valores_reales, f)
    
    # Lista donde guardaremos los f(x,y)
    list_fn = []
    
    # Guardamos los items en la lista
    for xn, yn in zip(runge_data['Xn'], runge_data['Yn']):
        list_fn.append(f(xn, yn))
    
    # calculamos termino (55f0 - 59f-1 + 37f-2 - 9f-3)
    term_fn = (55*list_fn[3] - 59*list_fn[2] + 37*list_fn[1] - 9*list_fn[0])        
    
    # Prediccion y1 = y0 + h/24 (55f0 - 59f-1 + 37f-2 - 9f-3)
    y_pred_1 = runge_data['Yn'][3] + (h/24) * term_fn
    
    # Calcular f(x,y_pred_1)
    f1_pred = f(0.8, y_pred_1)
    
    # Pasar corrector y1c = y0 + h/24 (9f1_pred + 19fo + 5f-1 + f-2)
    y_final = runge_data['Yn'][3] + (h/24) * (9 * f1_pred + 19 * list_fn[3] - 5 * list_fn[2] + list_fn[1])

    Xn = [runge_data['Xn'][1], runge_data['Xn'][2], runge_data['Xn'][3], 0.8]
    Yn = [runge_data['Yn'][1], runge_data['Yn'][2], runge_data['Yn'][3], y_final]
    
    # Crear el dataset en forma de diccionario
    data = {
        'Xn': Xn,
        'Yn': Yn
    }
    
    # Convertir a DataFrame para mejor visualización
    df = pd.DataFrame(data)
    
    return df
    
    # RECUERDA GARANTIZAR QUE LLEGUEN LA CANTIDAD DE DATOS QUE NECESITAS, QUE SOLO SE PUEDEN GUARDAR 4 EN LIST_FN

# Valores reales proporcionados
valores_reales = [2, 2.12, 2.30, 2.59, 3.06, 3.90]

# Condición inicial y(0) = 2, paso h = 0.1
condicion_inicial = (0, 2)
h = 0.1

# Condicion inicial para adamsBashford
condicion_inicial_adams = (0, 1)

pd.set_option('display.float_format', lambda x: '%.4f' % x)

# Definir la ecuación diferencial y' = (x + y - 1)^2 para rk4
def f(x, y):
    return (x + y - 1)**2

# Llamada a la función
dataset_euler = solver_euler(condicion_inicial, h, valores_reales)
dataset_rk4 = runge_kutta(condicion_inicial, h, valores_reales, f)
dataset_adams = adamsBashford((0,1), 0.2, valores_reales)

print("Tarea 1")
print(dataset_euler)
print("Tarea 2")
print(dataset_rk4)
print("Tarea 3")
print(dataset_adams)