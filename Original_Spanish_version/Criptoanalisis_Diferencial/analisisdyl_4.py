import subprocess
import os
import time
import openpyxl
import pandas as pd
import itertools
import sys


# Función principal
def ejecutar_codigo(x_1, x_2, tipo, nombre_codigo, n_bits, foo):

    # Funciones auxiliares
    def str_to_bin(s):
        return ''.join(format(ord(char), '08b') for char in s)

    def bin_to_str(bin_str):
        bin_str = bin_str.zfill((len(bin_str) + 7) // 8 * 8)
        chars = []

        try:
            for i in range(0, len(bin_str), 8):
                byte = bin_str[i:i+8]
                char_code = int(byte, 2)
                if 32 <= char_code <= 126:
                    chars.append(chr(char_code))
                else:
                    chars.append(str(char_code % 95 + 32))
            return ''.join(chars)
        except Exception as e:
            print(f"Error en bin_to_str: {e}, bin_str={bin_str}")
            return str(int(bin_str, 2))

    def reescribir_xslx(nuevo_titulo, entrada_actual):
        max_attempts = 5

        for attempt in range(max_attempts):
            if os.path.exists("exp.xlsx"):
                if os.path.exists(nuevo_titulo):
                    print(f"El archivo {nuevo_titulo} ya existe. Sobreescribiendo...")
                    os.remove(nuevo_titulo)
                os.rename("exp.xlsx", nuevo_titulo)
                return
            print(f"No se encontró exp.xlsx (intento {attempt+1}/{max_attempts}) para {entrada_actual}")
            time.sleep(2)
        print("Fallo al encontrar exp.xlsx. Finalizando.")
        sys.exit()

    def ejecutar_comando_con_reintentos(entrada, nombre_archivo, nombre_codigo, foo):

        if not entrada or entrada.isspace():
            print(f"Error: Entrada vacía o inválida: '{entrada}'")
            return False
        
        print(f"Ejecutando comando con entrada: '{entrada}'")
        exito = False
        max_attempts = 3

        for attempt in range(max_attempts):

            try:
                if nombre_codigo == "en.py":
                    subprocess.run(["python", nombre_codigo, entrada, foo], check=True, text=True)
                elif nombre_codigo == "m1_alter.py":
                    subprocess.run(["python", nombre_codigo, entrada], check=True, text=True)
                time.sleep(3)
                reescribir_xslx(nombre_archivo, entrada)
                exito = True
                break
            except subprocess.CalledProcessError as e:
                print(f"Error ejecutando (intento {attempt+1}/{max_attempts}): {e}")
            except Exception as e:
                print(f"Error inesperado (intento {attempt+1}/{max_attempts}): {e}")
        return exito

    def read_values_from_xlsx(filename):
        wb = openpyxl.load_workbook(filename)
        sheet = wb.active
        values = []

        for row in sheet.iter_rows(min_row=2, max_row=50, min_col=3, values_only=True):
            for cell in row:
                if cell is not None:
                    try:
                        values.append(float(cell))
                    except (TypeError, ValueError):
                        print(f"Advertencia: Valor no numérico en {filename}: {cell}")
        return values

    def float_to_bin(number, precision=100):

        if number < 0:
            return '-' + float_to_bin(-number, precision)
        integer_part = int(number)
        fractional_part = number - integer_part
        binary_integer_part = bin(integer_part).replace('0b', '')
        binary_fractional_part = []

        while precision:
            fractional_part *= 2
            bit = int(fractional_part)
            if bit == 1:
                fractional_part -= bit
                binary_fractional_part.append(1)
            else:
                binary_fractional_part.append(0)
            precision -= 1
        binary_fractional_part = ''.join(str(bit) for bit in binary_fractional_part)
        return f"{binary_integer_part}.{binary_fractional_part}"

    def print_salida(DPf, LPf):
        print(f"La probabilidad por aproximación diferencial es: {DPf}")
        print(f"La probabilidad por aproximación lineal (PAL) es: {LPf}")

    if tipo == "1" or tipo == "3": #Calculo DPf

        bin_x1 = str_to_bin(x_1)
        bin_x2 = str_to_bin(x_2)

        int_x1 = int(bin_x1, 2)
        int_x2 = int(bin_x2, 2)
        int_dx = int_x2 - int_x1
        bin_dx = bin(int_dx & ((1 << len(bin_x1)) - 1))[2:].zfill(len(bin_x1))
        int_x_plus_dx = (int_x1 + int_dx) % (2 ** len(bin_x1))
        bin_x_plus_dx = bin(int_x_plus_dx)[2:].zfill(len(bin_x1))

        print(f"Binario de x_1: {bin_x1}")
        print(f"Binario de x_2: {bin_x2}")
        print(f"Diferencia dx: {bin_dx}")
        print(f"x_1 + dx: {bin_x_plus_dx}")
        print(f"x_1 + dx en str: '{bin_to_str(bin_x_plus_dx)}'")
        print(f"dx en str: '{bin_to_str(bin_dx)}'")

        # Ejecutar comandos
        if not ejecutar_comando_con_reintentos(x_1, "exp_input_1.xlsx", nombre_codigo, foo):
            return 0, 0
        if not ejecutar_comando_con_reintentos(x_2, "exp_input_2.xlsx", nombre_codigo, foo):
            return 0, 0
        if not ejecutar_comando_con_reintentos(bin_to_str(bin_x_plus_dx), "exp_input_diferencia_x_dx.xlsx", nombre_codigo, foo):
            return 0, 0
        if not ejecutar_comando_con_reintentos(bin_to_str(bin_dx), "exp_input_dx.xlsx", nombre_codigo, foo):
            return 0, 0

        # Leer valores
        values_1 = read_values_from_xlsx("exp_input_1.xlsx")
        values_2 = read_values_from_xlsx("exp_input_2.xlsx")
        values_x_diferencia_x_dx = read_values_from_xlsx("exp_input_diferencia_x_dx.xlsx")
        values_dx = read_values_from_xlsx("exp_input_dx.xlsx")

        # Alinear longitudes
        min_length = min(len(values_1), len(values_2), len(values_x_diferencia_x_dx), len(values_dx))
        values_1 = values_1[:min_length]
        values_2 = values_2[:min_length]
        values_x_diferencia_x_dx = values_x_diferencia_x_dx[:min_length]
        values_dx = values_dx[:min_length]

        # Calcular diferencias
        differences_fdx = [a - b for a, b in zip(values_1, values_x_diferencia_x_dx)]

        # Contar coincidencias: f(x1) - f(x+dx) == f(dx)
        contador_coincidencias = 0
        umbral_igualdad = 1e-2
        for diff_fdx, val_dx in zip(differences_fdx, values_dx):
            if abs(diff_fdx - val_dx) < umbral_igualdad:
                contador_coincidencias += 1

        # Normalizar por 2^n
        DP_f = contador_coincidencias / (2 ** n_bits) if min_length > 0 else 0
        print(f"Debug DPf: contador={contador_coincidencias}, min_length={min_length}, 2^{n_bits}={2**n_bits}")

    else:
        DP_f = "--NO CALCULADO (Tipo 2)--"

    if tipo == "2" or tipo == "3": #Calculo PAL
        def obtener_datos_xlsx(archivo_xlsx):
            df = pd.read_excel(archivo_xlsx, header=None)
            datos = []

            for col in range(2, df.shape[1], 2):
                columna_x = df.iloc[1:101, col].astype(float)
                columna_y = df.iloc[1:101, col + 1].astype(float)
                datos.append((columna_x, columna_y))
            return datos

        def calcular_correlacion(datos):
            correlaciones = []

            for i in range(len(datos)):
                for j in range(i+1, len(datos)):
                    corr_x = datos[i][0].corr(datos[j][0])
                    corr_y = datos[i][1].corr(datos[j][1])
                    if pd.isna(corr_x) or pd.isna(corr_y):
                        continue
                    correlaciones.append((corr_x, corr_y))
            return correlaciones

        def calcular_PAL(correlaciones, umbral, n_bits):
            contador = 0

            for corr_x, corr_y in correlaciones:
                if abs(corr_x) >= umbral and abs(corr_y) >= umbral:
                    contador += 1
            PAL = (contador / (2**(n_bits-1))) ** 2
            print(f"Debug PAL: contador={contador}, divisor=2^({n_bits}-1)={2**(n_bits-1)}")
            return PAL

        archivo_xlsx = "exp_input_1.xlsx"
        archivo_xlsx2 = "exp_input_2.xlsx"

        datos = obtener_datos_xlsx(archivo_xlsx)
        datos2 = obtener_datos_xlsx(archivo_xlsx2)
        umbral = 0  # Umbral especificado

        correlaciones = calcular_correlacion(datos)
        correlaciones2 = calcular_correlacion(datos2)

        PAL = calcular_PAL(correlaciones, umbral, n_bits)
        PAL2 = calcular_PAL(correlaciones2, umbral, n_bits)
        PAL_combinada = PAL * PAL2

    else:
        PAL_combinada = "--NO CALCULADO (Tipo 1)--"

    print_salida(DP_f, PAL_combinada)
    return DP_f, PAL_combinada


# Definir las iteraciones
iteraciones = [
    {"nombre_codigo": "en.py", "n": 1, "n_bits": 8, "tipo": "3", "foo": "foo.npz"},
    {"nombre_codigo": "en.py", "n": 2, "n_bits": 16, "tipo": "3", "foo": "foo.npz"},
    {"nombre_codigo": "m1_alter.py", "n": 1, "n_bits": 8, "tipo": "3", "foo": None},
    {"nombre_codigo": "m1_alter.py", "n": 2, "n_bits": 16, "tipo": "3", "foo": None}]

# Procesar cada iteración
for idx, config in enumerate(iteraciones, 1):
    nombre_codigo = config["nombre_codigo"]
    n = config["n"]
    n_bits = config["n_bits"]
    tipo = config["tipo"]
    foo = config["foo"]

    print("--------------------------------------------------------------------------")
    print(f"Iteración {idx}: {nombre_codigo}, n={n}, n_bits={n_bits}, tipo={tipo}")
    print(f"se analizarán combinaciones de {n} letras")
    print("--------------------------------------------------------------------------")

    characters = list('AB01')
    combinations = itertools.product(characters, repeat=n)
    combinations_list = [''.join(combination) for combination in combinations]

    LISTACOMBINACION = []
    LISTAPAD = []
    LISTAPAL = []

    for i in combinations_list:
        for j in combinations_list:
            comb1 = i
            comb2 = j
            combinatoria = f"{comb1} + {comb2}"
            LISTACOMBINACION.append(combinatoria)
            print("--------------------------------------------------------------------------")
            print(f"Se calculará {comb1} + {comb2}")
            Resultados = ejecutar_codigo(str(comb1), str(comb2), tipo, nombre_codigo, n_bits, foo)
            LISTAPAD.append(Resultados[0])
            LISTAPAL.append(Resultados[1])

    df = pd.DataFrame({
        'Combinación': LISTACOMBINACION,
        'PAD': LISTAPAD,
        'PAL': LISTAPAL
    })

    output_file = f'{n_bits}bits_TIPO{tipo}-{nombre_codigo}.xlsx'
    df.to_excel(output_file, index=False)
    print(f"Resultados guardados en {output_file}")
    print("--------------------------------------------------------------------------")

print("-----------------------------CÁLCULO TERMINADO----------------------------")
print("--------------------------------------------------------------------------")