import cv2
import numpy as np
import os

def generar_netlist(ruta_imagen, ruta_salida):
    # Cargar imagen en escala de grises
    img = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"No se pudo cargar la imagen: {ruta_imagen}")
    
    # Normalizar la imagen de 0 (negro) a 1 (blanco)
    img_norm = img / 255.0
    M, N = img_norm.shape
    print(f"Generando netlist para red de {M}x{N} nodos...")

    # Rutas de librerias (Ajusta estas rutas a tu entorno local)
    lib_ngspice = "C:/Program Files/Qucs-S/share/qucs-s/xspice_cmlib/include/ngspice_mathfunc.inc"
    lib_mn = "./Mn.lib"
    lib_mp = "./Mp.lib"

    with open(ruta_salida, 'w') as f:
        f.write(f"* Generador de Netlist para Red Memristiva {M}x{N}\n")
        f.write(f'.INCLUDE "{lib_ngspice}"\n')
        f.write(f'.INCLUDE "{lib_mn}"\n')
        f.write(f'.INCLUDE "{lib_mp}"\n\n')

        f.write("* --- Estimulos de entrada y resistencias de inyeccion ---\n")
        # Nodos inyectores (Pixeles de la imagen)
        k = 0
        for r in range(1, M + 1):
            for c in range(1, N + 1):
                val_pixel = img_norm[r-1, c-1]
                f.write(f"R_in_{r}_{c} _net_in_{r}_{c} N{r}_{c}  50 tc1=0.0 tc2=0.0 \n")
                f.write(f"V_in_{r}_{c} _net_in_{r}_{c}  0 PWL(0 0 1p {val_pixel:.6f})\n")
                k += 1

        f.write("\n* --- Conexiones Memristivas Anti-Serie ---\n")
        
        # Enlaces Horizontales (M filas, N-1 columnas)
        f.write("* --- Enlaces Horizontales ---\n")
        for r in range(1, M + 1):
            for c in range(1, N):
                nodo_izq = f"N{r}_{c}"
                nodo_der = f"N{r}_{c+1}"
                f.write(f"XX_H_INV_{r}_{c}  {nodo_izq} NdH_A_{r}_{c} HPM_CHARGE_MN \n")
                f.write(f"XX_H_NORM_{r}_{c} NdH_B_{r}_{c} {nodo_der} HPM_CHARGE_MP \n")
                # Amperimetro Dummy
                f.write(f"VIM_H_{r}_{c} NdH_A_{r}_{c} NdH_B_{r}_{c} DC 0\n")

        # Enlaces Verticales (M-1 filas, N columnas)
        f.write("\n* --- Enlaces Verticales ---\n")
        for r in range(1, M):
            for c in range(1, N + 1):
                nodo_sup = f"N{r}_{c}"
                nodo_inf = f"N{r+1}_{c}"
                f.write(f"XX_V_INV_{r}_{c}  {nodo_sup} NdV_A_{r}_{c} HPM_CHARGE_MN \n")
                f.write(f"XX_V_NORM_{r}_{c} NdV_B_{r}_{c} {nodo_inf} HPM_CHARGE_MP \n")
                # Amperimetro Dummy
                f.write(f"VIM_V_{r}_{c} NdV_A_{r}_{c} NdV_B_{r}_{c} DC 0\n")

        # Bloque de Control Ngspice
        #f.write(".options method=trapezoidal maxord=1 maxstep=0.0001 \n")  #agregado para obligar a manejar los mismos pasos
        f.write("\n.control\n")
        f.write("tran 0.0001 0.061 0 \n")
        
        
        vars_to_plot = []

        f.write("\n* --- Calculo de Memristancias (Ramas Horizontales) ---\n")
        for r in range(1, M + 1):
            for c in range(1, N):
                nodo_izq = f"N{r}_{c}"
                nodo_der = f"N{r}_{c+1}"
                f.write(f"let VH_{r}_{c} = abs({nodo_izq} - {nodo_der})\n")
                f.write(f"let MemH_{r}_{c} = (VH_{r}_{c}) / (abs(i(VIM_H_{r}_{c})) + 1e-16)\n")
                vars_to_plot.append(f"MemH_{r}_{c}")

        f.write("\n* --- Calculo de Memristancias (Ramas Verticales) ---\n")
        for r in range(1, M):
            for c in range(1, N + 1):
                nodo_sup = f"N{r}_{c}"
                nodo_inf = f"N{r+1}_{c}"
                f.write(f"let VV_{r}_{c} = abs({nodo_sup} - {nodo_inf})\n")
                f.write(f"let MemV_{r}_{c} = (VV_{r}_{c}) / (abs(i(VIM_V_{r}_{c})) + 1e-16)\n")
                vars_to_plot.append(f"MemV_{r}_{c}")

        # Escribir el comando de guardado
        f.write("\n* --- Exportacion de datos ---\n")
        #linea_vars = " ".join(vars_to_plot)
        #f.write(f"write spice4qucs.tr1.plot {linea_vars}\n")
        f.write(f"write spice4qucs.tr1.plot all\n")
        
        f.write("destroy all\n")
        f.write("reset\n")
        f.write("exit\n")
        f.write(".endc\n")
        f.write(".END\n")
        
    print(f"Netlist generado exitosamente en: {ruta_salida}")

# Ejecucion del generador
if __name__ == "__main__":
    # Asegurate de tener tu imagen 'img.png' en el mismo directorio o proporciona la ruta completa
    imagen_entrada = "test.png" 
    archivo_netlist = "red.cir"
    
    try:
        generar_netlist(imagen_entrada, archivo_netlist)
    except Exception as e:
        print(f"Error: {e}")