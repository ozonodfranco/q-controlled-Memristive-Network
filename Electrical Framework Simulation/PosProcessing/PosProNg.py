import os
import re
import cv2
import numpy as np
import polars as pl

def parse_ngspice_ascii(raw_path):
    """
    Parsea un archivo RAW/PLOT ASCII de ngspice y lo convierte en un DataFrame de Polars.
    """
    with open(raw_path, 'r', encoding='latin1') as f:
        lines = f.readlines()
    
    variables = []
    num_vars = 0
    num_points = 0
    header_ended = False
    
    # 1. Leer metadatos y encabezado
    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        if line.startswith("No. Variables:"):
            num_vars = int(line.split(":")[1].strip())
        elif line.startswith("No. Points:"):
            num_points = int(line.split(":")[1].strip())
        elif line.startswith("Variables:"):
            idx += 1
            for _ in range(num_vars):
                parts = lines[idx].strip().split()
                # NGSpice guarda: [índice, nombre_variable, tipo]
                variables.append(parts[1].lower()) # ngspice usa minúsculas
                idx += 1
            continue
        elif line.startswith("Values:"):
            idx += 1
            header_ended = True
            break
        idx += 1

    if not header_ended:
        raise ValueError("No se encontró la sección 'Values:' en el archivo.")

    print(f"Leyendo {num_points} puntos temporales para {num_vars} variables...")

    # 2. Extraer el bloque numérico rápidamente
    # En formato ASCII, el primer valor de cada punto tiene el índice del punto delante
    raw_data = []
    current_point = []
    
    for line in lines[idx:]:
        tokens = line.strip().split()
        if not tokens:
            continue
        
        # Si hay más de un token o el primero parece un entero de índice inicial
        if len(tokens) == 2 and tokens[0].isdigit() and len(current_point) == 0:
            val = float(tokens[1])
        else:
            val = float(tokens[0])
            
        current_point.append(val)
        
        if len(current_point) == num_vars:
            raw_data.append(current_point)
            current_point = []

    # 3. Convertir a Polars DataFrame
    df = pl.DataFrame(raw_data, schema=variables, orient="row")
    return df

def get_grid_dimensions(df):
    """
    Detecta automáticamente las dimensiones M (filas) y N (columnas) 
    analizando los nombres de las columnas en Polars.
    """
    cols = df.columns
    max_r, max_c = 0, 0
    
    # Buscar patrones memh_r_c o memv_r_c
    pattern = re.compile(r"mem[hv]_(\d+)_(\d+)")
    for col in cols:
        match = pattern.match(col)
        if match:
            r, c = int(match.group(1)), int(match.group(2))
            if r > max_r: max_r = r
            if c > max_c: max_c = c
            
    # Para memv el máximo r es M-1, para memh el máximo c es N-1
    # Por seguridad sumamos 1 si es necesario, evaluando la topología:
    # MemH va hasta (M, N-1) -> max_r = M, max_c = N-1 -> N = max_c + 1
    # MemV va hasta (M-1, N) -> max_r = M-1 -> M = max_r + 1 (si evaluamos ambos)
    
    # Para mayor precisión:
    m_h = [int(re.match(r"memh_(\d+)_\d+", c).group(1)) for c in cols if c.startswith("memh_")]
    n_v = [int(re.match(r"memv_\d+_(\d+)", c).group(1)) for c in cols if c.startswith("memv_")]
    
    M = max(m_h) if m_h else max_r
    N = max(n_v) if n_v else max_c + 1
    return M, N

def procesar_simulacion_ngspice(ruta_raw, dir_salida, Roff=110000.0, porcMoff=2.0):
    os.makedirs(dir_salida, exist_ok=True)
    
    # 1. Cargar datos con Polars
    df = parse_ngspice_ascii(ruta_raw)
    M, N = get_grid_dimensions(df)
    print(f"Red detectada en simulación: {M}x{N}")
    
    # Umbral para detección de borde
    Mum = (porcMoff / 100.0) * Roff
    
    # 2. Pre-seleccionar columnas de interés para evitar búsquedas en el bucle
    # Polars permite seleccionar con expresiones regulares
    time_series = df["time"].to_numpy()
    num_steps = len(time_series)
    
    # 3. Iterar temporalmente para generar las imágenes
    print(f"Generando imágenes de salida en: {dir_salida} ...")
    
    for ts in range(num_steps):
        # Extraer la fila actual como diccionario para acceso instantáneo
        row_dict = df.row(ts, named=True)
        
        # Construir matrices Mp (Verticales: M-1 x N) y Mq (Horizontales: M x N-1)
        Mp = np.zeros((M - 1, N), dtype=np.float64)
        Mq = np.zeros((M, N - 1), dtype=np.float64)
        
        # Llenar Mq (Horizontales)
        for r in range(1, M + 1):
            for c in range(1, N):
                col_name = f"memh_{r}_{c}"
                Mq[r-1, c-1] = row_dict.get(col_name, 0.0)
                
        # Llenar Mp (Verticales)
        for r in range(1, M):
            for c in range(1, N + 1):
                col_name = f"memv_{r}_{c}"
                Mp[r-1, c-1] = row_dict.get(col_name, 0.0)
        
        # --- LÓGICA DE DETECCIÓN DE BORDE (Idéntica a tu código Numba) ---
        edge_map = np.ones((M, N), dtype=np.uint8) * 255
        
        mask_p = Mp >= Mum
        mask_q = Mq >= Mum
        
        Mp_pad = np.pad(mask_p, ((1, 1), (0, 0)), mode='constant')
        Mq_pad = np.pad(mask_q, ((0, 0), (1, 1)), mode='constant')
        
        is_edge = (Mp_pad[:-1, :] | Mp_pad[1:, :] | Mq_pad[:, :-1] | Mq_pad[:, 1:])
        edge_map[is_edge] = 0
        
        # Guardar imagen del paso actual
        img_filename = os.path.join(dir_salida, f"ngspice_iter_{ts:04d}.jpg")
        cv2.imwrite(img_filename, edge_map)
        
    print(f"¡Procesamiento completo! Se generaron {num_steps} imágenes exitosamente.")

if __name__ == "__main__":
    # Asegúrate de apuntar al archivo generado por ngspice
    archivo_plot = "spice4qucs.tr1.plot"
    carpeta_salida = "output_ngspice_frames"
    
    # Usa exactamente los mismos parámetros Roff que en tu script de Python
    alpha_param = 500
    Rinit = 1.1
    Roff_val = 2 * alpha_param * Rinit  # 1100.0 Ohms
    
    try:
        procesar_simulacion_ngspice(archivo_plot, carpeta_salida, Roff=Roff_val, porcMoff=2.0)
    except Exception as e:
        print(f"Error en el postprocesamiento: {e}")