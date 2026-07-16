import numpy as np
import cv2
import os
import time
from numba import njit, prange
import glob



# ==============================================================================
# Section 1: Core (JIT) - Charge controlled Model
# ==============================================================================

@njit(fastmath=True)
def calculate_memristance_charge(q, eta, X0, Ron, Roff, kappa):
    """
    Traducción directa del modelo Maple a Python (Ohms).
    MN: Polaridad negativa (eta = -1)
    MP: Polaridad positiva (eta = 1)
    """
    dR = Roff - Ron
    exp_p = np.exp(4.0 * kappa * q)
    exp_p2 = np.exp(8.0 * kappa * q) # exp(8kq) es exp(4kq)^2
    
    exp_m = np.exp(-4.0 * kappa * q)
    exp_m2 = np.exp(-8.0 * kappa * q)

    if eta == -1:  # Modelo MN
        if q <= 0.0:
            val = (X0 - 1.0) * (X0 - 2.0) * dR * exp_p - ((X0 - 1.0)**2) * dR * exp_p2 + Ron
        else:
            val = dR * (X0**2) * exp_m2 - dR * X0 * (X0 + 1.0) * exp_m + Roff
            
    else:  # Modelo MP (eta == 1)
        if q <= 0.0:
            val = -dR * X0 * (X0 + 1.0) * exp_p + dR * (X0**2) * exp_p2 + Roff
        else:
            val = -((X0 - 1.0)**2) * dR * exp_m2 + (X0 - 1.0) * (X0 - 2.0) * dR * exp_m + Ron

    return val

@njit(parallel=True, fastmath=True)
def simulation_step(u_curr, u_next, Mp, Mq, Mp1, Mp2, Mq1, Mq2, 
                    Pimagnoise, Rin, Roff, Ron, kappa, dt, X0, qp, qq, qp2, qq2):
    
    M, N = u_curr.shape
    inv_Rin = 1.0 / Rin
    
    # --- STEP 1: Get new voltaje ---
    for i in prange(M):
        for j in range(N):
            numerator = Pimagnoise[i, j] * inv_Rin
            denominator = inv_Rin
            
            if i > 0:
                g = 1.0 / Mp[i-1, j]
                numerator += u_curr[i-1, j] * g
                denominator += g
            if i < M - 1:
                g = 1.0 / Mp[i, j]
                numerator += u_curr[i+1, j] * g
                denominator += g
            if j > 0:
                g = 1.0 / Mq[i, j-1]
                numerator += u_curr[i, j-1] * g
                denominator += g
            if j < N - 1:
                g = 1.0 / Mq[i, j]
                numerator += u_curr[i, j+1] * g
                denominator += g
                
            u_next[i, j] = numerator / denominator

    # --- step 2: update memrsitors (BACKWARD EULER) ---
    for i in prange(M):
        for j in range(N):
            
            # --- Vertical (Mp) ---
            if i < M - 1:
                v_drop = u_curr[i+1, j] - u_curr[i, j]
                
                sum_M = Mp1[i, j] + Mp2[i, j]
                v1 = v_drop * (Mp1[i, j] / sum_M)
                v2 = v_drop - v1

                I1 = v1 / Mp1[i, j]
                I2 = v2 / Mp2[i, j]
                
                qp[i, j] += dt * I1
                qp2[i, j] += dt * I2
                
                Mp1[i, j] = calculate_memristance_charge(qp[i, j], -1, X0, Ron, Roff, kappa)
                Mp2[i, j] = calculate_memristance_charge(qp2[i, j], 1, X0, Ron, Roff, kappa)
                Mp[i, j] = Mp1[i, j] + Mp2[i, j]

            # --- Horizontal (Mq) ---
            if j < N - 1:
                v_drop = u_curr[i, j+1] - u_curr[i, j]
                
                sum_M = Mq1[i, j] + Mq2[i, j]
                
                v1 = v_drop * (Mq1[i, j] / sum_M)
                v2 = v_drop - v1
                
                I1 = v1 / Mq1[i, j]
                I2 = v2 / Mq2[i, j]
                
                qq[i, j] += dt * I1
                qq2[i, j] += dt * I2
                
                Mq1[i, j] = calculate_memristance_charge(qq[i, j], -1, X0, Ron, Roff, kappa)
                Mq2[i, j] = calculate_memristance_charge(qq2[i, j], 1, X0, Ron, Roff, kappa)
                Mq[i, j] = Mq1[i, j] + Mq2[i, j]

    # --- step 3: next cycle ---
    u_curr[:] = u_next[:]


# ==============================================================================
# Section 2: processing, control and numerical exportation  (.npz)
# ==============================================================================

def process_memristive_grid(image_path, output_dir):
    # Parámetros base
    Rinit = 1.1; alpha_param = 500; Roff = 2 * alpha_param * Rinit
    Ron = 1.0; beta_param = 22; Rin = Roff / beta_param
    mu = 1e-14; Delta = 10e-9
    
    
    X0 = (Roff - Rinit) / (Roff - Ron)
    kappa = (mu * Ron) / (Delta**2 )
    
    tfin = 0.061 
    dt = 0.0001
    tn = int(tfin / dt)
    
    img = cv2.imread(image_path)
    if img is None: return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    Pimagnoise = cv2.normalize(gray.astype('float64'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    
    M, N = Pimagnoise.shape
    
    
    u_curr = np.zeros((M, N), dtype=np.float64)
    u_next = np.empty_like(u_curr)
    
    Mp1 = np.full((M-1, N), Rinit, dtype=np.float64); Mp2 = np.full((M-1, N), Rinit, dtype=np.float64)
    Mp = Mp1 + Mp2; qp = np.zeros((M-1, N), dtype=np.float64)
    qp2 = np.zeros((M-1, N), dtype=np.float64)
    
    Mq1 = np.full((M, N-1), Rinit, dtype=np.float64); Mq2 = np.full((M, N-1), Rinit, dtype=np.float64)
    Mq = Mq1 + Mq2; qq = np.zeros((M, N-1), dtype=np.float64)
    qq2 = np.zeros((M, N-1), dtype=np.float64)
    

    historial_time = np.linspace(0, tfin, tn, endpoint=False, dtype=np.float64)
    historial_Mp = np.zeros((tn, M-1, N), dtype=np.float32)
    historial_Mq = np.zeros((tn, M, N-1), dtype=np.float32)


    base_name = os.path.basename(image_path).split('.')[0]
    history_dir = os.path.join(output_dir, f"historial_{base_name}")
    os.makedirs(history_dir, exist_ok=True)
    
    porcMoff = 2.0
    Mum = (porcMoff / 100.0) * Roff

    print(f"Star simulation... {os.path.basename(image_path)}")
    start_time = time.time()
    

    for ts in range(tn):
        simulation_step(u_curr, u_next, Mp, Mq, Mp1, Mp2, Mq1, Mq2,
                        Pimagnoise, Rin, Roff, Ron, kappa, dt, X0, qp, qq, qp2, qq2)
        

        historial_Mp[ts, :, :] = Mp
        historial_Mq[ts, :, :] = Mq
        

        edge_map = np.ones((M, N), dtype=np.uint8) * 255
        
        mask_p = Mp >= Mum
        mask_q = Mq >= Mum        
        Mp_pad = np.pad(mask_p, ((1, 1), (0, 0)), mode='constant')
        Mq_pad = np.pad(mask_q, ((0, 0), (1, 1)), mode='constant')
        
        is_edge = (Mp_pad[:-1, :] | Mp_pad[1:, :] | Mq_pad[:, :-1] | Mq_pad[:, 1:])
        edge_map[is_edge] = 0
        
        iter_path = os.path.join(history_dir, f"iter_{ts:04d}.jpg")
        cv2.imwrite(iter_path, edge_map)

    tiempo_sim = time.time() - start_time
    print(f"Simulation time: {tiempo_sim:.4f}s")
    print(f"Historical img saves in: {history_dir}")
    
    # Guardar el resultado final en la carpeta principal de output
    final_path = os.path.join(output_dir, f"Final_Chague_{os.path.basename(image_path)}")
    cv2.imwrite(final_path, edge_map)

    # --------------------------------------------------------------------------
    # MODIFICACIÓN 3: Exportación comprimida congruente con NGSpice
    # Guardamos los tensores 3D en un archivo binario nativo optimizado.
    # --------------------------------------------------------------------------
    npz_path = os.path.join(output_dir, f"netlist_data_{base_name}.npz")
    print(f"Generate memristor states historial...")
    np.savez_compressed(
        npz_path,
        time=historial_time,
        MemV=historial_Mp,  # MemV equivale a Mp (enlaces verticales M-1 x N)
        MemH=historial_Mq,  # MemH equivale a Mq (enlaces horizontales M x N-1)
        shape_grid=np.array([M, N]),
        parameters=np.array([Rin, Ron, Roff, kappa, X0])
    )
    print(f"successful! file: {npz_path}\n")


if __name__ == "__main__":
    input_folder = "input"  
    output_folder = "output"
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    extensions = ['*.jpg', '*.png']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(input_folder, ext)))
        
    if not files:
        print("Images no found!. Create a random image for probe the grid...")
        dummy = np.random.randint(0, 255, (10, 10), dtype=np.uint8)
        cv2.imwrite(os.path.join(input_folder, "dummy_test.jpg") if os.path.exists(input_folder) else "dummy_test.jpg", dummy)
        files = ["dummy_test.jpg" if not os.path.exists(input_folder) else os.path.join(input_folder, "dummy_test.jpg")]

    for f in files:
        process_memristive_grid(f, output_folder)