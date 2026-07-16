import os
import re
import cv2
import numpy as np
import polars as pl
import matplotlib.pyplot as plt

# Graphic style for scientific publications (IEEE / JCR)
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'lines.linewidth': 2.0,
    'grid.linestyle': '--',
    'grid.alpha': 0.6
})

def calcular_fom_pratt(borde_py, borde_spice, alpha=1.0/9.0):
    """
    Calculates Pratt's Figure of Merit (FOM) between two binary edge maps.
    borde_py, borde_spice: Boolean or uint8 arrays where True/255 represents an edge.
    """
    N_c = np.sum(borde_py > 0)
    N_r = np.sum(borde_spice > 0)
    
    # Base cases for extreme convergence
    if N_c == 0 and N_r == 0:
        return 100.0  # Both engines perfectly agree that the background is empty
    if N_c == 0 or N_r == 0:
        return 0.0    # Total disagreement: one sees edges and the other does not
        
    # Create distance map from the reference edge (NGSpice)
    ref_invertida = np.where(borde_spice > 0, 0, 255).astype(np.uint8)
    mapa_distancias = cv2.distanceTransform(ref_invertida, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    
    # Extract distances d(i) only at coordinates where Python detected an edge
    distancias_di = mapa_distancias[borde_py > 0]
    
    # Apply Pratt's summation
    sumatoria_pratt = np.sum(1.0 / (1.0 + alpha * (distancias_di ** 2)))
    
    # Normalize by the maximum number of elements and convert to percentage
    fom = (sumatoria_pratt / max(N_c, N_r)) * 100.0
    return fom

def cargar_ngspice_rapido(ruta_plot):
    """Loads and parses the ngspice ASCII .plot file using Polars."""
    print(f"[1/4] Reading SPICE file: {ruta_plot}...")
    with open(ruta_plot, 'r', encoding='latin1') as f:
        lines = f.readlines()
        
    variables = []
    num_vars = 0
    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        if line.startswith("No. Variables:"):
            num_vars = int(line.split(":")[1].strip())
        elif line.startswith("Variables:"):
            idx += 1
            for _ in range(num_vars):
                variables.append(lines[idx].strip().split()[1].lower())
                idx += 1
            continue
        elif line.startswith("Values:"):
            idx += 1
            break
        idx += 1

    raw_data = []
    current_point = []
    for line in lines[idx:]:
        tokens = line.strip().split()
        if not tokens: continue
        val = float(tokens[1]) if (len(tokens) == 2 and tokens[0].isdigit() and not current_point) else float(tokens[0])
        current_point.append(val)
        if len(current_point) == num_vars:
            raw_data.append(current_point)
            current_point = []

    return pl.DataFrame(raw_data, schema=variables, orient="row")

def validar_frameworks(ruta_npz, ruta_plot, t_start=0.001, t_end=0.061, porcMoff=2.0):
    # 1. Load Python data (.npz)
    print(f"[2/4] Loading Numba history: {ruta_npz}...")
    npz_data = np.load(ruta_npz)
    t_py = npz_data['time']
    MemV_py = npz_data['MemV']  # Shape: (tn, M-1, N)
    MemH_py = npz_data['MemH']  # Shape: (tn, M, N-1)
    
    M, N = npz_data['shape_grid']
    Rin, Ron, Roff, kappa, X0 = npz_data['parameters']
    rango_dinamico = Roff - Ron
    Mum = (porcMoff / 100.0) * Roff

    # 2. Load SPICE data and extract matrices by column
    df_spice = cargar_ngspice_rapido(ruta_plot)
    t_spice = df_spice["time"].to_numpy()
    
    # Flatten matrices per iteration to compare component by component
    # Total memristors in the grid: (M-1)*N vertical + M*(N-1) horizontal
    num_memv = (M - 1) * N
    num_memh = M * (N - 1)
    total_mems = num_memv + num_memh
    
    print(f"[3/4] Aligning by interpolation in window t=[{t_start}s, {t_end}s]...")
    
    # Filter Python indices within the time window
    idx_ventana = np.where((t_py >= t_start) & (t_py <= t_end))[0]
    t_eval = t_py[idx_ventana]
    n_steps_eval = len(t_eval)
    
    # Build 2D evaluation matrix: (evaluated_times, total_memristors)
    matriz_py = np.zeros((n_steps_eval, total_mems), dtype=np.float32)
    matriz_spice = np.zeros((n_steps_eval, total_mems), dtype=np.float32)
    
    # Fill flattened Python data for the window
    for k, idx_t in enumerate(idx_ventana):
        matriz_py[k, :num_memv] = MemV_py[idx_t].flatten()
        matriz_py[k, num_memv:] = MemH_py[idx_t].flatten()
        
    # Fill and interpolate SPICE data column by column
    col_idx = 0
    # Vertical links: memv_r_c
    for r in range(1, M):
        for c in range(1, N + 1):
            col_name = f"memv_{r}_{c}"
            val_spice_raw = df_spice[col_name].to_numpy() if col_name in df_spice.columns else np.zeros_like(t_spice)
            matriz_spice[:, col_idx] = np.interp(t_eval, t_spice, val_spice_raw)
            col_idx += 1
            
    # Horizontal links: memh_r_c
    for r in range(1, M + 1):
        for c in range(1, N):
            col_name = f"memh_{r}_{c}"
            val_spice_raw = df_spice[col_name].to_numpy() if col_name in df_spice.columns else np.zeros_like(t_spice)
            matriz_spice[:, col_idx] = np.interp(t_eval, t_spice, val_spice_raw)
            col_idx += 1

    # 3. METRICS CALCULATION (THE PILLARS)
    print("[4/4] Calculating point-by-point metrics...")
    nrmse_temporal = []
    fom_temporal = []
    
    for k in range(n_steps_eval):
        val_p = matriz_py[k]
        val_s = matriz_spice[k]
        
        # Pillar 1: NRMSE (%)
        rmse = np.sqrt(np.mean((val_s - val_p)**2))
        nrmse = (rmse / rango_dinamico) * 100.0
        nrmse_temporal.append(nrmse)
        
        # Pillar 2: Pratt's Figure of Merit (FOM) in edge application
        # For the FOM we need to reconstruct the 2D spatial topology of the memristors
        # We join vertical and horizontal into a single MxN mask
        borde_p_vert = (val_p[:num_memv].reshape(M-1, N) >= Mum)
        borde_p_horiz = (val_p[num_memv:].reshape(M, N-1) >= Mum)
        
        borde_s_vert = (val_s[:num_memv].reshape(M-1, N) >= Mum)
        borde_s_horiz = (val_s[num_memv:].reshape(M, N-1) >= Mum)
        
        # We map the link edges to the MxN node grid
        mask_py = np.zeros((M, N), dtype=np.uint8)
        mask_spice = np.zeros((M, N), dtype=np.uint8)
        
        # If a link exceeds the threshold, we mark its adjacent nodes as edge (255)
        mask_py[:-1, :][borde_p_vert] = 255
        mask_py[1:, :][borde_p_vert] = 255
        mask_py[:, :-1][borde_p_horiz] = 255
        mask_py[:, 1:][borde_p_horiz] = 255
        
        mask_spice[:-1, :][borde_s_vert] = 255
        mask_spice[1:, :][borde_s_vert] = 255
        mask_spice[:, :-1][borde_s_horiz] = 255
        mask_spice[:, 1:][borde_s_horiz] = 255
        
        fom = calcular_fom_pratt(mask_py, mask_spice)
        fom_temporal.append(fom)

    # --- NUMERICAL REPORT IN CONSOLE (FOR THE PAPER TABLES) ---
    print("\n" + "="*55)
    print(f" FIGURES OF MERIT SUMMARY (Grid {M}x{N} - Window {t_start}s to {t_end}s)")
    print("="*55)
    print(f" Evaluated points:         {n_steps_eval} time frames")
    print(f" Average NRMSE (Pillar 1): {np.mean(nrmse_temporal):.4f}%  (Target: < 1.0%)")
    print(f" Maximum NRMSE:            {np.max(nrmse_temporal):.4f}%")
    print(f" Average Pratt's FOM:      {np.mean(fom_temporal):.2f}%   (Target: > 90.0%)")
    print("="*55)

    # --- GENERATION OF JCR PLOT ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    
    # Continuous error plot (NRMSE)
    ax1.plot(t_eval * 1000, nrmse_temporal, color='#d62728', label='NRMSE (%)')
    ax1.set_ylabel('NRMSE (%)')
    ax1.set_title('Temporal Evolution of Normalized Root Mean Square Error (NGSpice vs Numba)')
    ax1.grid(True)
    ax1.legend(loc='upper right')
    
    # Edge equivalence plot (Pratt's FOM)
    ax2.plot(t_eval * 1000, fom_temporal, color='#2ca02c', label="Pratt's FOM (%)")
    ax2.set_xlabel('Simulation Time (ms)')
    ax2.set_ylabel("Pratt's FOM (%)")
    ax2.set_ylim(0, 105) # Full range to see spatial stability
    ax2.grid(True)
    ax2.legend(loc='lower right')
    
    plt.tight_layout()
    ruta_grafica = "figura_merito_fom_jcr.png"
    plt.savefig(ruta_grafica, dpi=300, bbox_inches='tight')
    print(f"\nHigh-resolution plot generated and saved at: {ruta_grafica}!")

if __name__ == "__main__":
    # Point to your real files for the 40x40 grid
    archivo_npz = "netlist_data_test.npz" 
    archivo_spice = "spice4qucs.tr1.plot"
    
    try:
        # We evaluate starting from 1 millisecond to skip the initial power-on transient
        validar_frameworks(archivo_npz, archivo_spice, t_start=0.001, t_end=0.061, porcMoff=2.0)
    except Exception as e:
        print(f"Error in validation: {e}")