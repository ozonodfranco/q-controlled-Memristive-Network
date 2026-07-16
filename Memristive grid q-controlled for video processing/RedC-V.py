import numpy as np
import cv2
import os
import time
from numba import njit, prange

# ==============================================================================
# SECTION 1: MATHEMATICAL CORE (JIT) - FLUX-CONTROLLED MODEL
# ==============================================================================

@njit(fastmath=True)
def calculate_memristance_charge(q, eta, X0, Ron, Roff, kappa):
    """
    Direct translation of the Maple model to Python (Ohms).
    MN: Negative polarity (eta = -1)
    MP: Positive polarity (eta = 1)
    """
    dR = Roff - Ron
    exp_p = np.exp(4.0 * kappa * q)
    exp_p2 = np.exp(8.0 * kappa * q) 
    
    exp_m = np.exp(-4.0 * kappa * q)
    exp_m2 = np.exp(-8.0 * kappa * q)

    if eta == -1:  # MN Model
        if q <= 0.0:
            val = (X0 - 1.0) * (X0 - 2.0) * dR * exp_p - ((X0 - 1.0)**2) * dR * exp_p2 + Ron
        else:
            val = dR * (X0**2) * exp_m2 - dR * X0 * (X0 + 1.0) * exp_m + Roff
            
    else:  # MP Model (eta == 1)
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
    
    # --- STEP 1: CALCULATE NEW VOLTAGE ---
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

    # --- STEP 2: UPDATE MEMRISTORS (BACKWARD EULER) ---
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

    # --- STEP 3: PREPARE NEXT CYCLE ---
    u_curr[:] = u_next[:]


# ==============================================================================
# SECTION 2: FRAME PROCESSING AND VIDEO CONTROL
# ==============================================================================

def initialize_grid_states(M, N, Rinit):
    """Pre-allocate and initialize all grid matrices for a given resolution."""
    u_curr = np.zeros((M, N), dtype=np.float64)
    u_next = np.empty_like(u_curr)
    
    Mp1 = np.full((M-1, N), Rinit, dtype=np.float64)
    Mp2 = np.full((M-1, N), Rinit, dtype=np.float64)
    Mp = Mp1 + Mp2
    qp = np.zeros((M-1, N), dtype=np.float64)
    qp2 = np.zeros((M-1, N), dtype=np.float64)
    
    Mq1 = np.full((M, N-1), Rinit, dtype=np.float64)
    Mq2 = np.full((M, N-1), Rinit, dtype=np.float64)
    Mq = Mq1 + Mq2
    qq = np.zeros((M, N-1), dtype=np.float64)
    qq2 = np.zeros((M, N-1), dtype=np.float64)
    
    return u_curr, u_next, Mp, Mq, Mp1, Mp2, Mq1, Mq2, qp, qq, qp2, qq2

def reset_grid_states(u_curr, Mp1, Mp2, qp, qp2, Mq1, Mq2, qq, qq2, Rinit):
    """Reset grid states to initial conditions for the next frame without reallocating memory."""
    u_curr.fill(0.0)
    Mp1.fill(Rinit); Mp2.fill(Rinit)
    qp.fill(0.0); qp2.fill(0.0)
    Mq1.fill(Rinit); Mq2.fill(Rinit)
    qq.fill(0.0); qq2.fill(0.0)

def process_single_frame(gray_frame, states, params):
    """
    Processes a single grayscale frame through the memristive grid.
    Returns the final edge map (binary image).
    """
    u_curr, u_next, Mp, Mq, Mp1, Mp2, Mq1, Mq2, qp, qq, qp2, qq2 = states
    
    # FIX: Added Rinit to the unpacking (9 elements total now)
    Rin, Roff, Ron, kappa, dt, X0, tn, Mum, Rinit = params 
    
    # Normalize input frame
    Pimagnoise = cv2.normalize(gray_frame.astype('float64'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    
    # Reset states for the new frame
    # FIX: Use the unpacked Rinit directly instead of the wrong index params[-3]
    reset_grid_states(u_curr, Mp1, Mp2, qp, qp2, Mq1, Mq2, qq, qq2, Rinit)
    
    # Run simulation steps (No intermediate saving)
    for _ in range(tn):
        simulation_step(u_curr, u_next, Mp, Mq, Mp1, Mp2, Mq1, Mq2,
                        Pimagnoise, Rin, Roff, Ron, kappa, dt, X0, qp, qq, qp2, qq2)
    
    # Edge extraction for the final state
    M, N = gray_frame.shape
    edge_map = np.ones((M, N), dtype=np.uint8) * 255
    
    mask_p = Mp >= Mum
    mask_q = Mq >= Mum
    
    Mp_pad = np.pad(mask_p, ((1, 1), (0, 0)), mode='constant')
    Mq_pad = np.pad(mask_q, ((0, 0), (1, 1)), mode='constant')
    
    is_edge = (Mp_pad[:-1, :] | Mp_pad[1:, :] | Mq_pad[:, :-1] | Mq_pad[:, 1:])
    edge_map[is_edge] = 0
    
    return edge_map

if __name__ == "__main__":
    # --- CONFIGURATION ---
    input_video_path = "video.mp4"   # Change this to your video path
    output_video_path = "output_video.mp4"
    separated_frames_dir = "separated_frames"
    
    # Simulation Parameters
    Rinit = 1.1; alpha_param = 500; Roff = 2 * alpha_param * Rinit
    Ron = 1.0; beta_param = 22; Rin = Roff / beta_param
    mu = 1e-14; Delta = 10e-9
    
    X0 = (Roff - Rinit) / (Roff - Ron)
    kappa = (mu * Ron) / (Delta**2)
    
    tfin = 0.038 
    dt = 0.0001
    tn = int(tfin / dt)
    
    porcMoff = 2.0
    Mum = (porcMoff / 100.0) * Roff

    # Create output directories
    os.makedirs(separated_frames_dir, exist_ok=True)
    
    # --- VIDEO READING SETUP ---
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_video_path}")
        exit()
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video loaded: {width}x{height} @ {fps} FPS | Total frames: {total_frames}")
    
    # --- VIDEO WRITING SETUP ---
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height), isColor=False)
    
    # Pre-allocate memory for the grid based on the first frame's resolution
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        exit()
        
    first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    M, N = first_gray.shape
    
    print("Initializing memristive grid states...")
    states = initialize_grid_states(M, N, Rinit)
    params = (Rin, Roff, Ron, kappa, dt, X0, tn, Mum, Rinit)
    
    print("Starting video processing...\n")
    start_time = time.time()
    
    frame_idx = 0
    
    # --- PROCESS FIRST FRAME ---
    frame_start_time = time.time()
    processed_frame = process_single_frame(first_gray, states, params)
    frame_process_time = time.time() - frame_start_time
    
    cv2.imwrite(os.path.join(separated_frames_dir, f"frame_{frame_idx:05d}.jpg"), processed_frame)
    out_video.write(processed_frame)
    
    frame_idx += 1
    print(f"Frame {frame_idx}/{total_frames} processed in {frame_process_time:.2f} seconds")
    
    # --- PROCESS REMAINING FRAMES ---
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Measure time for this specific frame
        frame_start_time = time.time()
        
        # Process frame
        processed_frame = process_single_frame(gray, states, params)
        
        # Calculate elapsed time for this frame
        frame_process_time = time.time() - frame_start_time
        
        # Save separated frame
        frame_path = os.path.join(separated_frames_dir, f"frame_{frame_idx:05d}.jpg")
        cv2.imwrite(frame_path, processed_frame)
        
        # Write to output video
        out_video.write(processed_frame)
        
        frame_idx += 1
        
        # Print progress with exact time per frame
        print(f"Frame {frame_idx}/{total_frames} processed in {frame_process_time:.2f} seconds")

    # --- CLEANUP ---
    cap.release()
    out_video.release()
    
    total_time = time.time() - start_time
    print(f"\nProcessing complete!")
    print(f"Total time: {total_time:.2f}s | Average per frame: {total_time/total_frames:.4f}s")
    print(f"Output video saved at: {output_video_path}")
    print(f"Separated frames saved in: {separated_frames_dir}/")