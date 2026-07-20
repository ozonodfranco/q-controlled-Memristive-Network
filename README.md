# q-controlled Memristive Network as an Edge Detection Video Processor ⚡

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.x](https://img.shields.io/badge/python-3.x-blue.svg)](https://www.python.org/)
[![Simulation: NGSpice](https://img.shields.io/badge/Simulation-NGSpice-red.svg)](http://ngspice.sourceforge.net/)

This repository contains the source code, simulation models, and experimental testbenches accompanying the research paper: **"q-controlled Memristive Network used as an Edge Detection Video Processor"**.

The project demonstrates the use of a high-performance Python solver (accelerated via Numba JIT) to simulate large-scale memristive grids for analog image and video processing. Furthermore, it includes a comprehensive physical circuit validation workflow to prove the algorithm's exact topological and electrical congruence against standard SPICE simulators.

---

## 📂 Repository Structure

The repository is divided into five main modules to ensure the full reproducibility of all static, dynamic, and electrical benchmarking tests presented in the paper:

### 1. `img Test/` (Foundational Test Bench)
Contains the foundational test image (`test.png`) used throughout the research.
* Features simple but strategically placed geometric shapes designed to fundamentally test the network's spatial dynamics, boundary detection capabilities, and corner/edge node degrees.

### 2. `Memristive grid q-controlled for image processing/` (Static Vision Processing)
Contains the core Numba-accelerated Python implementation for static image edge detection.
* **`RedCIMG_HM.py`**: The main script. It reads an input image from the `input/` directory, normalizes pixel intensities to continuous voltage equivalents ($0\text{--}1\text{ V}$), and computes the time-domain evolution of the memristive grid via Backward Euler integration.
* **Outputs generated**:
  * Frame-by-frame transient state images saved inside `output/history_test/`.
  * The final converged binary boundary map saved as `final_chague_test` in the `output/` folder.
  * Exports the continuous memristance states of all horizontal and vertical fuses into `netlist_data_test.npz` for subsequent electrical validation.

### 3. `Memristive grid q-controlled for video processing/` (Dynamic Video Processing)
Extends the mathematical framework to process continuous, high-density video streams frame-by-frame.
* **Video Pipeline**: Processes an input `video.mp4` file and generates the complete edge-detected video stream as `output_video.mp4`.
* **Frame Extraction**: Separates and saves all individual processed spatial frames within the `separated_frames/` directory for detailed visual inspection and temporal verification.

### 4. `Electrical Framework Simulation/` (Physical SPICE Verification)
Housed in a three-stage pipeline, this module executes the physical circuit simulation workflow using **NGSpice** to validate the Python solver against actual nodal hardware solvers.
* **`preprocessing/`**: Contains `GenCirNg.py`, which reads `test.png` and automatically synthesizes a complete SPICE netlist (`red.cir`), mapping pixel intensities to independent voltage sources and configuring anti-series memristive fuses.
* **`processing/`**: The simulation environment where the generated netlist is executed via the NGSpice command-line engine to perform the transient analysis, generating the raw `spice4qucs.tr1.plot` data file.
* **`postprocessing/`**: Contains `PosProNg.py`. This script parses the raw NGSpice transient data, reconstructs the spatial topology, and generates the hardware-solved output images for every time step inside `output_ngspice_frames/` (formatted as `ngspice_iter_XXXX.png`).

### 5. `Electrical Benchmark/` (Quantitative Figures of Merit)
The ultimate validation suite that strictly compares the transient responses from NGSpice (`spice4qucs.tr1.plot`) and the Python solver (`netlist_data_test.npz`). 
* **`ElectricalB.py`**: Analyzes the data and generates numerical reports and graphical plots for two critical validation metrics:
  * **Normalized Root Mean Square Error (NRMSE)**: Evaluates continuous state-variable tracking across all memristors over time. A near-zero NRMSE confirms that our first-order Backward Euler integration faithfully mirrors the second-order adaptive numerical integration of NGSpice without divergence.
  * **Pratt's Figure of Merit (FOM)**: Assesses topological boundary preservation. Standard binary metrics (like IoU/Jaccard) heavily penalize transient 1-pixel spatial displacements caused by minor phase shifts between adaptive SPICE stepping and our fixed-step solver during rapid resistive switching. Pratt's FOM gracefully accounts for these micro-displacements based on Euclidean distance, proving that spatial edge geometry is fully preserved (>93%) in the steady-state visual regime.

---

## 🛠️ Requirements and Installation

To run the scripts in this repository, you will need **Python 3.8+** and the following core libraries:

```bash
pip install numpy scipy numba opencv-python matplotlib polars
