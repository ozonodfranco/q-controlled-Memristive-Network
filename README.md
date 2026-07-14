# q-controlled Memristive Network as an Edge Detection Video Processor

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.x](https://img.shields.io/badge/python-3.x-blue.svg)](https://www.python.org/)
[![Simulation: NGSpice](https://img.shields.io/badge/Simulation-NGSpice-red.svg)](http://ngspice.sourceforge.net/)

This repository contains the official source code, circuit simulation frameworks, and validation datasets for the research paper: **"q-controlled Memristive Network used as an Edge Detection Video Processor"**.

It provides a high-performance Python solver (accelerated via Numba JIT) to simulate large-scale memristive grids for analog image and video processing. Furthermore, it includes a comprehensive electrical validation framework to prove the algorithm's exact topological and electrical congruence against standard SPICE simulators.

---

## 📁 Repository Structure

The repository is organized into five main directories, covering everything from fundamental image testing to full video processing and electrical benchmarking:
```text
📦 q-controlled Memristive Network
 ┣ 📂 img Test
 ┣ 📂 Memristive grid q-controlled for image processing
 ┣ 📂 Memristive grid q-controlled for video processing
 ┣ 📂 Electrical Framework Simulation
 ┗ 📂 Electrical Benchmark
```
1. img Test
Contains the foundational test image (test.png) used throughout the research. This image features simple but strategically placed geometric shapes designed to fundamentally test the network's spatial dynamics and boundary detection capabilities.

2. Memristive grid q-controlled for image processing
Contains the core Python implementation for static image edge detection.

RedCIMG_HM.py: The main script. It reads an input image from the input/ directory and computes the time-domain evolution of the memristive grid.

Outputs:

Generates frame-by-frame output images reflecting the grid's transient state inside output/history_test/.

Saves the final converged boundary map as final_chague_test in the output/ folder.

Exports the continuous memristance states of the horizontal and vertical fuses into netlist_data_test.npz for later electrical validation.

3. Memristive grid q-controlled for video processing
Extends the mathematical framework to process dynamic video streams.

Processes an input video.mp4 file and generates the edge-detected result as output_video.mp4.

Extracts and saves all individual processed frames within the separated_frames/ directory for detailed inspection.

4. Electrical Framework Simulation
This directory houses the physical circuit simulation workflow using NGSpice to validate the Python solver. It is divided into three stages:

preprocessing/: Contains GenCirNg.py, which reads test.png and synthesizes a complete SPICE netlist (red.cir) mapping the pixel intensities to voltage sources.

processing/: The environment where the generated netlist is executed via the NGSpice command line to perform the transient analysis, outputting the spice4qucs.tr1.plot file.

postprocessing/: Contains PosProNg.py. This script parses the raw NGSpice transient response and reconstructs the spatial topology, generating the output images for every time step inside output_ngspice_frames/ (formatted as ngspice_iter_XXXX.png).

5. Electrical Benchmark
The ultimate validation suite. Contains ElectricalB.py, which strictly compares the transient responses from NGSpice (spice4qucs.tr1.plot) and the Python solver (netlist_data_test.npz). It generates both numerical reports and high-quality plots for two critical figures of merit: NRMSE and Pratt's FOM.

📊 Scientific Validation Metrics Context
To rigorously prove that our Python solver acts as an exact numerical surrogate for physical memristive circuits (rather than a mere software approximation), ElectricalB.py evaluates two key metrics:

Normalized Root Mean Square Error (NRMSE):
Evaluates the continuous state-variable tracking across all memristors in the grid over time. A near-zero NRMSE confirms that the first-order Backward Euler integration in our Python framework faithfully mirrors the second-order adaptive numerical integration of NGSpice without divergence.

Pratt's Figure of Merit (FOM):
While NRMSE evaluates the continuous analog state, Pratt's FOM assesses topological boundary preservation. Standard binary metrics (like IoU/Jaccard) heavily penalize the transient 1-pixel spatial displacements caused by minor phase shifts between the adaptive SPICE solver and our fixed-step solver during rapid resistance switching. Pratt's FOM gracefully accounts for these micro-displacements based on Euclidean distance, proving that the spatial edge geometry is fully preserved in the steady-state visual regime.

How to Run
(Provide brief instructions here on how to set up the environment, e.g., required libraries like numpy, numba, opencv-python, polars, and how to execute the main scripts).

Bash
# Example:
pip install -r requirements.txt
cd "Memristive grid q-controlled for image processing"
python RedCIMG_HM.py
Authors: Arturo Sarmiento-Reyes, Juan Manuel Ugalde-Franco

Institution: Instituto Nacional de Astrofísica, Óptica y Electrónica (INAOE), Puebla, Mexico.
