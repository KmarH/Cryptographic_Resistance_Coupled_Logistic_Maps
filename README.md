# Cryptographic Resistance of the Coupled Simple and Generalized Logistic Maps

Code used for the article "Resistencia Criptográfica del Mapa Logístico Simple y Generalizado con Acoplamiento" (Cryptographic Resistance of the Simple and Generalized Logistic Map with Coupling). School of Physics, University of Costa Rica.

Two chaotic encryption schemes are studied, one built on the coupled simple logistic map (LM) and one on the coupled generalized logistic map (GLM). Their resistance is tested in two ways: a chaotic attack that recovers the parameters and initial conditions of the map with CBPSO (chaotic-based particle swarm optimization, as described by Yuexi Peng, 2020), and a comparative differential and linear cryptanalysis of both schemes.

Before implementing CBPSO, two preliminary steps were done: a basic PSO on the Rosenbrock function, and the recreation of the article by Sheludko (2022) for the simple logistic map, but using an inertia weight with the form of an exponential function. With this, the full CBPSO process of interest for the project was then built.

The plots of the parameters and the chaotic attack were done separately for each type of map, while the comparative differential cryptanalysis was done jointly. The plots and data of the final report come from the `CBPSOResults` and `Graphs` notebooks.

## Structure

The repository has the original Spanish version and an English translation with the same structure.

| ENG | Original_Spanish_version | Content |
|---|---|---|
| `Rosenbrock.ipynb` | `Rosenbrock.ipynb` | Basic PSO on the Rosenbrock function. |
| `Exponential_PSO-Simple.ipynb` | `Exponencial_PSO-Simple.ipynb` | Recreation of Sheludko (2022) for the simple logistic map, with exponential inertia weight. |
| `Simple_Coupled_Map/CBPSOResults_LM.ipynb` | `Logistico_Simple_Acoplado/CBPSOResultados_LM.ipynb` | CBPSO attack on the coupled LM: parameters first, then initial conditions. MSE and average distance over the cycles. |
| `Simple_Coupled_Map/Graphs2025.ipynb` | `Logistico_Simple_Acoplado/Graficas2025.ipynb` | MSE as a function of each parameter, and 3D/contour plots against x0, for the coupled LM. |
| `Generalized_Coupled_Map/CBPSOResults_GLM.ipynb` | `Logistico_Generalizado_Acoplado/CBPSOResultados_GLM.ipynb` | CBPSO attack on the coupled GLM. |
| `Generalized_Coupled_Map/GraphsMSE.ipynb` | `Logistico_Generalizado_Acoplado/GraficasECM.ipynb` | MSE plots for the coupled GLM. |
| `Differential_Cryptanalysis/` | `Criptoanalisis_Diferencial/` | `en.py` (GLM scheme) and `m1_alter.py` (LM scheme) encrypt a message given as argument; `analisis_PAD&PAL.py` and `analisisdyl_4.py` run them over combinations of inputs and compute the differential (PAD) and linear (PAL) approximation probabilities. The `.xlsx` files are the results, with a summary table. |

In `ENG/` only the text was translated: markdown, comments, printed messages and plot labels. Variable and function names, and the saved outputs of the notebooks, are the original ones, so the figures stored inside the notebooks keep their Spanish labels. The data files (`.xlsx`, `foo.npz`) are the same in both folders.

## Versión en español

La versión original en español, con su README, está en `Original_Spanish_version/`.
