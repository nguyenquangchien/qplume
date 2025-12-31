# QPlume #

Python-based code to simulate submerged, buoyant jet flow ("plumes") using quadtree grids. 

### Purpose

* Hydrodynamic model development: involving fluid flow, advection and diffusion processes
* Numerical implementation: a test framework for quadtree grid development.

### Code

* Set-up: Just copy the folder `src` to your computer system.
* Configuration: Adjust the parameters as needed, especially those `CAPITALISED` variables in the source code.
* Dependencies: `matplotlib`, `numpy`, `scipy`.
* Execution: Run Python code from the source directory: `python simulate.py`
* Interaction: When prompted, enter the scenario code, e.g. `A1`, `B1`, etc. Or, enter `CUSTOM` to input your own parameters.
* Logging: For each time interval, the current time is printed out to the screen, together with the number of cells in U-grid and C-grid.
* Visualization: A color plot of concentration distribution on the C-grid, and a vector plot of flow velocity on the U-grid.

### Development 

* More validation against experiments
* Code refactoring.
