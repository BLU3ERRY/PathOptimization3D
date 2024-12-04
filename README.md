# PathOptimization3D

This project demonstrates the use of Particle Swarm Optimization (PSO) to optimize a racing line for a track. The goal is to determine the optimal path for a vehicle, minimizing the lap time while adhering to track constraints. The optimization process is visualized in real-time, and several utility functions are provided to support the process.

This project is inspired by and based on the work found [here](https://github.com/ParsaD23/Racing-Line-Optimization-with-PSO). Special thanks to the original author for the inspiration and initial guidance!

## Features
- Optimization of racing lines using Particle Swarm Optimization (PSO)
- Real-time visualization of the optimization process
- Interactive plots of the racing line, track boundaries, and sector boundaries
- Functions to compute track boundaries and smooth the reference path

## Project Structure
- **main.py**: The main script for running the optimization. Loads data, sets up plotting, and performs the optimization.
- **utilities.py**: Contains helper functions for computing track boundaries, plotting, and path smoothing.
- **cost_functions.py**: Defines the cost function for lap time and additional constraints.
- **pso_optimization.py**: Implements the Particle Swarm Optimization (PSO) algorithm.
- **constants.py**: Defines various constants used throughout the project (e.g., track width, vehicle parameters).

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/racing-line-optimization-pso.git
   cd racing-line-optimization-pso
   ```

2. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage

To run the optimization process, execute the `main.py` script:

```sh
python main.py
```

Ensure that you have the necessary data files (e.g., `reference_path.csv`) placed in the `Data` directory. The script will load the reference path, compute track boundaries, and run the optimization.

### Input Data
- **reference_path.csv**: A CSV file containing the reference path (center line) coordinates (`x`, `y`, `z`).

## Visualization
- The optimization process will be visualized in real-time with the current racing line updating on the plot.
- After the optimization is complete, the final racing line and boundaries are plotted.
- Lap time history per iteration is also visualized to show the progress of the optimization.

## Dependencies
- Python 3.8+
- NumPy
- Pandas
- Matplotlib
- SciPy

Install dependencies using:
```sh
pip install numpy pandas matplotlib scipy
```

## Acknowledgements
This project was inspired by the [Racing Line Optimization with PSO](https://github.com/ParsaD23/Racing-Line-Optimization-with-PSO) by ParsaD23. It follows similar principles but extends and modifies the implementation to add more visualization and a modular structure.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
