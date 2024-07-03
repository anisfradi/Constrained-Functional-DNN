# Neural Network Training for Functional Data using Penalized Geodesic Distance, Tangent space MSE loss with Legendre Polynomials as the basis.  
  
This project involves training neural networks for functional data using the penalized geodesic distance loss and MSE loss on the tangent space with Legendre Polynomials for basis decomposition. 


## Project Structure  
  
.
├── main.py
├── utilities.py
├── geometry_utils.py
├── legendre_utils.py
├── neural_networks.py
├── metrics.py
├── data_loaders.py
├── requirements.txt
└── data/
├── Beta_simulated.csv

  
## Directory Contents  
  
- `main.py`: The main script that ties all the modular components together to perform training and evaluation of neural networks.  
- `utilities.py`: Contains helper functions related to mathematical operations and dataset creation.  
- `geometry_utils.py`: Includes geometric transformation functions.  
- `legendre_utils.py`: Contains Legendre-related functions.  
- `neural_networks.py`: Defines neural network models and training functions.  
- `metrics.py`: Functions for calculating evaluation metrics and losses.  
- `data_loaders.py`: Functions for loading and manipulating datasets.  
- `requirements.txt`: Required packages and their versions.  
- `data/`: Directory containing the data file `Beta_simulated.csv`.  
  
## Getting Started  
  
### Prerequisites  
  
Ensure you have Python 3.x installed. It is recommended to use a virtual environment to manage dependencies.  
  
### Installing  
  
1. Clone the repository to your local machine:  
  
git clone <repository-url>  
 
2. Navigate to the project directory:

cd project_directory  
 
3. Create and activate a virtual environment:

python -m venv .venv  
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`  
 
4. Install the required packages:

pip install -r requirements.txt  
 

Running the Project
 
Execute the main.py script to train the neural network and evaluate it on the Beta dataset.
python main.py  
 

--------- 


OpenMP Conflict Warning
 
If you encounter the following warning:

Found Intel OpenMP ('libiomp') and LLVM OpenMP ('libomp') loaded at the same time.  
you can set environment variables to handle this conflict (uncomment the first three lines of main.py):
import os  
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  
os.environ["OMP_NUM_THREADS"] = "1"  
 

ModuleNotFoundError for geomstats
 
Ensure you have installed the latest version of geomstats:
pip install --upgrade geomstats  
 

