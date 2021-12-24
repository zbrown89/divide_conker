# =========================================================================== #
"""
Created on Fri Dec 24 09:10:23 2021
@authors: Zachery Brown

Description: This file, corrfunc_driver.py, is used to run the various 
routines of the Divide+ConKer algorithm. The user specifies a "data" and 
"randoms" catalog, as well as a cfg file. This driver will compute the 2pcf
expanded to a desired Legendre multipole, diagonal elements of the npcf to a 
given n_max, and the 3pcf up to a desired Legendre multipole. While this 
driver does not perform the 3pcf edge-correction, a routine for that may be 
found in utils.py.
"""
# =========================================================================== #

# Import routines from src
from src.divide import DivideCatalog
from src.conker_series import ConKerConvolveCatalog
from src.summation import Summation2PCF, Summation3PCF, SummationNPCFdiag

# Set the name of the "data" and "randoms" catalogs (should be in ./data)
data_file = 'dr12S_data.fits'
rand_file = 'dr12S_randoms.fits'

# Set the name of the cfg file (should be in ./params)
cfg_file = 'bossDR12paramsCoarse.txt'

# Partition the random catalog
DivideCatalog(rand_file,cfg_file,
              
              # Opt
              save_plan = True,
              plot_results = False,
              save_plot = False,
              verbose = True)

# Convolve kernels in series
ConKerConvolveCatalog(data_file,rand_file,cfg_file,
                      # Opt
                      store_rand = True,
                      ell_max=5,
                      verbose=True)

# Compute the expanded 2pcf
Summation2PCF(data_file,rand_file,cfg_file,
              # Opt
              ell_max=2,
              plot_correlation = False,
              save_plot = False, 
              verbose = True)

# Compute the expanded 3pcf
Summation3PCF(data_file,rand_file,cfg_file,
              # Opt
              ell_max=5,
              verbose = True)

# Compute the diagonal elements fo the npcf to a given order, n
SummationNPCFdiag(data_file,rand_file,cfg_file,
                  # Opt
                  n_max=5,
                  verbose = True)


