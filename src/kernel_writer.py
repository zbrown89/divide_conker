# =========================================================================== #
"""
Created on Tue Feb  8 15:31:39 2022
@authors: Zachery Brown

Description: The kernel writer is used at the begining of the convolution step
to create and store all necessary kernels. Each of its functions corresponds
to one of the correlation function modes of the main driver.
"""
# =========================================================================== #

# Retrieve util functions
import src.utils as u
import src.print_statements as ps

# Required python imports
import time

# numpy required
import numpy as np

# ========================= KERNEL WRITER =================================== #


class KernelWriter:
    # An object to write the kernels needed for particular npcf routines
    # Each mode will have its own function
    
    
    def __init__(self,corr):
        # Setup file names
        
        self.file_loc = './conv/'+corr.data_file.split(
            '.fits')[0]+'_'+corr.cfg_file.split('.txt')[0]
            
    
    def write_2pcf_mu_wedges(self,corr):
        # A function to write the mu wedge kernels
        
        # Start the timing
        kernel_start_time = time.perf_counter()
        
        # Set up the mu binning
        mu_bin_centers = u.edge2cen(corr.mu_bin_edges)
        mu_bin_edges = corr.mu_bin_edges
        
        for mu_idx in range(len(mu_bin_centers)):
            # For each mu bin
            
            for s_idx in range(len(corr.desired_s_bin_centers)):
                # Loop through all radial steps
                
                # Get the kernel grid
                kern_grid = u.mu_wedge_kernel(
                    corr.desired_s_bin_centers[s_idx],
                    corr.g_s,(mu_bin_edges[mu_idx],
                              mu_bin_edges[mu_idx+1]))
                
                # Write the kernel grid to file
                np.save(self.file_loc+'_mu_wedge_2pcf_conv/'+
                        'K_{}_{}.npy'.format(s_idx,mu_idx),kern_grid)
                
            # PS to mark mu wedge completion
            if corr.verbose:
                ps.verb_kernel_mu_wedge_progress(mu_idx,len(mu_bin_centers))
    
        # End timing
        kernel_end_time = time.perf_counter()
        self.time = kernel_end_time - kernel_start_time
        return
    
    
    