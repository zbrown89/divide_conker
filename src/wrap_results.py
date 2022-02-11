# =========================================================================== #
"""
Created on Thu Feb 10 15:35:17 2022
@authors: Zachery Brown

Description: This file, wrap_results.py is used to run the final steps of the 
ConKer algorithm. It requires a directory of grid files, saved by running the 
convolution step.
"""
# =========================================================================== #

# Retrieve util functions and print statemtnes
import src.utils as u
import src.print_statements as ps

# Required python imports
import time
import json

# numpy required
import numpy as np

# astropy fits required for file reading/writing
from astropy.io import fits

# ========================= 2PCF MU WEDGE WRAP ============================== #


def MuWedgeWrapper(corr, mu_edges: np.array, 
                   plot_correlation: bool, save_correlation_plot: bool):

    # List mu bin properties
    # Mark the begining of the kernel writing procedure
    mu_bin_edges = mu_edges
    ps.req_mu_bin_readout(mu_bin_edges)
    
    # Set output dirs and end
    data_loc = './conv/'+corr.data_file.split(
        '.fits')[0]+'_'+corr.cfg_file.split(
        '.txt')[0]+'_mu_wedge_2pcf_conv/'
    if corr.data_type == 'galaxy':
        rand_loc = './conv/'+corr.rand_file.split(
            '.fits')[0]+'_'+corr.cfg_file.split(
            '.txt')[0]+'_mu_wedge_2pcf_conv/'
        
    # Try to read in a kernel grid to check that the previous 
    #   step has been completed, Delete if successful
    try:
        # Try to read in a kernel grid at mu_min to check that the previous 
        #   step has been completed, delete if successful
        ktest = np.load(data_loc+'K_0_0.npy')
        del ktest
    except FileNotFoundError:
        ps.fail_no_files_wrap()
        return
    
    if corr.data_type == 'galaxy':
        # Get the nD and nR normalization values for the full catalogs
        data_full = fits.open('./data/'+corr.data_file)[1].data
        rand_full = fits.open('./data/'+corr.rand_file)[1].data
        nD = np.sum(data_full['wts'])
        nR = np.sum(rand_full['wts'])
        del data_full, rand_full
        
    # Initalize the correlation dictionary for data storage
    corr_dict = {}
    
    # Set s and initialize W_mu and B_mu arrays
    # Set up the mu binning
    mu_bin_centers = u.edge2cen(mu_edges)
    corr_dict['s'] = np.linspace(
        corr.cfg_set['sMin'],corr.cfg_set['sMax'],corr.cfg_set['sBinN'])
    for mu_idx in range(len(mu_bin_centers)):
        corr_dict['W_mu'+str(mu_idx)] = np.zeros(len(corr_dict['s']))
        corr_dict['B_mu'+str(mu_idx)] = np.zeros(len(corr_dict['s']))
    
    # Initialize timing dict
    with open(data_loc+'timing_info.txt') as timeDic:
        timing = json.load(timeDic)
        
    # Grabs timing info from the divide plan
    timing['T_DIVP'] = fits.open('./divide/plans/'+
                                 corr.div_file.split('.fits')[0]+
                                 '_'+corr.cfg_file.split('.txt')[0]+
                                 '.fits')[1].header['T_DIV']
    timing['T_FILEWR'] = 0.
    timing['T_WRAP'] = 0.
    
    # Grab the total partition box number
    _, _, _, total_boxes, _ = u.getLOSbox(corr.div_file, corr.cfg_file, 0)
    
    for box_idx in range(total_boxes):
        # Loop through each partitioned region
        
        # Start file sum timer
        file_start_time = time.perf_counter()
        
        # Read in W (and B) grids
        W_ = u.fits_to_grid_unwrapper(
            data_loc+'W_p{}_of_{}.fits'.format(box_idx+1,total_boxes))
        if corr.data_type == 'galaxy':
            B_ = (nD/nR)*u.fits_to_grid_unwrapper(
                rand_loc+'B_p{}_of_{}.fits'.format(box_idx+1,total_boxes))
        elif corr.data_type == 'lyman_alpha_1D':
            B_ = u.fits_to_grid_unwrapper(
                data_loc+'B_p{}_of_{}.fits'.format(box_idx+1,total_boxes))
        
        # End file sum timer and update
        file_end_time = time.perf_counter()
        timing['T_FILEWR'] += file_end_time - file_start_time
        
        for s_idx in range(len(corr_dict['s'])):
            # Loop through each s step
            
            for mu_idx in range(len(mu_bin_centers)):
                # For each mu bin
                
                # Start file sum timer
                file_start_time = time.perf_counter()
                
                # Read in W_i_mu (and Bi_mu) grids
                W_i_mu = u.fits_to_grid_unwrapper(
                    data_loc+'W_{}_{}_p{}_of_{}.fits'.format(
                        s_idx,mu_idx,box_idx+1,total_boxes))
                if corr.data_type == 'galaxy':
                    B_i_mu = (nD/nR)*u.fits_to_grid_unwrapper(
                    rand_loc+'B_{}_{}_p{}_of_{}.fits'.format(
                        s_idx,mu_idx,box_idx+1,total_boxes))
                elif corr.data_type == 'lyman_alpha_1D':
                    B_i_mu = u.fits_to_grid_unwrapper(
                    data_loc+'B_{}_{}_p{}_of_{}.fits'.format(
                        s_idx,mu_idx,box_idx+1,total_boxes))  

                # End file sum timer and update
                file_end_time = time.perf_counter()
                timing['T_FILEWR'] += file_end_time - file_start_time
                
                # Start summation timer
                sum_start_time = time.perf_counter()
                
                # Create a normalization factor 
                # Sum/avg over W_i_mu/B_i_mu grids for this kernel size
                density_arr_W = W_*W_i_mu
                density_arr_B = B_*B_i_mu
                if corr.data_type == 'galaxy':
                    norm = 1.
                    corr_dict['W_mu'+str(mu_idx)][s_idx] += norm*np.sum(
                        density_arr_W)
                    corr_dict['B_mu'+str(mu_idx)][s_idx] += norm*np.sum(
                        density_arr_B)
                elif corr.data_type == 'lyman_alpha_1D':
                    norm = 1/(total_boxes)
                    corr_dict['W_mu'+str(mu_idx)][s_idx] += norm*np.average(
                        density_arr_W[density_arr_W!=0.])
                    corr_dict['B_mu'+str(mu_idx)][s_idx] += norm*np.average(
                        density_arr_B[density_arr_B!=0.])
                                
                # End summation timer and update
                sum_end_time = time.perf_counter()
                timing['T_WRAP'] += sum_end_time - sum_start_time
                
        # PS to mark end of part
        if corr.verbose:
            ps.req_part_done(box_idx,total_boxes)
        
    # Compute the correlation functions
    for mu_idx in range(len(mu_bin_centers)):
        corr_dict['xi_mu'+str(mu_idx)] = corr_dict[
            'W_mu'+str(mu_idx)]/corr_dict['B_mu'+str(mu_idx)]
    
    # PS verb for file writing
    if corr.verbose:
        ps.verb_file_write()
        
    # Create fits column list for the data
    corr_cols=[]
    
    for key in corr_dict.keys():
        # Loop through each column in the corr pkg
        
        # Write the data to a fits column and append
        corr_cols.append(fits.Column(
            name=key,array=corr_dict[key],format='E'))
        
    # Write the correlation data to a fits table
    corr_table = fits.BinTableHDU.from_columns(corr_cols)
    
    # Append the header with timing information
    # Keep file writing/reading time split between convolution and wrapping
    corr_table.header.set('T_DIVP',timing['T_DIVP'])
    corr_table.header.set('T_MAP',timing['T_MAP'])
    corr_table.header.set('T_CONV',timing['T_CONV'])
    corr_table.header.set('T_KERN',timing['T_KERN'])
    corr_table.header.set('T_FILE',timing['T_FILE'])
    corr_table.header.set('T_FILEWR',timing['T_FILEWR'])
    corr_table.header.set('T_WRAP',timing['T_WRAP'])
    
    # Write the corr pkg to file
    # Overwrite is fine here because the routine is fast
    corr_table.writeto('./corr/'+corr.data_file.split('.fits')[0]+
                       '_'+corr.cfg_file.split('.txt')[0]+
                       '/corr_pkg_2pcf.fits',
                       overwrite=True)
    
    if plot_correlation == False:
        # If the user does not wish to plot results
        # PS end and timing breakdown
        ps.req_wrap_timer(corr.data_type,
                          np.round(timing['T_FILEWR']+
                                   timing['T_WRAP'],decimals=4))
        if corr.verbose:
            ps.verb_wrap_timing_breakdown(
                corr.data_type,np.round(timing['T_WRAP'],decimals=4),
                np.round(timing['T_FILEWR'],decimals=4))
        ps.req_bar()
        return    
    

