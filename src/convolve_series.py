# =========================================================================== #
"""
Created on Thu Feb 10 13:35:12 2022
@authors: Zachery Brown

Description: This file, convolve_series.py is for convolving kernels with the 
density field in the ConKer algorithm. The object, ConKerBox applies 
this process to one of the partitioned regions. Print statements and verbose
options correspond to the series implementation. The driver routine applies 
the method to all partitioned regions for a given catalog. Detailed time 
records are kept and saved. 

WARNING: These routines generate large numbers of files corresponding
         to three dimensional grids. Make sure you have adequate disk space 
         available when using them.
"""
# =========================================================================== #

# Retrieve util functions
import src.utils as u
import src.print_statements as ps

# Required scipy functions
from scipy.signal import fftconvolve

# Required python imports
import time

# numpy required
import numpy as np

# astropy fits required for file reading/writing
from astropy.io import fits

# =================== CONKER CONVOLVE PER PARTITION ========================= #


class ConKerBox:
    # Convolves the selected kernels for one of the partitioned regions
    # Reqs data type, cfg file, and the box number
    # Will only work if the DivideScheme file has been saved!
    # Designed to run in series
    # TODO -> Multiprocessing options in future versions
    # Each function corresponds to a different mode
    
    
    def __init__(self, corr, box_idx: int):
        # Reads in a data, randoms, and cfg file
        # Sets up print statemetns if inital box
        # Accounts for timing
        
        # Determines if the randoms need to be saved
        self.rand_save = corr.store_rand
        
        # Grabs info from the divide plan
        # Not failable at this stage if it has already been
        #   cleared by the driver
        self.LOS,self.conv_box_lims,self.map_box_lims,self.total_boxes,\
            self.shift_condition=u.getLOSbox(
                corr.div_file, corr.cfg_file,box_idx)

        # Set the rounding precision (very important paramter for this step!)
        self.rounding_pr = corr.cfg_set['pR']
            
        # Set box ID
        self.box_idx = box_idx
        
        # Set output dirs and end
        self.data_loc = './conv/'+corr.data_file.split(
            '.fits')[0]+'_'+corr.cfg_file.split(
            '.txt')[0]+'_mu_wedge_2pcf_conv/'
        if (corr.store_rand == True)&(corr.data_type == 'galaxy'):
            self.rand_loc = './conv/'+corr.rand_file.split(
                '.fits')[0]+'_'+corr.cfg_file.split(
                '.txt')[0]+'_mu_wedge_2pcf_conv/'
        return
    
            
    def mapToDensityField(self,corr):
        # A function to map the objects to 3D grids
        
        # Starts the timing
        self.start_map_time = time.perf_counter()
        
        # Read in the data and randoms 
        data_full = fits.open('./data/'+corr.data_file)[1].data
        if corr.data_type == 'galaxy':
            rand_full = fits.open('./data/'+corr.rand_file)[1].data
            
        # Sets the readout values for the LOS (ignores numerical shift)
        self.LOS_ra_readout = np.asarray([self.LOS[0]])
        
        if self.shift_condition == True:
            # If the catalog is shifted (SGC)
            
            # Shift the RA values to the other side of the sky
            data_full['ra'] = u.shift_RA(data_full['ra'])
            if corr.data_type == 'galaxy':
                rand_full['ra'] = u.shift_RA(rand_full['ra'])
            
            # Fix the readout value for the print statements
            self.LOS_ra_readout = u.shift_RA(np.asarray([self.LOS[0]]))
            
        # Map the data and randoms to four boxes
        # Two for convolution (inner) regions, two for mapping (outer) regions
        # Outer region first 
        self.data = u.coordBoxSlice(data_full,self.map_box_lims)
        if corr.data_type == 'galaxy':
            self.rand = u.coordBoxSlice(rand_full,self.map_box_lims)
        
        # Inner region next
        # The "_0" flag marks the convolution (inner) region
        self.data_0 = u.coordBoxSlice(data_full,self.conv_box_lims)
        if corr.data_type == 'galaxy':
            self.rand_0 = u.coordBoxSlice(rand_full,self.conv_box_lims)
            # Get the nD and nR normalization values for the full catalogs
            self.nD = np.sum(data_full['wts'])
            self.nR = np.sum(rand_full['wts'])
            
        # Delete the large fits catalogs
        del data_full
        if corr.data_type == 'galaxy':
            del rand_full
                
        # Map the LUT radii and redshifts, padding by 0.01
        if corr.data_type == 'galaxy':
            LUT_radii, LUT_redshifts = u.interpolate_r_z(
                self.rand['z'].min()-0.01,
                self.rand['z'].max()+0.01, corr.cosmo)  
        elif corr.data_type == 'lyman_alpha_1D':
            LUT_radii, LUT_redshifts = u.interpolate_r_z(
                self.data['z'].min()-0.01,
                self.data['z'].max()+0.01, corr.cosmo) 
            
        # Define the data (and randoms if needed) XYZ coordinates
        # O(N) operation
        # Outer region first 
        # Coordinates are transformed to local cartesian wrt the LOS
        # The "_0" flag marks the convolution (inner) region
        data_XYZ = np.array(u.sky2localCart((self.data['ra'],self.data['dec'],
                                             LUT_radii(self.data['z'])),
                                            self.LOS)).T
        data_XYZ_0 = np.array(u.sky2localCart((self.data_0['ra'],
                                               self.data_0['dec'],
                                               LUT_radii(self.data_0['z'])),
                                              self.LOS)).T
        if corr.data_type == 'galaxy':
            rand_XYZ = np.array(
                u.sky2localCart((self.rand['ra'],self.rand['dec'],
                                 LUT_radii(self.rand['z'])),self.LOS)).T
            rand_XYZ_0 = np.array(
                u.sky2localCart((self.rand_0['ra'],self.rand_0['dec'],
                                 LUT_radii(self.rand_0['z'])),self.LOS)).T
                
        # PS verb shows the limits of the mapping and convolution regions
        if self.box_idx == 0:
            if corr.verbose:
                ps.verb_LOS_readout(self.LOS_ra_readout, self.LOS,
                                    self.map_box_lims, self.conv_box_lims)
                ps.verb_coord_transfer()
            
        # Define the centers and edges of the mapping (outer) box cells
        # Bins will be used to map the inner region as well
        if corr.data_type == 'galaxy':
            bin_XYZ = rand_XYZ
        elif corr.data_type == 'lyman_alpha_1D':
            bin_XYZ = data_XYZ
        self.grid_edges = [
            u.makeBinEdges((bin_XYZ.T[0].min()-corr.g_s,
                            bin_XYZ.T[0].max()+corr.g_s),corr.g_s),
            u.makeBinEdges((bin_XYZ.T[1].min()-corr.g_s,
                            bin_XYZ.T[1].max()+corr.g_s),corr.g_s),
            u.makeBinEdges((bin_XYZ.T[2].min()-corr.g_s,
                            bin_XYZ.T[2].max()+corr.g_s),corr.g_s)]
        self.grid_centers = [
            u.edge2cen(np.asarray(self.grid_edges[0])),
            u.edge2cen(np.asarray(self.grid_edges[1])),
            u.edge2cen(np.asarray(self.grid_edges[2]))]
        
        # Map the objects to a grid (histogram)
        if corr.data_type == 'galaxy':
            self.D_g = np.histogramdd(
                data_XYZ,bins=self.grid_edges,weights=self.data['wts'])[0]
            self.D_g_0 = np.histogramdd(
                data_XYZ_0,bins=self.grid_edges,weights=self.data_0['wts'])[0]
            self.R_g = (self.nD/self.nR)*np.histogramdd(
                rand_XYZ,bins=self.grid_edges,weights=self.rand['wts'])[0]
            self.R_g_0 = (self.nD/self.nR)*np.histogramdd(
                rand_XYZ_0,bins=self.grid_edges,weights=self.rand_0['wts'])[0]
        elif corr.data_type == 'lyman_alpha_1D':
            D_g_num = np.histogramdd(
                data_XYZ,bins=self.grid_edges,
                weights=self.data['wts'])[0]
            D_g_denom = np.histogramdd(
                data_XYZ,bins=self.grid_edges,
                weights=np.ones(len(self.data['wts'])))[0]
            D_g_denom[D_g_denom == 0.] = np.inf
            self.D_g = D_g_num/D_g_denom
            D_g_0_num = np.histogramdd(
                data_XYZ_0,bins=self.grid_edges,
                weights=self.data_0['wts'])[0]
            D_g_0_denom = np.histogramdd(
                data_XYZ_0,bins=self.grid_edges,
                weights=np.ones(len(self.data_0['wts'])))[0]
            D_g_0_denom[D_g_0_denom == 0.] = np.inf
            self.D_g_0 = D_g_0_num/D_g_0_denom
            R_g_num = np.histogramdd(
                data_XYZ,bins=self.grid_edges,
                weights=self.data['sys_wts'])[0]
            R_g_denom = np.histogramdd(
                data_XYZ,bins=self.grid_edges,
                weights=np.ones(len(self.data['sys_wts'])))[0]
            R_g_denom[R_g_denom == 0.] = np.inf
            self.R_g = R_g_num/R_g_denom
            R_g_0_num = np.histogramdd(
                data_XYZ_0,bins=self.grid_edges,
                weights=self.data_0['sys_wts'])[0]
            R_g_0_denom = np.histogramdd(
                data_XYZ_0,bins=self.grid_edges,
                weights=np.ones(len(self.data_0['sys_wts'])))[0]
            R_g_0_denom[R_g_0_denom == 0.] = np.inf
            self.R_g_0 = R_g_0_num/R_g_0_denom
            
        # PS to mark histogramming
        if self.box_idx == 0:
            if corr.verbose:
                ps.verb_hist_succ()
        
        # Mark timing for mapping
        self.end_map_time = time.perf_counter()
        self.map_time = self.end_map_time - self.start_map_time
        
        # Initialize a value for the file writing time
        self.file_time = 0.
        
        # Start file writing time
        self.file_start_time_0 = time.perf_counter()
        
        # Create a mask to reduce file sized
        self.mask = u.make_grid_mask(self.R_g_0)
        
        # Write the W0 and B0 grids (inner)
        # These values correspond to the convolution regions
        if corr.data_type == 'galaxy':
            u.grid_to_fits_wrapper(self.D_g_0-self.R_g_0,self.mask,
                                   self.data_loc+'W_p{}_of_{}.fits'.format(
                                       self.box_idx+1,self.total_boxes))
            if self.rand_save:
                u.grid_to_fits_wrapper((self.nR/self.nD)*self.R_g_0,self.mask,
                                       self.rand_loc+
                                       'B_p{}_of_{}.fits'.format(
                                           self.box_idx+1,self.total_boxes))
        elif corr.data_type == 'lyman_alpha_1D':
            u.grid_to_fits_wrapper(self.D_g_0,self.mask,
                                   self.data_loc+'W_p{}_of_{}.fits'.format(
                                       self.box_idx+1,self.total_boxes))
            u.grid_to_fits_wrapper(self.R_g_0,self.mask,
                                   self.data_loc+'B_p{}_of_{}.fits'.format(
                                       self.box_idx+1,self.total_boxes))
        # End file times for now
        self.file_end_time_0 = time.perf_counter()
        self.file_time += self.file_end_time_0 - self.file_start_time_0
        
        # PS to mark mapping and end
        if self.box_idx == 0:
            if corr.verbose:
                ps.verb_grid_file_succ()
        return
            
            
    def radialConvolveMuWedges(self, corr):
        # A function to convolve the density field with 
        #   mu wedge kernels
        
        # Set up the mu binning
        mu_bin_centers = u.edge2cen(corr.mu_bin_edges)
        
        # Set up timing
        self.kernel_time = 0.
        self.conv_time = 0.
        
        for s_idx in range(len(corr.desired_s_bin_centers)):
            # Loop through all radial steps
            
            for mu_idx in range(len(mu_bin_centers)):
                # For each mu bin
                
                # Starts the kernel timer
                kernel_start_time = time.perf_counter()
                
                # Load the kernel
                ker_grid = np.load(self.data_loc+
                                   'K_{}_{}.npy'.format(s_idx,mu_idx))
                
                # End kernel timer and update
                kernel_end_time = time.perf_counter()
                self.kernel_time += kernel_end_time-kernel_start_time
                
                # Start convolution timer
                conv_start_time = time.perf_counter()
                
                # Run the convolution step
                if corr.data_type == 'galaxy':
                    W_i_mu = self.mask*np.round(
                        fftconvolve(self.D_g-self.R_g,ker_grid,mode='same'),
                        decimals=self.rounding_pr)
                    if self.rand_save:
                        B_i_mu = self.mask*np.round(
                            fftconvolve(self.R_g,ker_grid,mode='same'),
                            decimals=self.rounding_pr)
                elif corr.data_type == 'lyman_alpha_1D':
                    W_i_mu = self.mask*np.round(
                        fftconvolve(self.D_g,ker_grid,mode='same'),
                        decimals=self.rounding_pr)
                    B_i_mu = self.mask*np.round(
                        fftconvolve(self.R_g,ker_grid,mode='same'),
                        decimals=self.rounding_pr)
                
                # End convolution timer and update
                conv_end_time = time.perf_counter()
                self.conv_time += conv_end_time - conv_start_time
                
                # Start file timer
                file_start_time_1 = time.perf_counter()
                
                # Write the grids to file
                if corr.data_type == 'galaxy':
                    u.grid_to_fits_wrapper(
                        W_i_mu,self.mask,self.data_loc+
                        'W_{}_{}_p{}_of_{}.fits'.format(
                            s_idx,mu_idx,self.box_idx+1,
                            self.total_boxes))
                    if self.rand_save:
                        u.grid_to_fits_wrapper(
                            B_i_mu,self.mask,self.rand_loc+
                            'B_{}_{}_p{}_of_{}.fits'.format(
                                s_idx,mu_idx,self.box_idx+1,
                                self.total_boxes))
                elif corr.data_type == 'lyman_alpha_1D':
                    u.grid_to_fits_wrapper(
                        W_i_mu,self.mask,self.data_loc+
                        'W_{}_{}_p{}_of_{}.fits'.format(
                            s_idx,mu_idx,self.box_idx+1,
                            self.total_boxes))
                    u.grid_to_fits_wrapper(
                        B_i_mu,self.mask,self.data_loc+
                        'B_{}_{}_p{}_of_{}.fits'.format(
                            s_idx,mu_idx,self.box_idx+1,
                            self.total_boxes))
                        
                # End file timer and update
                file_end_time_1 = time.perf_counter()
                self.file_time += file_end_time_1 - file_start_time_1
                
            # PS if initial box
            if self.box_idx == 0:
                if corr.verbose:
                    ps.verb_radial_progress(
                        corr.desired_s_bin_centers[s_idx])
        
        # PS to mark the end of a partition region
        ps.req_part_done(self.box_idx,self.total_boxes)
        return


