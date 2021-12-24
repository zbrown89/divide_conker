# =========================================================================== #
"""
Created on Wed Dec 15 17:59:39 2021
@authors: Zachery Brown

Description: This file, conker_series.py is for convolving kernels with the 
density field in the Divide+ConKer algorithm. The object, ConKerBox applies 
this process to one of the partitioned regions. Print statements and verbose
options correspond to the series implementation. The driver routine applies 
the method to all partitioned regions for a given catalog. Detailed time 
records are kept and saved. 

WARNING: These routines generate large numbers of files corresponding
         to three dimensional grids. Make sure you have adequate disk space 
         available when using them. The user may choose (with caution!) to set
         ftype = 'npy' when running the driver. This process is faster but 
         results in much larger files. The default 'fits' method uses a 
         built-in scheme to "sparsify" the arrays for storage. The npy method
         preserves spatial information, the fits method does not.
"""
# =========================================================================== #

# Retrieve util functions
import src.utils as u

# Required scipy functions
from scipy.signal import fftconvolve

# Required python imports
import time
import json
import os

# numpy required
import numpy as np

# astropy fits required for file reading/writing
from astropy.io import fits

# =================== CONKER CONVOLVE PER PARTITION ========================= #


class ConKerBox:
    # Convolves the selected kernels for one of the partitioned regions
    # Reqs data catalog, randoms catalog, cfg file, and the box number
    # Will only work if the DivideScheme file has been saved!
    # Designed to run in series
    # TODO -> Multiprocessing options in future versions
    
    
    def __init__(self, data_file: str, rand_file: str, cfg_file: bool,
                 store_rand: str, box_idx: int, verbose: bool):
        # Reads in a data, randoms, and cfg file
        # Sets up print statemetns if inital box
        # Accounts for timing
        
        # Determines if the randoms need to be saved
        self.rand_save = store_rand
        
        # Set the data and randoms file string 
        self.datafileString = data_file.split('.fits')[0]
        if self.rand_save:
            self.randfileString = rand_file.split('.fits')[0]
        
        # PS if initial box to mark the start of the routine
        if box_idx == 0:
            print('\n'+'===================== '+'CONKER CONVOLUTION STEP'+
                  ' ====================='+'\n')
            print('Convolving kernels with catalog '+self.datafileString)
            
            # PS verb to display the catalog filenames and cfg file
            if verbose:
                print('Preparing to measure correlation functions...')
                print('Data file: '+data_file.split('.fits')[0])
                print('Randoms file: '+rand_file.split('.fits')[0])
                print('Configuration: '+cfg_file.split('.txt')[0])
                
        # Grabs info from the divide plan
        # Not failable at this stage if it has already been
        #   cleared by the driver
        self.LOS,self.conv_box_lims,self.map_box_lims,self.total_boxes,\
            self.shift_condition=u.getLOSbox(rand_file, cfg_file, box_idx)
            
        # Reads in the data, randoms, and cfg names
        self.data_file = data_file
        self.rand_file = rand_file
        self.cfg_file = cfg_file
        
        # Sets box index
        self.box_idx = box_idx

        # Sets verbose
        self.verbose = verbose
        
        # Stores parameters from the cfg file
        with open('./params/'+self.cfg_file) as cfgDict:
            cfg_set = json.load(cfgDict)
        self.cosmo = (cfg_set['c'],cfg_set['H0'],
                      (cfg_set['OmM'],0.,1.-cfg_set['OmM']))
        
        # Sets the s-bins using the cfg file
        # Uses the s-bins to store the grid size g_S
        self.desired_s_bin_centers = np.linspace(
            cfg_set['sMin'],cfg_set['sMax'],cfg_set['sBinN'])
        self.desired_s_bin_edges = u.cen2edge(self.desired_s_bin_centers)
        self.g_s = self.desired_s_bin_centers[1]-self.desired_s_bin_centers[0]
        
        # Set the rounding precision (very important paramter for this step!)
        self.rounding_pr = cfg_set['pR']
        
        # Determine if the density field will be weighted
        self.wtd = cfg_set['wtd']
        
        # Starts the timing
        self.start_map_time = time.perf_counter()
        
        # Read in the data and randoms 
        data_full = fits.open('./data/'+self.data_file)[1].data
        rand_full = fits.open('./data/'+self.rand_file)[1].data
        
        # Sets the readout values for the LOS (ignores numerical shift)
        self.LOS_ra_readout = np.asarray([self.LOS[0]])
        
        if self.wtd == False:
            # If weights are not goint to be used
            
            # Create arrays of weight 1 for data and randoms
            data_full['wts'] = np.ones(len(data_full),dtype='float')
            rand_full['wts'] = np.ones(len(rand_full),dtype='float')
            
        if self.shift_condition == True:
            # If the catalog is shifted (SGC)
            
            # Shift the RA values to the other side of the sky
            data_full['ra'] = u.shift_RA(data_full['ra'])
            rand_full['ra'] = u.shift_RA(rand_full['ra'])
            
            # Fix the readout value for the print statements
            self.LOS_ra_readout = u.shift_RA(np.asarray([self.LOS[0]]))
            
        # Map the data and randoms to four boxes
        # Two for convolution (inner) regions, two for mapping (outer) regions
        # Outer region first 
        self.data = u.coordBoxSlice(data_full,self.map_box_lims)
        self.rand = u.coordBoxSlice(rand_full,self.map_box_lims)
        
        # Inner region next
        # The "_0" flag marks the convolution (inner) region
        self.data_0 = u.coordBoxSlice(data_full,self.conv_box_lims)
        self.rand_0 = u.coordBoxSlice(rand_full,self.conv_box_lims)
        
        # Get the nD and nR normalization values for the full catalogs
        self.nD = np.sum(data_full['wts'])
        self.nR = np.sum(rand_full['wts'])
        
        # Delete the large fits catalogs
        del data_full, rand_full
        
        # PS if initial box
        if self.box_idx == 0:
            
            # PS verb shows the limits of the mapping and convolution regions
            if self.verbose:
                print('Initial LOS at (RA: '+
                      str(np.round(self.LOS_ra_readout[0],decimals=2))+
                      ', DEC: '+str(np.round(self.LOS[1],decimals=2))+')')
                print('Initial mapping box is '+
                      str(np.round(self.map_box_lims[0][1]-
                                   self.map_box_lims[0][0],decimals=2))+
                      ' deg. by '+str(np.round(self.map_box_lims[1][1]-
                                               self.map_box_lims[1][0],
                                               decimals=2))+' deg.')
                print('Initial convolution box is '+
                      str(np.round(self.conv_box_lims[0][1]-
                                   self.conv_box_lims[0][0],decimals=2))+
                      ' deg. by '+str(np.round(self.conv_box_lims[1][1]-
                                               self.conv_box_lims[1][0],
                                               decimals=2))+' deg.')
     
        
    def radialConvolveLegendre(self, ell_max: int, ftype: str = 'fits'):
        # A function to convolve the density field with 
        #   Y_ell_max^_ell to Y_-ell_max^ell kernels
        # ftype may be set to 'fits' (default) or 'npy'
        # 'npy' is significantly faster but creates larger files
        
        # Store ell_max and define the ell_steps
        self.ell_max = ell_max
        self.ell_step = np.linspace(0,ell_max,ell_max+1,dtype=int)
        
        # Map the LUT radii and redshifts, padding by 0.01
        LUT_radii, LUT_redshifts = u.interpolate_r_z(
            self.rand['z'].min()-0.01, self.rand['z'].max()+0.01, self.cosmo)
        
        # Define the data and randoms XYZ coordinates
        # O(N) operation
        # Outer region first 
        # Coordinates are transformed to local cartesian wrt the LOS
        data_XYZ = np.array(u.sky2localCart((self.data['ra'],self.data['dec'],
                                             LUT_radii(self.data['z'])),
                                            self.LOS)).T
        rand_XYZ = np.array(u.sky2localCart((self.rand['ra'],self.rand['dec'],
                                             LUT_radii(self.rand['z'])),
                                            self.LOS)).T
        
        # Inner region next
        # The "_0" flag marks the convolution (inner) region
        data_XYZ_0 = np.array(u.sky2localCart((self.data_0['ra'],
                                               self.data_0['dec'],
                                               LUT_radii(self.data_0['z'])),
                                              self.LOS)).T
        rand_XYZ_0 = np.array(u.sky2localCart((self.rand_0['ra'],
                                               self.rand_0['dec'],
                                               LUT_radii(self.rand_0['z'])),
                                              self.LOS)).T
        
        # Define the centers and edges of the mapping (outer) box cells
        # Bins will be used to map the inner region as well
        self.grid_edges = [
            u.makeBinEdges((data_XYZ.T[0].min()-self.g_s,
                            data_XYZ.T[0].max()+self.g_s),self.g_s),
            u.makeBinEdges((data_XYZ.T[1].min()-self.g_s,
                            data_XYZ.T[1].max()+self.g_s),self.g_s),
            u.makeBinEdges((data_XYZ.T[2].min()-self.g_s,
                            data_XYZ.T[2].max()+self.g_s),self.g_s)]
        self.grid_centers = [
            u.edge2cen(np.asarray(self.grid_edges[0])),
            u.edge2cen(np.asarray(self.grid_edges[1])),
            u.edge2cen(np.asarray(self.grid_edges[2]))]
        
        # PS if initial box
        if self.box_idx == 0:
            
            # PS verb to mark successful coordinate transformation
            if self.verbose:
                print('Successful coordinate transformation...')
                
        # Map the data galaxies to a grid (NGP method)
        self.D_g = np.histogramdd(
            data_XYZ,bins=self.grid_edges,weights=self.data['wts'])[0]
        self.D_g_0 = np.histogramdd(
            data_XYZ_0,bins=self.grid_edges,weights=self.data_0['wts'])[0]
        
        # PS if initial box
        if self.box_idx == 0:
            
            # PS verb to mark successful data histogram
            if self.verbose:
                print('Successful data histogram...')
                
        # Map the random galaxies to a grid 
        # Normalize this grid to the overall sum of data weights
        self.R_g = (self.nD/self.nR)*np.histogramdd(
            rand_XYZ,bins=self.grid_edges,weights=self.rand['wts'])[0]
        self.R_g_0 = (self.nD/self.nR)*np.histogramdd(
            rand_XYZ_0,bins=self.grid_edges,weights=self.rand_0['wts'])[0]
        
        # PS if initial box
        if self.box_idx == 0:
            
            # PS verb to mark successful randoms histogram
            if self.verbose:
                print('Successful randoms histogram...')
                
        # If initial box, create temporary directory for files
        if self.box_idx == 0:
            
            try:
                # Try to make an appropriate directory to store the grids
                # Tag it with the data and cfg names
                os.makedirs('./grids/'+self.datafileString+'_'
                            +self.cfg_file.split('.txt')[0])
                
            except FileExistsError:
                # If the directory already exists
                
                # Trigger a failure
                # This prevents overwriting grid data by mistake
                print('\n'+'==!==!==!==!==!==!==!==!==!== '+'FAILURE'+
                      ' ==!==!==!==!==!==!==!==!==!=='+'\n')
                print('You already have a temporary directory for this '+
                      'catalog and cfg!')
                print("Check ./grids/ to make sure you don't have "+
                      "existing data")
                print('\n'+'!==!==!==!==!==!==!==!==!==!==!==!=='+
                      '!==!==!==!==!==!==!==!==!==!==!'+'\n')
                return
            
            if self.rand_save:
                try:
                    # Try to make an appropriate dir to store the randoms
                    # Tag it with the randoms and cfg names
                    os.makedirs('./grids/'+self.randfileString+'_'
                                +self.cfg_file.split('.txt')[0])
                    
                except FileExistsError:
                    # If the directory already exists
                    
                    # Trigger a failure
                    # This prevents overwriting grid data by mistake
                    print('\n'+'==!==!==!==!==!==!==!==!==!== '+'FAILURE'+
                          ' ==!==!==!==!==!==!==!==!==!=='+'\n')
                    print('You already have a temporary directory for this '+
                          'catalog and cfg!')
                    print("Check ./grids/ to make sure you don't have "+
                          "existing data")
                    print('\n'+'!==!==!==!==!==!==!==!==!==!==!==!=='+
                          '!==!==!==!==!==!==!==!==!==!==!'+'\n')
                    return
        
        # Set the name for the newly created directory
        tempDir = './grids/'+self.datafileString+'_'+\
            self.cfg_file.split('.txt')[0]+'/'
            
        # Set the name of a newly created directory for randoms
        if self.rand_save:
            tempDirRand = './grids/'+self.randfileString+'_'+\
                self.cfg_file.split('.txt')[0]+'/'
        
        # End the mapping time stamp
        self.end_map_time = time.perf_counter()
        
        # Initialize a value for the file writing time
        self.file_time = 0.
        
        # Start file writing time
        file_start_time_B = time.perf_counter()
        
        # Make a mask to reduce future file sizes
        # Data mask
        data_mask = u.make_grid_mask(self.D_g_0)
        
        # Randoms mask
        rand_mask = u.make_grid_mask(self.R_g_0)
        
        # Combined mask
        self.mask = u.make_grid_mask(data_mask+rand_mask)
        
        # Delete the individual masks for data and randoms
        del data_mask, rand_mask
        
        if ftype == 'fits':
            # If the requested grid filetype is fits
                        
            # Write the W0 and B0 grids (inner)
            # These values correspond to the convolution regions
            u.grid_to_fits_wrapper(self.D_g_0-self.R_g_0,self.mask,
                                   tempDir+'W_p{}_of_{}.fits'.format(
                                       self.box_idx+1,self.total_boxes))
            if self.rand_save:
                u.grid_to_fits_wrapper((self.nR/self.nD)*self.R_g_0,self.mask,
                                       tempDirRand+'B_p{}_of_{}.fits'.format(
                                           self.box_idx+1,self.total_boxes))
        elif ftype == 'npy':
            # If the requested grid filetype is npy
            
            # Write the W0 and B0 grids (inner)
            # These values correspond to the convolution regions
            np.save(tempDir+'W_p{}_of_{}.npy'.format(
                self.box_idx+1,self.total_boxes),self.mask*(
                    self.D_g_0-self.R_g_0))
            if self.rand_save:
                np.save(tempDirRand+'B_p{}_of_{}.npy'.format(
                    self.box_idx+1,self.total_boxes),
                    (self.nR/self.nD)*self.R_g_0*self.mask)
            
        # End file writing time and update
        file_end_time_B = time.perf_counter()
        self.file_time += file_end_time_B - file_start_time_B
        
        # PS if initial box
        if self.box_idx == 0:
            
            # PS verb to mark W0 and B0 files written
            if self.verbose:
                print('Wrote initial region grid files...')
                
        # Initialize kernel construction time
        self.kernel_time = 0.
        
        # Initialize convolution time
        self.conv_time = 0.
        
        for s_idx in range(len(self.desired_s_bin_centers)):
            # Loop through all radial steps
            
            for ell_idx in range(len(self.ell_step)):
                # Loop through all ell steps
                
                # Define the m_steps from -ell_max to ell_max in int steps
                m_step = np.linspace(-self.ell_step[ell_idx],
                                     self.ell_step[ell_idx],
                                     2*self.ell_step[ell_idx]+1,dtype=int)
                
                for m_idx in range(len(m_step)):
                    # Loop through all m values for ell
                    
                    # Make kernel if this is partition 0
                    if self.box_idx == 0:
                        
                        # Starts the kernel timer
                        kernel_start_time_W = time.perf_counter()
                        
                        # Creates the kernel and defines the REAL and IMAG
                        #   grids corresponding to the ylm functions
                        kern_grid_RE, kern_grid_IM = u.ylmKernel(
                            self.desired_s_bin_centers[s_idx],self.g_s,
                            m_=m_step[m_idx],n_=self.ell_step[ell_idx])
                        
                        # End kernel timer and update
                        kernel_end_time_W = time.perf_counter()
                        self.kernel_time += kernel_end_time_W -\
                            kernel_start_time_W
                            
                        # Start the file timer
                        file_start_time_W = time.perf_counter()
                        
                        # Write the kernel grid(s) to file
                        # REAL kernel first
                        np.save(tempDir+
                                'K_{}_{}_{}_RE.npy'.format(
                                    s_idx,self.ell_step[ell_idx],
                                    m_step[m_idx]),kern_grid_RE)
                        
                        if m_step[m_idx] != 0:
                            # If the m value is not 0
                            
                            # Write the IMAG kernel
                            np.save(tempDir+
                                    'K_{}_{}_{}_IM.npy'.format(
                                        s_idx,self.ell_step[ell_idx],
                                        m_step[m_idx]),kern_grid_IM)
                            
                        # End file timer and update
                        file_end_time_W = time.perf_counter()
                        self.file_time += file_end_time_W - file_start_time_W
                        
                    # Otherwise load kernel(s)
                    elif self.box_idx != 0:
                        
                        # Start the file timer
                        file_start_time_W = time.perf_counter()
                        
                        # Load the REAL kernel
                        kern_grid_RE = np.load(
                            tempDir+'K_{}_{}_{}_RE.npy'.format(
                                s_idx,self.ell_step[ell_idx],
                                m_step[m_idx]))
                        
                        if m_step[m_idx] != 0:
                            # If the m value is not 0
                            
                            # Load the IMAG kernel
                            kern_grid_IM = np.load(
                                tempDir+'K_{}_{}_{}_IM.npy'.format(
                                    s_idx,self.ell_step[ell_idx],
                                    m_step[m_idx]))
                            
                        # End file timer and update
                        file_end_time_W = time.perf_counter()
                        self.file_time += file_end_time_W - file_start_time_W
                            
                    # Start convolution timer
                    conv_start_time = time.perf_counter()
                    
                    # Convolve with the density grid (outer region)
                    # Mask for reduction of outside regions
                    W_i_ell_m_RE = self.mask*np.round(fftconvolve(
                        self.D_g-self.R_g,kern_grid_RE,mode='same'),
                        decimals=self.rounding_pr)
                    if self.rand_save:
                        B_i_ell_m_RE = self.mask*np.round(fftconvolve(
                            self.R_g,kern_grid_RE,mode='same'),
                            decimals=self.rounding_pr)
                    
                    if m_step[m_idx] != 0:
                        # If the m value is not 0
                        
                        # Convolve with the density grid (outer region) again
                        #   this time with the IMAG kernel
                        # Mask for reduction of outside regions
                        W_i_ell_m_IM = self.mask*np.round(fftconvolve(
                            self.D_g-self.R_g,kern_grid_IM,mode='same'),
                            decimals=self.rounding_pr)
                        if self.rand_save:
                            B_i_ell_m_IM = self.mask*np.round(fftconvolve(
                                self.R_g,kern_grid_IM,mode='same'),
                                decimals=self.rounding_pr)
                        
                    # End convolution timer and update
                    conv_end_time = time.perf_counter()
                    self.conv_time += conv_end_time - conv_start_time
                    
                    # Start file timer
                    file_start_time_W = time.perf_counter()
                    
                    if ftype == 'fits':
                        # If the requested grid filetype is fits
                        
                        # Write the REAL W and B grids
                        u.grid_to_fits_wrapper(
                            W_i_ell_m_RE,self.mask,
                            tempDir+'W_{}_{}_{}_RE_p{}_of_{}.fits'.format(
                                s_idx,self.ell_step[ell_idx],m_step[m_idx],
                                self.box_idx+1,self.total_boxes))
                        if self.rand_save:
                            u.grid_to_fits_wrapper(
                                (self.nR/self.nD)*B_i_ell_m_RE,
                                self.mask,tempDirRand+
                                'B_{}_{}_{}_RE_p{}_of_{}.fits'.format(
                                    s_idx,self.ell_step[ell_idx],m_step[m_idx],
                                    self.box_idx+1,self.total_boxes))
                        
                    elif ftype == 'npy':
                        # If the requested grid filetype is npy
                        
                        # Write the REAL W and B grids
                        np.save(tempDir+'W_{}_{}_{}_RE_p{}_of_{}.npy'.format(
                            s_idx,self.ell_step[ell_idx],m_step[m_idx],
                            self.box_idx+1,self.total_boxes),
                            W_i_ell_m_RE*self.mask)
                        if self.rand_save:
                            np.save(tempDirRand+
                                    'B_{}_{}_{}_RE_p{}_of_{}.npy'.format(
                                    s_idx,self.ell_step[ell_idx],m_step[m_idx],
                                    self.box_idx+1,self.total_boxes),
                                (self.nR/self.nD)*B_i_ell_m_RE*self.mask)
                    
                    if m_step[m_idx] != 0:
                        # If the m value is not 0
                        
                        if ftype == 'fits':
                            # If the requested grid filetype is fits
                            
                            # Write the IMAG W and B grids
                            u.grid_to_fits_wrapper(
                                W_i_ell_m_IM,self.mask,
                                tempDir+'W_{}_{}_{}_IM_p{}_of_{}.fits'.format(
                                    s_idx,self.ell_step[ell_idx],m_step[m_idx],
                                    self.box_idx+1,self.total_boxes))
                            if self.rand_save:
                                u.grid_to_fits_wrapper(
                                    (self.nR/self.nD)*B_i_ell_m_IM,
                                    self.mask,tempDirRand+
                                    'B_{}_{}_{}_IM_p{}_of_{}.fits'.format(
                                        s_idx,self.ell_step[ell_idx],
                                        m_step[m_idx],
                                        self.box_idx+1,self.total_boxes))
                            
                        elif ftype == 'npy':
                            # If the requested grid filetype is npy
                            
                            # Write the IMAG W and B grids
                            np.save(
                                tempDir+'W_{}_{}_{}_IM_p{}_of_{}.npy'.format(
                                    s_idx,self.ell_step[ell_idx],m_step[m_idx],
                                    self.box_idx+1,self.total_boxes),
                                W_i_ell_m_IM*self.mask)
                            if self.rand_save:
                                np.save(tempDirRand+
                                        'B_{}_{}_{}_IM_p{}_of_{}.npy'.format(
                                        s_idx,self.ell_step[ell_idx],
                                        m_step[m_idx],
                                        self.box_idx+1,self.total_boxes),
                                    (self.nR/self.nD)*B_i_ell_m_IM*self.mask)
                            
                    # End file timer and update
                    file_end_time_W = time.perf_counter()
                    self.file_time += file_end_time_W - file_start_time_W
                    
            # PS if initial box
            if self.box_idx == 0:
                
                # PS verb to mark the end of a radial step
                if self.verbose:
                    print('Finished writing files'+
                          ' for radial step s1 = {} Mpc (or Mpc/h)'.format(
                              np.round(
                                  self.desired_s_bin_centers[s_idx],
                                  decimals=2)))
        
        # PS to mark the end of a partition region
        print('Finished partition {} of {}'.format(self.box_idx+1,
                                                   self.total_boxes))
        return
            
    
# =================== SERIES CONVOLUTION STEP DRIVER ======================== #


def ConKerConvolveCatalog(data_file: str, rand_file: str, cfg_file: bool,
                          store_rand: str, ell_max: int, verbose: bool,
                          ftype: str = 'fits'):
    # A function to run the convolution step of the algorithm in series
    # Wraps the ConKerBox radial convolution for every partition
    
    try:
        # Check to make sure the divide plan exists
        # Also grab the total number of boxes
        _, _, _, total_boxes, _ = u.getLOSbox(rand_file, cfg_file, 0)
        
    except FileNotFoundError:
        # If the file isn't found
        
        # Trip a failure message
        print('\n'+'==!==!==!==!==!==!==!==!==!== '+'FAILURE'+
              ' ==!==!==!==!==!==!==!==!==!=='+'\n')
        print('ConKer cannot find the divide scheme!')
        print("Make sure you've run DivideCatalog() "+
              "with save_plan = True")
        print('Requires a partition corresponding to both randoms '+
              'and cfg files')
        print('\n'+'!==!==!==!==!==!==!==!==!==!==!==!=='+
              '!==!==!==!==!==!==!==!==!==!==!'+'\n')
        return
    
    # Create a dictionary of times and initialize each one to 0
    times = {}
    times['T_MAP'] = 0.
    times['T_CONV'] = 0.
    times['T_KERN'] = 0.
    times['T_FILE'] = 0.
    
    for boxID in range(total_boxes):
        # For each of the boxes in the divide plan
        
        # Run the convolution with verbose=verbose if this is partition 0
        if boxID == 0:
            cb = ConKerBox(data_file = data_file,rand_file = rand_file,
                           cfg_file = cfg_file, store_rand = store_rand,
                           box_idx = boxID,
                           verbose = verbose)
            cb.radialConvolveLegendre(ell_max = ell_max,ftype=ftype)
            
        # Run the convolution with verbose=False if this is another partition
        else:
            cb = ConKerBox(data_file = data_file,rand_file = rand_file,
                           cfg_file = cfg_file, store_rand = store_rand,
                           box_idx = boxID,
                           verbose = False)
            cb.radialConvolveLegendre(ell_max = ell_max,ftype=ftype)
            
        # Update the timing trackers with each partition
        try:
            # May fail if a previous failure message has been tripped
            times['T_MAP'] += cb.end_map_time - cb.start_map_time
            times['T_CONV'] += cb.conv_time
            times['T_FILE'] += cb.file_time
            
        except AttributeError:
            # Return if something else has failed in ConKerBox()
            return
        
        # Update the kernel time if this is partions 0
        if boxID == 0:
            times['T_KERN'] += cb.kernel_time
            
    # PS to return the total CPU runtime
    print('ConKer Convolution Step took'+
          ' {} s CPU time'.format(times['T_MAP']+times['T_CONV']+
                                  times['T_KERN']+times['T_FILE']))
    
    # PS verb for timing breakdown by process
    if verbose:
        print('   Mapping time = {} CPU s'.format(times['T_MAP']))
        print('   Convolution time = {} CPU s'.format(times['T_CONV']))
        print('   Kernel creation time = {} CPU s'.format(times['T_KERN']))
        print('   File writing time = {} CPU s'.format(times['T_FILE']))
        
    # Get the name of the dictionary to store timing results
    tempDir = './grids/'+data_file.split('.fits')[0]+\
        '_'+cfg_file.split('.txt')[0]+'/'
           
    # Save the timing breakdown
    with open(tempDir+'timing_info.txt', "w") as file:
        json.dump(times, file)
    
    # PS for final section
    print('\n'+'================================'+
          '==================================='+'\n')
    return
    
    
                