# =========================================================================== #
"""
Created on Thu Feb 10 15:48:11 2022
@authors: Zachery Brown

Description: This driver is used to store parameters and oversee calculations 
of the npcf. Each npcf convolution mode corresponds to individual routines. 
The divide() function is applicable to all modes.
"""
# =========================================================================== #

# Retrieve ConKer routines
import src.print_statements as ps
import src.utils as u
from src.divide import DivideScheme
from src.kernel_writer import KernelWriter
from src.convolve_series import ConKerBox
from src.wrap_results import MuWedgeWrapper

# astropy fits required for file handling
from astropy.io import fits

# Required python imports
import json
import os

# numpy required
import numpy as np

# ====================== CORRELATION DRIVER ================================= #


class Correlation:
    # An object to support calculation of the npcf with ConKer
    # Stores parameters globally and tracks timing
    # Runs each step of the algorithm
    
    
    def __init__(self, data_file: str, data_type: str, cfg_file: str,
                 randoms_file: str = None, verbose: bool = False):
        # Store the user defined parameters and check functionality
        
        # Set up print statments
        self.verbose = verbose
        
        # Set data file and initial print statements
        self.data_file = data_file
        ps.req_corr_initialize(self.data_file)
        
        # Check to make sure data file exists and print the number of 
        #   objects in the requested catalog
        try:
            self.data = fits.open('./data/'+self.data_file)[1].data
            self.n_obj = len(self.data)
            ps.req_count_obj(self.n_obj)
        except FileNotFoundError:
            ps.fail_no_file(self.data_file)
            return
        
        # Check to make sure the requested data type is supported by
        #   this version of ConKer. Add to the supported_types list when
        #   implementing new funcionality
        supported_types = ['galaxy','lyman_alpha_1D']
        self.data_type = data_type
        if self.data_type not in supported_types:
            ps.fail_unsupported_data_type(self.data_type,supported_types)
            return
        ps.req_catalog_type(self.data_type)
        
        if self.data_type == 'galaxy':
            # If galaxies, look for corresponding randoms
            
            # Check that randoms have been requested
            if randoms_file == None:
                ps.fail_no_randoms(self.data_type)
                return
            
            # Set randoms file
            self.rand_file = randoms_file
            ps.req_rand_file(self.rand_file)
            
            # Check that the randoms exist
            try:
                self.n_obj = len(fits.open(
                    './data/'+self.rand_file)[1].data)
                ps.req_count_obj(self.n_obj)
            except FileNotFoundError:
                ps.fail_no_file(self.rand_file)
                return
            
        # Set the cfg file and check to make sure it exists
        self.cfg_file = cfg_file
        ps.req_cfg_file(self.cfg_file)
        try:
            with open('./params/'+self.cfg_file) as cfgDict:
                self.cfg_set = json.load(cfgDict)
        except FileNotFoundError:
            ps.fail_no_cfg(self.cfg_file)
            return
        
        # Set cosmology and s bin properties
        # TODO -> non-flat cosmology options
        # Sets the s-bins using the cfg file
        # Uses the s-bins to store the grid size g_S
        self.cosmo = (self.cfg_set['c'],self.cfg_set['H0'],
                      (self.cfg_set['OmM'],0.,1.-self.cfg_set['OmM']))
        self.desired_s_bin_centers = np.linspace(
            self.cfg_set['sMin'],self.cfg_set['sMax'],self.cfg_set['sBinN'])
        self.desired_s_bin_edges = u.cen2edge(self.desired_s_bin_centers)
        self.g_s = self.desired_s_bin_centers[1]-self.desired_s_bin_centers[0]
        if self.verbose:
            ps.verb_cosmo_readout(self.cosmo)
            ps.verb_binning_readout(self.desired_s_bin_centers,
                                    self.desired_s_bin_edges,self.g_s)
            
        # Checks whether the shiftRA function needs to be applied (SGC data)
        self.shift_condition = (int(((self.data['ra'].max()-
                                      self.data['ra'].min())//1)+1) == 360)
        
        # Mark initial step completion and set which file will be partitioned
        if self.data_type == 'galaxy':
            self.div_file = self.rand_file
        elif self.data_type == 'lyman_alpha_1D':
            self.div_file = self.data_file
        ps.req_corr_init_done()
        return
    
    
    def divide(self, save_plan: bool, plot_results: bool, save_plot: bool,
               theta_p: float = None):
        # A function to run the divide plan routines with a user defined
        #   set of parameters. Optional params for saving, plotting, etc.
        
        # Begin
        ps.req_div_begin(self.div_file)
        
        # Set divide params
        self.save_plan = save_plan
        self.plot_results = plot_results
        self.save_plot = save_plot
        self.theta_p = theta_p
        
        # Initialize the divide scheme
        div_plan = DivideScheme(self)
        
        # Deal with failures for not plotting
        if (self.plot_results == False)&(self.save_plot == True):
            ps.fail_plot_div()
            return
            
        # Run the divide routine
        div_plan.partitionCatalog(self)
        return

    
    def convolve_2pcf_mu_wedges(self, mu_edges: np.array,
                                store_rand: bool = True):
        # A driver routine for the convolution step in mu wedge 2pcf mode
        # This function runs each partition in series
        
        # Begin and make sure div plan is there
        # Also grab the total number of boxes
        ps.req_begin_conv(self.data_file)
        try:
            _, _, _, total_boxes, _ = u.getLOSbox(
                self.div_file, self.cfg_file, 0)
        except FileNotFoundError:
            ps.fail_no_div_plan()
            return
        
        # Create a dictionary of times and initialize each one to 0
        times = {}
        times['T_KERN'] = 0.
        times['T_MAP'] = 0.
        times['T_CONV'] = 0.
        times['T_FILE'] = 0.
        
        # Make sure there wont be issues with file overwriting
        try:
            os.makedirs('./conv/'+self.data_file.split('.fits')[0]+
                        '_'+self.cfg_file.split('.txt')[0]+
                        '_mu_wedge_2pcf_conv')
        except FileExistsError:
            ps.fail_conv_dir_overwrite()
            return
        self.store_rand = store_rand
        if (self.store_rand == True)&(self.data_type == 'galaxy'):
            try:
                os.makedirs('./conv/'+self.rand_file.split('.fits')[0]+
                            '_'+self.cfg_file.split('.txt')[0]+
                            '_mu_wedge_2pcf_conv')
            except FileExistsError:
                ps.fail_conv_dir_overwrite()
                return
        
        # List mu bin properties
        # Mark the begining of the kernel writing procedure
        self.mu_bin_edges = mu_edges
        ps.req_mu_bin_readout(self.mu_bin_edges)
        if self.verbose:
            ps.verb_kernel_begin()
        
        # Write kernels and time
        kernels = KernelWriter(self)
        kernels.write_2pcf_mu_wedges(self)
        times['T_KERN'] += kernels.time
        if self.verbose:
            ps.verb_kernel_end()
            
        for boxID in range(total_boxes):
            # For each of the boxes in the divide plan

            # Run the convolution 
            cb = ConKerBox(self, box_idx = boxID)
            cb.mapToDensityField(self)
            times['T_MAP'] += cb.map_time
            times['T_FILE'] += cb.file_time
            cb.radialConvolveMuWedges(self)
            times['T_KERN'] += cb.kernel_time
            times['T_CONV'] += cb.conv_time
            times['T_FILE'] += cb.file_time
            
        # PS to return the total CPU runtime
        ps.req_conv_tot_time(np.round(times['T_KERN']+times['T_MAP']+
                                      times['T_FILE']+times['T_CONV'],
                                      decimals=4))
        if self.verbose:
            ps.verb_conv_time_breakdown(
                np.round(times['T_KERN'],decimals=4),
                np.round(times['T_MAP'],decimals=4),
                np.round(times['T_CONV'],decimals=4),
                np.round(times['T_FILE'],decimals=4))

        # Save the timing breakdown
        with open(cb.data_loc+'timing_info.txt', "w") as file:
            json.dump(times, file)

        # PS for final section
        ps.req_bar()
        return
    
    
    def wrap_2pcf_mu_wedges(self, mu_edges: np.array, 
                            plot_correlation: bool,
                            save_correlation_plot: bool):
        # A function to sum over or average over grids for the 2pcf
        # Applicable to the mu wedge convolution mode
        
        # Begin
        ps.req_begin_wrap(self.data_type,self.data_file)

        if (plot_correlation == False)&(save_correlation_plot == True):
            # Immediately trip a failure message if the user wishes to save 
            #   a plot but did not create it by specifying plot_result = True
            ps.fail_plot_wrap()
            return
        
        # Try creating the necessary directory in case it doesn't exist
        # This will be used by all correlation wrapper processes
        # Each one attempts to create this dir
        # Thus, no failure message if it already exists
        try:
            os.makedirs('./corr/'+self.data_file.split('.fits')[0]+
                        '_'+self.cfg_file.split('.txt')[0])
        except:
            FileExistsError
        
        # Wrap the 2pcf
        MuWedgeWrapper(self,mu_edges,plot_correlation,save_correlation_plot)
        return


