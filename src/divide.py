# =========================================================================== #
"""
Created on Wed Dec 15 13:53:56 2021
@authors: Zachery Brown, Benjamin Levi

Description: This file, divide.py, is used to partition the galaxy catalog
into sections. Each partition corresponds to a rectangular window in the 
RA, DEC coordinates. The partition spans the entire range of Z. Included is 
the DivideScheme object. It's associated function partitionCatalog() has 
options for saving, plotting, and displaying the angular "boxes". Several
warning messages are included for likely mistakes. Combined with the driver 
routine, print statements applicable to jupyter or any python terminal
are included with the "verbose" option. Showing plots works best in jupyter.
"""
# =========================================================================== #

# Retrieve util functions
import src.utils as u

# Required python imports
import time
import json

# numpy required
import numpy as np

# astropy fits required for file reading/writing
from astropy.io import fits

# ============================ DIVIDE PLAN ================================== #


class DivideScheme:
    # Creates a plan for partitioning the catalog into angular pillars
    # Result will be applicable to any data set using these
    #   randoms and parameters plan
    # TODO -> Add hard check that boxes do not stretch beyond spherical limits
    
    
    def __init__(self, rand_file: str, cfg_file: str, verbose: bool):
        # Reads in a randoms file and parameters
        # Sets up print statements
        # Accounts for timing
        
        # Starts the timing
        # Divide plan will be timed as a single process
        self.start_time = time.perf_counter()
        
        # Creates a convenient filename string
        # Essentially just removes the ".fits" extension
        self.fileString = rand_file.split('.fits')[0]
        
        # PS to begin the process
        # Uses the previously defined file string
        print('\n'+'=========================== '+'DIVIDE STEP'+
              ' ==========================='+'\n')
        print('Creating a divide plan for '+self.fileString)
        
        # Reads in the randoms
        self.rand = fits.open('./data/'+rand_file)[1].data
        
        # Checks whether the shiftRA function needs to be applied (SGC data)
        self.shift_condition = (int(((self.rand['ra'].max()-
                                      self.rand['ra'].min())//1)+1) == 360)

        # Uncommment the line below as a hot-fix if the shift condition
        #  is not working properly
        #self.shift_condition = True
        
        # Shift if the numerical min/max of RA are approximately 
        #   0 and 360 respectively
        if self.shift_condition == True:
            self.rand['ra'] = u.shift_RA(self.rand['ra'])
            
        # Opens the configuration file and stores params 
        #   necessary for the divide plan
        # Inlcudes cosmology and s bin properties
        # TODO -> non-flat cosmology options
        self.cfg_file = cfg_file
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
        
        # Sets the verbose parameter
        self.verbose = verbose
        
        # PS verb to mark that cosmology and binning have been initialized
        if self.verbose:
            print('Cosmology and binning parameters initialized...')
            
            
    def partitionCatalog(self, save_plan: bool,
                         plot_results: bool, save_plot: bool, theta_user = None):
        # This function actually creates the partition
        # User options for saving the plan, plotting, and saving the plot
        # WARNING -> Plotting works best in jupyter
        
        if (plot_results == False)&(save_plot == True):
            # Immediately trip a failure message if the user wishes to save 
            #   a plot but did not create it by specifying plot_result = True
            print('\n'+'==!==!==!==!==!==!==!==!==!== '+'FAILURE'+
                  ' ==!==!==!==!==!==!==!==!==!=='+'\n')
            print('You must run the plotting routine to save the sky plot')
            print('Retry DivideCatalog() with plot_results = True')
            print('\n'+'!==!==!==!==!==!==!==!==!==!==!==!=='+
                  '!==!==!==!==!==!==!==!==!==!==!'+'\n')
            return
        
        # Initalizes a list for the lines of sight (LOS) boxes and their
        #   corresponding RA and DEC limits
        self.LOS_boxes = []
        
        if theta_user is not None:
            self.theta_P = theta_user
        
        # Optional theta_user patameter
        # Over-writes the automated theta_p corresponding to the partition size
        elif theta_user == None:
            # Define theta_P with the maximum s bin edge plus the grid size
            self.theta_P = np.degrees((self.desired_s_bin_edges.max()+self.g_s)/
                                      u.z2r(self.rand['z'].min(),self.cosmo))
        
        if 100*(np.radians(self.theta_P)**2/2) > 5.:
            # Include a failure for improper partitions
            # If \theta_P**2/2 is larger than 0.95%, trip warning
            # Temporarily set the error limit to: 5% (User may change this)
            # 0.95% warning ensures no errors at this step. This is the "safe"
            #   option when creating a divide plan
            # TODO -> Investigate errors here
            print('\n'+'==!==!==!==!==!==!==!==!==!== '+'FAILURE'+
                  ' ==!==!==!==!==!==!==!==!==!=='+'\n')
            print('Your maximum s-bin is too large to safely partition!')
            print('Try again with smaller s-bins or increase your '+
                  'minimum redshift cut')
            print('\n'+'!==!==!==!==!==!==!==!==!==!==!==!=='+
                  '!==!==!==!==!==!==!==!==!==!==!'+'\n')
            return
        
        # Define the overall limits in RA and DEC for this catalog
        # Introduces one extra degree in either direction of RA and DEC limits
        # This allows a one degree mismatch between the data and randoms mask
        self.ra_lims = (self.rand['ra'].min()-1.,self.rand['ra'].max()+1.)
        self.dec_lims = (self.rand['dec'].min()-1.,self.rand['dec'].max()+1.)
        
        # Defines the DEC bins applicable to all rows 
        # First gets the edges, then the centers
        dec_bin_edges = u.makeBinEdges(self.dec_lims,2*self.theta_P)
        dec_bin_centers = u.edge2cen(dec_bin_edges)
        
        for dec_idx in range(len(dec_bin_centers)):
            # Loops through each row
            # TODO -> Check to ensure all pairs are mapped here
            
            # Gets the value of cos(DEC) at bin center
            conv_bins_cos_dec_adjust = u.cosDeg(dec_bin_centers[dec_idx])
            
            
            # Gets the value of min(cos(DEC)) for the whole bin
            map_bins_cos_dec_adjust=min([u.cosDeg(dec_bin_edges[dec_idx]),
                                         u.cosDeg(dec_bin_edges[dec_idx+1])])
            
            # Makes RA bins adjusting the width for cos(DEC)
            # First gets the edges, then the centers
            ra_bin_edges_row = u.makeAlphaBins((dec_bin_edges[dec_idx],
                                                dec_bin_edges[dec_idx+1]),
                                               conv_bins_cos_dec_adjust,
                                               self.rand,self.theta_P)
            ra_bin_centers_row = u.edge2cen(ra_bin_edges_row)
            
            for ra_idx in range(len(ra_bin_centers_row)):
                # Loops through each RA bin in this row
                
                # Defines the proposed LOS in angular coordinates
                LOS = (ra_bin_centers_row[ra_idx],dec_bin_centers[dec_idx])
                
                # Sets the limits on RA and DEC for the convolution boxes
                # Stores all limit pairs as tuples
                conv_box_ra = (ra_bin_edges_row[ra_idx],
                               ra_bin_edges_row[ra_idx+1])
                conv_box_dec = (dec_bin_edges[dec_idx],
                                dec_bin_edges[dec_idx+1])
                
                # Sets the limits on RA and DEC for the mapping boxes
                # Allows an additional theta_P or \theta_P/cos(DEC) 
                #   on each side to avoid missing pairs
                map_box_ra = (ra_bin_centers_row[ra_idx]-
                              2*(self.theta_P/map_bins_cos_dec_adjust),
                              ra_bin_centers_row[ra_idx]+
                              2*(self.theta_P/map_bins_cos_dec_adjust))
                map_box_dec = (dec_bin_edges[dec_idx]-
                               self.theta_P,dec_bin_edges[dec_idx+1]+
                               self.theta_P)
                
                # Appends the list of LOS boxes with the LOS coordinates
                #   and box lims
                self.LOS_boxes.append((LOS,
                                       (conv_box_ra,conv_box_dec),
                                       (map_box_ra,map_box_dec)))
                
        # PS verb to mark the completion of the angular partitioning
        if self.verbose:
            print('Created {} total angular boxes...'.format(
                len(self.LOS_boxes)))
            
        # Finish time stamp
        self.end_time = time.perf_counter()
            
        if (save_plan == False)&(plot_results == False):
            # If the user will not save the plan, or plot, print results
            
            # PS ending statements including times
            print('Divide Step took {} s CPU time'.format(
                self.end_time-self.start_time))
            print('\n'+'==================================='+
                  '================================'+'\n')
            return
            
        if save_plan == True:
            # If the user wishes the save the divide plan
            # Using the names and data locations, write the divide plan
            # Creates a .fits file to be used as the save plan
            
            # PS verb to mark the file prep
            if self.verbose:
                print('Preparing the divide plan file...')
                
            # Create lists of indices for the LOS and box limits
            LOS_ids = [(0,0),(0,1)]
            BOX_ids = [(1,0,0),(1,0,1),(1,1,0),(1,1,1),
                       (2,0,0),(2,0,1),(2,1,0),(2,1,1)]
            
            # Create a list of column names for the file entries
            col_names = ['LOS_ra','LOS_dec','cb_ra_min','cb_ra_max',
                         'cb_dec_min','cb_dec_max','mb_ra_min','mb_ra_max',
                         'mb_dec_min','mb_dec_max']
            
            # Initialize a list of fits columns
            fits_columns = []
            
            for col_i in range(len(col_names)):
                # Loop over each column index
                
                if (col_i == 0)|(col_i == 1):
                    # If either the LOS RA or LOS DEC column (special indices)
                    
                    # Append the columns list with a fits column containing
                    #   either LOS RA or LOS DEC
                    fits_columns.append(
                        fits.Column(name=col_names[col_i],array=np.asarray(
                            [self.LOS_boxes[box_idx][LOS_ids[col_i][0]]
                             [LOS_ids[col_i][1]] 
                             for box_idx in range(len(self.LOS_boxes))]),
                            format='E'))
                    
                else:
                    # Else, use box indices
                    
                    # Append the columns list with a fits column containing 
                    #   mapping or convolution box limits
                    # i-2 index because of LOS indices list
                    fits_columns.append(
                        fits.Column(name=col_names[col_i],array=np.asarray(
                            [self.LOS_boxes[box_idx][BOX_ids[col_i-2][0]]
                             [BOX_ids[col_i-2][1]][BOX_ids[col_i-2][2]] 
                             for box_idx in range(len(self.LOS_boxes))]),
                            format='E'))
                    
            # Collect columns and create a .fits table
            self.cols = fits_columns
            divide_table = fits.BinTableHDU.from_columns(fits_columns)
            
            # Header contains 3 values, as follows:
            # 'BCTOT': total number of angular boxes
            # 'SHIFT_RA': whether or not to shift RA values (SGC galaxies)
            # 'T_DIVP': the total CPU time for this step in sec
            divide_table.header.set('BCTOT',len(self.LOS_boxes))
            divide_table.header.set('SHIFT_RA',self.shift_condition)
            divide_table.header.set('DIV_TIME',self.end_time-self.start_time)
            
            # File is written to the 'divide/plans' directory
            divide_table.writeto('./divide/plans/'+self.fileString+
                                 '_'+self.cfg_file.split('.txt')[0]+
                                 '.fits',overwrite=True)
            
            if plot_results == False:
                # If the user will not do anything more
                
                # PS ending statements including times
                print('Divide Step took {} s CPU time'.format(
                    self.end_time-self.start_time))
                print('\n'+'==================================='+
                      '================================'+'\n')
                return
                
        if plot_results == True:
            # If the user wishes to plot the results of the divde step
            # Includes the option to save the plot to file
            
            # PS verb if user is making plot
            if self.verbose:
                print('Plotting angular boxes now...')
                
            # Perform plt imports and settings changes only if plot 
            #   will be made
            # WARNING -> usetex=True requires a native LATEX distribution!
            import matplotlib.pyplot as plt
            plt.rc('text', usetex=True)
            plt.rc('font', family='serif')
            plt.rc('savefig', bbox = 'tight')
            
            # Sets the plot title and initializes the figure
            # Title defaults to a string of the randoms filename
            #   before the first underscore!
            title=self.fileString.split('_')[0]+' Divide Step Boxes'
            fig = plt.figure(figsize=(9, 6))
            
            # Initializes the axes for the plot and sets the ticks
            # Includes a util function for the shift condition
            ax = fig.add_subplot(111, projection='mollweide',
                                 facecolor ='ghostwhite')
            ra_ticks, org = u.ticksAndOrg(self.shift_condition)
            ax.set_xticklabels(ra_ticks)
            ax.title.set_fontsize(15)
            ax.set_xlabel('Right Ascension')
            ax.xaxis.label.set_fontsize(12)
            ax.set_ylabel('Declination')
            ax.yaxis.label.set_fontsize(12)
            
            # Reduces the size of the randoms catalog to plot and 
            #   gets their coordinates
            red_idx = u.reduceSample(self.rand['ra'],500) 
            galRA, galDEC = u.shiftForMoll(
                self.rand['ra'][red_idx],self.rand['dec'][red_idx],org)
            
            # Plot the galaxies and include the title
            ax.scatter(galRA,galDEC,marker='*',s=0.5,alpha=0.7,color='silver')
            ax.set_title(title)
            
            for box_idx in range(len(self.LOS_boxes)):
                # Loops though each LOS box
                
                # Gets the RA and DEC for the LOS and plots it with an X
                los_ra, los_dec = u.shiftForMollPoint(
                    self.LOS_boxes[box_idx][0][0],
                    self.LOS_boxes[box_idx][0][1],org)
                ax.plot(los_ra, los_dec, marker='x',c='b',alpha=0.9)
                
                # Gets the RA and DEC of the convolution box and 
                #   plots it in red
                cb_ra, cb_dec = u.shiftForMoll(
                    u.plotBox(self.LOS_boxes[box_idx][1])[0],
                    u.plotBox(self.LOS_boxes[box_idx][1])[1],org)
                ax.plot(cb_ra, cb_dec, c='r', ls='-',lw=1,alpha=0.7)
                
            if save_plot == True:
                # If the user wishes the save the plot
                
                # PS verb to give the location of the figure
                if self.verbose:
                    print('Saving sky plot in ./divide/figures/...')
                    
                # Saves the plot using the name of the randoms and 
                #   the parameter set
                plt.savefig('./divide/figures/'+self.fileString+'_'+
                            self.cfg_file.split('.txt')[0]+'_sky_view.png',
                            dpi=150)
                
            # PS ending statements including times
            print('Divide Step took {} s CPU time'.format(
                self.end_time-self.start_time))
            print('\n'+'==================================='+
                  '================================'+'\n')
            return
                
            
# ========================= DIVIDE STEP DRIVER ============================== #
                
                
def DivideCatalog(rand_file: str, cfg_file: str, save_plan: bool,
                  plot_results: bool, save_plot: bool, verbose: bool, theta_user = None):
    # A function to run the divide plan routines with a user defined set of 
    #   parameters. Opttional params for saving, plotting, etc.
    
    # Initialize the plan by specifying randoms and cfg files
    divp = DivideScheme(rand_file=rand_file,
                        cfg_file=cfg_file,
                        verbose=verbose)
    
    # Run the partition scheme and decide what to do with the results
    divp.partitionCatalog(save_plan=save_plan,
                          plot_results=plot_results,
                          save_plot=save_plot,
                          theta_user=theta_user)
    return            
                
            
            
