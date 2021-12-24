# =========================================================================== #
"""
Created on Thu Dec 16 10:10:31 2021
@authors: Zachery Brown


Description: This file, summation.py is used to run the final steps of the 
Divide+ConKer algorithm. It requires a directory of grid files, saved by 
running the convolution step. Any ell_max arguments <= than the one used 
in convolution. Each routine sums over and packages the NPCF for a unique N.
"""
# =========================================================================== #

# Retrieve util functions
import src.utils as u

# Required python imports
import time
import json
import os

# numpy required
import numpy as np

# astropy fits required for file reading/writing
from astropy.io import fits

# ========================= 2PCF SUMMATION STEP ============================= #


def Summation2PCF(data_file: str, rand_file: str, cfg_file: str, ell_max: int,
                  plot_correlation: bool, save_plot: bool, verbose: bool):
    # A function to sum over grids and compute the Legendre multipoles of 
    #   the 2PCF
    # Requires data and cfg to find the grids, rand to find the divide plen
    # ell_max must be <= ell_max in convolution grid
    # Computes even mltipoles
    # Options to plot and save 2PCF multipoles
    
    # Specify the grid file directory and data name
    datafileString = data_file.split('.fits')[0]
    randfileString = rand_file.split('.fits')[0]
    tempDir = './grids/'+datafileString+'_'+cfg_file.split('.txt')[0]+'/'
    tempDirRand = './grids/'+randfileString+'_'+cfg_file.split('.txt')[0]+'/'

    # PS to begin routine
    print('\n'+'=================== '+'CONKER 2PCF SUMMATION STEP'+
          ' ===================='+'\n')
    print('Summing over grids for '+datafileString)
            
    if (plot_correlation == False)&(save_plot == True):
        # Immediately trip a failure message if the user wishes to save 
        #   a plot but did not create it by specifying plot_result = True
        print('\n'+'==!==!==!==!==!==!==!==!==!== '+'FAILURE'+
              ' ==!==!==!==!==!==!==!==!==!=='+'\n')
        print('You must run the plotting routine '+
              'to save the correlation plot(s)')
        print('Retry Summation2PCF() with plot_correlation = True')
        print('\n'+'!==!==!==!==!==!==!==!==!==!==!==!=='+
              '!==!==!==!==!==!==!==!==!==!==!'+'\n')
        return
    
    try:
        # Try to read in a kernel grid at ell_max to check that the previous 
        #   step has been completed, Delete if successful
        ktest = np.load(tempDir+'K_0_{}_0_RE.npy'.format(ell_max))
        del ktest
        
    except FileNotFoundError:
        # Trip a failure message if the user wishes to scompute the 2PCF
        #   multipoles up to some ell_max beyond the ones saved in grids
        print('\n'+'==!==!==!==!==!==!==!==!==!== '+'FAILURE'+
              ' ==!==!==!==!==!==!==!==!==!=='+'\n')
        print('ConKer cannot find your convolution files')
        print('This may be for two reasons:')
        print('   1: You DID NOT run the convolution step')
        print('   2: You DID run the convolution step, but '+
              'at a lower ell_max')
        print('Either way, run the convolution step again!')
        print('\n'+'!==!==!==!==!==!==!==!==!==!==!==!=='+
              '!==!==!==!==!==!==!==!==!==!==!'+'\n')
        return
    
    try:
        # Looks for two test files to get the correct ftype (data and rand)
        btest = np.load(tempDirRand+'B_0_{}_0_RE.npy'.format(ell_max))
        btest2 = np.load(tempDir+'W_0_{}_0_RE.npy'.format(ell_max))
        del btest, btest2
        ftype = 'npy'
        
    except FileNotFoundError:
        # If it wasn't able to find the npy file, set ftype to fits
        ftype = 'fits'
        
    # Open the cfg file
    with open('./params/'+cfg_file) as cfgDict:
        cfg_set = json.load(cfgDict)
        
    # Set ell list 
    # Assumes odd multipoles are not computed
    ell_step = list(range(0,ell_max+1,2))
    
    # Read in the data and randoms (for normalization)
    data_full = fits.open('./data/'+data_file)[1].data
    rand_full = fits.open('./data/'+rand_file)[1].data
        
    # Get the nD and nR normalization values for the full catalogs
    nD = np.sum(data_full['wts'])
    nR = np.sum(rand_full['wts'])
    
    # Delete the large fits catalogs
    del data_full, rand_full
        
    # Initalize the correlation dictionary for data storage
    corr_dict = {}
    
    # Set s and initialize B to 0
    corr_dict['s'] = np.linspace(
        cfg_set['sMin'],cfg_set['sMax'],cfg_set['sBinN'])
    corr_dict['B'] = np.zeros(len(corr_dict['s']))
    
    # For each ell initialize W_ell
    for ell_idx in range(len(ell_step)):
        corr_dict['W'+str(ell_step[ell_idx])] = np.zeros(len(corr_dict['s']))
        
    # Initialize timing dict
    with open(tempDir+'timing_info.txt') as timeDic:
        timing = json.load(timeDic)
        
    # Grabs timing info from the divide plan
    timing['T_DIVP'] = fits.open('./divide/plans/'+
                                 rand_file.split('.fits')[0]+
                                 '_'+cfg_file.split('.txt')[0]+
                                 '.fits')[1].header['DIV_TIME']
        
    # Initialize file time for summation and file writing
    timing['T_FILESM'] = 0.
    
    # Initialize file time for summation
    timing['T_SUM'] = 0.
    
    # Grab the total partition box number
    _, _, _, total_boxes, _ = u.getLOSbox(rand_file, cfg_file, 0)
    
    # Try creating the necessary directory in case it doesn't exist
    # This will be used by all correlation summation processes
    # Each one attempts to create this dir
    # Thus, no failure message if it already exists
    try:
        os.makedirs('./correlations/'+
                    data_file.split('.fits')[0]+'_'+cfg_file.split('.txt')[0])
    except:
        FileExistsError
        
    for box_idx in range(total_boxes):
        # Loop through each partitioned region
        
        # Start file sum timer
        file_start_time = time.perf_counter()
        
        if ftype == 'npy':
            # If the user saved grids as npy files
        
            # Read in W and B grids
            W_ = np.load(tempDir+'W_p{}_of_{}.npy'.format(
                box_idx+1,total_boxes))
            B_ = (nD/nR)*np.load(tempDirRand+'B_p{}_of_{}.npy'.format(
                box_idx+1,total_boxes))
            
        elif ftype == 'fits':
            # If the user saved grids as fits files
            
            # Read in W and B grids
            W_ = u.fits_to_grid_unwrapper(tempDir+'W_p{}_of_{}.fits'.format(
                box_idx+1,total_boxes))
            B_ = (nD/nR)*u.fits_to_grid_unwrapper(
                tempDirRand+'B_p{}_of_{}.fits'.format(
                    box_idx+1,total_boxes))
        
        # End file sum timer and update
        file_end_time = time.perf_counter()
        timing['T_FILESM'] += file_end_time - file_start_time
        
        for s_idx in range(len(corr_dict['s'])):
            # Loop through each s step
            
            # Start file sum timer
            file_start_time = time.perf_counter()
            
            if ftype == 'npy':
                # If the user saved grids as npy files
                
                # Reaad in the B_1 convolved grid
                # Use ell=0 and RE kernel for B
                B1 = (nD/nR)*np.load(
                    tempDirRand+'B_{}_0_0_RE_p{}_of_{}.npy'.format(
                        s_idx,box_idx+1,total_boxes))
                
            elif ftype == 'fits':
                # If the user saved grids as fits files
                
                # Reaad in the B_1 convolved grid
                # Use ell=0 and RE kernel for B
                B1 = (nD/nR)*u.fits_to_grid_unwrapper(
                    tempDirRand+'B_{}_0_0_RE_p{}_of_{}.fits'.format(
                        s_idx,box_idx+1,total_boxes))
            
            # End file sum timer and update
            file_end_time = time.perf_counter()
            timing['T_FILESM'] += file_end_time - file_start_time
            
            # Start summation timer
            sum_start_time = time.perf_counter()
            
            # Sum over B grids for this kernel size
            corr_dict['B'][s_idx] += np.sum(B_*B1)
            
            # End summation timer and update
            sum_end_time = time.perf_counter()
            timing['T_SUM'] += sum_end_time - sum_start_time
            
            for ell_idx in range(len(ell_step)):
                # Loop through each ell step
                
                # Start file timer
                file_start_time = time.perf_counter()
                
                if ftype == 'npy':
                    # If the user saved grids as npy files
                    
                    # Read in the W_1_ell grid
                    W1ell = np.load(
                        tempDir+'W_{}_{}_0_RE_p{}_of_{}.npy'.format(
                            s_idx,ell_step[ell_idx],box_idx+1,total_boxes))
                    
                elif ftype == 'fits':
                    # If the user saved grids as fits files
                    
                    # Read in the W_1_ell grid
                    W1ell = u.fits_to_grid_unwrapper(
                        tempDir+'W_{}_{}_0_RE_p{}_of_{}.fits'.format(
                            s_idx,ell_step[ell_idx],box_idx+1,total_boxes))
                    
                # End file timer and update
                file_end_time = time.perf_counter()
                timing['T_FILESM'] += file_end_time - file_start_time
                
                # Start summation timer
                sum_start_time = time.perf_counter()
                
                # Create a normalization factor 2*ell+1
                # Correct for the scipy ylm normalization when m=0
                norm = (2*ell_step[ell_idx]+1)*(
                    u.ylm_norm_m0(0)/u.ylm_norm_m0(ell_step[ell_idx]))
                
                # Sum over W_1_ell grids for this kernel size
                corr_dict['W'+
                          str(ell_step[ell_idx])][s_idx] += norm*np.sum(
                              W_*W1ell)
                                
                # End summation timer and update
                sum_end_time = time.perf_counter()
                timing['T_SUM'] += sum_end_time - sum_start_time
                                
        # PS verb to mark the end of a partitioned region
        if verbose:
            print('Finished partition {} of {}'.format(box_idx+1,total_boxes))
            
    for ell_idx in range(len(ell_step)):
        # Loop through each ell step
        
        # Compute the correlation functions
        corr_dict['xi'+str(ell_step[ell_idx])] = corr_dict[
            'W'+str(ell_step[ell_idx])]/corr_dict['B']
    
    # PS verb for file writing
    if verbose:
        print('Writing 2pcf to file...')
        
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
    # Keep file writing/reading time split between convolution and summation
    corr_table.header.set('T_DIVP',timing['T_DIVP'])
    corr_table.header.set('T_MAP',timing['T_MAP'])
    corr_table.header.set('T_CONV',timing['T_CONV'])
    corr_table.header.set('T_KERN',timing['T_KERN'])
    corr_table.header.set('T_FILE',timing['T_FILE'])
    corr_table.header.set('T_FILESM',timing['T_FILESM'])
    corr_table.header.set('T_SUM',timing['T_SUM'])
    
    # Write the corr pkg to file
    # Overwrite is fine here because the routine is fast
    corr_table.writeto('./correlations/'+data_file.split('.fits')[0]+
                       '_'+cfg_file.split('.txt')[0]+'/corr_pkg_2pcf.fits',
                       overwrite=True)
    
    if plot_correlation == False:
        # If the user does not wish to plot results
        
        # PS end and timing breakdown
        print('ConKer 2pcf Summation Step took {} s CPU time'.format(
            timing['T_FILESM']+timing['T_SUM']))
        if verbose:
            print('   Summation time = {} CPU s'.format(timing['T_SUM']))
            print('   File reading time '+
                  '= {} CPU s'.format(timing['T_FILESM']))
        print('\n'+'================================='+
              '=================================='+'\n')
        return
        
    elif plot_correlation == True:
        # If plot is requested
        
        # PS verb to mark plt routine
        if verbose:
            print('Preparing to plot the 2pcf multipoles...')
            
        # Import plt and set params only if needed
        # WARNING -> usetex=True requires a native LATEX distribution!
        import matplotlib.pyplot as plt
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.rc('savefig', bbox = 'tight')
        
        for ell_idx in range(len(ell_step)):
            # For each ell step (even multipoles)
            
            # Create a figure and axes
            fig = plt.figure(figsize=(7, 5))
            ax = fig.add_subplot(111, facecolor ='ghostwhite')
            
            # Plot the 2pcf multipole
            # Error bars are based on "naive" poisson errors from
            #   counts in the B distribution for ell 0
            ax.errorbar(corr_dict['s'],corr_dict['xi{}'.format(
                ell_step[ell_idx])]*corr_dict['s']**2,
                        yerr=(np.sqrt(np.abs(corr_dict['B']))/\
                              corr_dict['B'])*corr_dict['s']**2,
                        xerr=np.ones(len(corr_dict['s']))*\
                            (corr_dict['s'][1]-corr_dict['s'][0])/2,
                        ls='',capsize=3,lw=1,marker='o',
                        ecolor='navy',mec='navy',mfc='skyblue')
                
            # Set the title and axes labels, accounting for all ell
            title = data_file.replace(
                '_', '\_').split('.fits')[0]+r' 2pcf $\ell={}$'.format(
                    ell_step[ell_idx])
            ax.set_title(title)
            ax.set_xlabel(r'$s$ [$h^{-1}$Mpc]')
            ax.xaxis.label.set_fontsize(14)
            ax.set_ylabel(r'$s^2\xi_2^{}$'.format(
                ell_step[ell_idx])+r' [$h^{-2}$Mpc$^{2}$]')
            ax.yaxis.label.set_fontsize(14)
            
            if (save_plot == False)&(ell_step[ell_idx] == ell_step[-1]):
                # If the user does not wish to save the plot
                # Applies to last ell of run
                
                # PS end statement
                print('ConKer 2pcf Summation Step took {} s CPU time'.format(
                    timing['T_SUM']+timing['T_FILESM']))
                
                # PS verb for timing breakdown
                if verbose:
                    print('   Summation time = {} CPU s'.format(
                        timing['T_SUM']))
                    print('   File writing time = {} CPU s'.format(
                        timing['T_FILESM']))
                print('\n'+'==============================='+
                      '===================================='+'\n')
                return
            
            if save_plot == True:
                # If the user will be saving the plot
                
                if ell_idx == 0:
                    # If the first value of ell
                    
                    # PS verb to mark plot saving
                    if verbose:
                        print('Saving plot(s)...')
                        
                # Save each of the figures
                plt.savefig('./correlations/'+data_file.split('.fits')[0]+'_'+
                            cfg_file.split('.txt')[0]+'/plot_2pcf_ell'+
                            str(ell_step[ell_idx])+'.png',dpi=150)
                
                if (ell_step[ell_idx] == ell_step[-1]):
                    # If this is the final ell step
                    
                    # PS end statement
                    print('ConKer 2pcf Summation '+
                          'Step took {} s CPU time'.format(
                              timing['T_SUM']+timing['T_FILESM']))
                    
                    # PS verb for timing breakdown
                    if verbose:
                        print('   Summation time = {} CPU s'.format(
                            timing['T_SUM']))
                        print('   File reading time = {} CPU s'.format(
                            timing['T_FILESM']))
                    print('\n'+'==============================='+
                          '===================================='+'\n')
                    return
        

# ========================= 3PCF SUMMATION STEP ============================= #


def Summation3PCF(data_file: str, rand_file: str, cfg_file: str, ell_max: int,
                  verbose: bool, plot_correlation: bool = False,
                  save_plot: bool = False):
    # A function to sum over grids and compute the Legendre multipoles of 
    #   the 3PCF
    # Requires data and cfg to find the grids, rand to find the divide plen
    # ell_max must be <= ell_max in convolution grid
    # Computes all multipoles and implements an edge correction
    # TODO -> Options to plot and save 3PCF panels
    # Edge correction is NOT explicitly calculated in this function
    # However, it computes the terms needed for it (R_l' terms)
    
    # Specify the grid file directory and data name
    datafileString = data_file.split('.fits')[0]
    randfileString = rand_file.split('.fits')[0]
    tempDir = './grids/'+datafileString+'_'+cfg_file.split('.txt')[0]+'/'
    tempDirRand = './grids/'+randfileString+'_'+cfg_file.split('.txt')[0]+'/'
    
    # PS to begin routine
    print('\n'+'=================== '+'CONKER 3PCF SUMMATION STEP'+
          ' ===================='+'\n')
    print('Summing over grids for '+datafileString)
            
    if (plot_correlation == False)&(save_plot == True):
        # Immediately trip a failure message if the user wishes to save 
        #   a plot but did not create it by specifying plot_result = True
        print('\n'+'==!==!==!==!==!==!==!==!==!== '+'FAILURE'+
              ' ==!==!==!==!==!==!==!==!==!=='+'\n')
        print('You must run the plotting routine '+
              'to save the correlation plot(s)')
        print('Retry Summation3PCF() with plot_correlation = True')
        print('\n'+'!==!==!==!==!==!==!==!==!==!==!==!=='+
              '!==!==!==!==!==!==!==!==!==!==!'+'\n')
        return
    
    try:
        # Try to read in a kernel grid at ell_max to check that the previous 
        #   step has been completed, Delete if successful
        ktest = np.load(tempDir+'K_0_{}_0_RE.npy'.format(ell_max))
        del ktest
        
    except FileNotFoundError:
        # Trip a failure message if the user wishes to compute the 3PCF
        #   multipoles up to some ell_max beyond the ones saved in grids
        print('\n'+'==!==!==!==!==!==!==!==!==!== '+'FAILURE'+
              ' ==!==!==!==!==!==!==!==!==!=='+'\n')
        print('ConKer cannot find your convolution files')
        print('This may be for two reasons:')
        print('   1: You DID NOT run the convolution step')
        print('   2: You DID run the convolution step, but '+
              'at a lower ell_max')
        print('Either way, run the convolution step again!')
        print('\n'+'!==!==!==!==!==!==!==!==!==!==!==!=='+
              '!==!==!==!==!==!==!==!==!==!==!'+'\n')
        return
    
    try:
        # Looks for a test file(s) to get the correct ftype
        btest = np.load(tempDirRand+'B_0_{}_0_RE.npy'.format(ell_max))
        wtest = np.load(tempDir+'W_0_{}_0_RE.npy'.format(ell_max))
        del btest, wtest
        ftype = 'npy'
        
    except FileNotFoundError:
        # If it wasn't able to find the npy file, set ftype to fits
        ftype = 'fits'
        
    # Open the cfg file
    with open('./params/'+cfg_file) as cfgDict:
        cfg_set = json.load(cfgDict)
        
    # Set ell list 
    # Assumes all multipoles are computed
    ell_step = list(range(0,ell_max+1,1))

    # Read in the data and randoms (for normalization)
    data_full = fits.open('./data/'+data_file)[1].data
    rand_full = fits.open('./data/'+rand_file)[1].data
        
    # Get the nD and nR normalization values for the full catalogs
    nD = np.sum(data_full['wts'])
    nR = np.sum(rand_full['wts'])
    
    # Delete the large fits catalogs
    del data_full, rand_full
    
    # Initialize a dict to store W and B before edge correction
    dist_dict = {}
    
    # Gets arrays of bins s1 an s2 (unique pairs)
    dist_dict['s1'],dist_dict['s2'] = u.squareRadialBins(
        np.linspace(cfg_set['sMin'],cfg_set['sMax'],cfg_set['sBinN']))
        
    # Get arrays of s1 and s2 indices for the loop
    s1_step, s2_step = u.squareRadialBins(
        np.linspace(0,cfg_set['sBinN']-1,cfg_set['sBinN'],dtype='int'))
    
    # Set initial B to 0
    dist_dict['B'] = np.zeros(len(dist_dict['s1']))
    
    # For each ell initialize W_ell and f_N for edge correction
    for ell_idx in range(len(ell_step)):
        dist_dict['W'+str(ell_step[ell_idx])]=np.zeros(len(dist_dict['s1']))
        dist_dict['R'+str(ell_step[ell_idx])]=np.zeros(len(dist_dict['s1']))
    
    # Initialize timing dict
    with open(tempDir+'timing_info.txt') as timeDic:
        timing = json.load(timeDic)
        
    # Grabs timing info from the divide plan
    timing['T_DIVP'] = fits.open('./divide/plans/'+
                                 rand_file.split('.fits')[0]+
                                 '_'+cfg_file.split('.txt')[0]+
                                 '.fits')[1].header['DIV_TIME']
        
    # Initialize file time for summation and file writing
    timing['T_FILESM'] = 0.
    
    # Initialize file time for summation
    timing['T_SUM'] = 0.
    
    # Grab the total partition box number
    _, _, _, total_boxes, _ = u.getLOSbox(rand_file, cfg_file, 0)
    
    # Try creating the necessary directory in case it doesn't exist
    # This will be used by all correlation summation processes
    # Each one attempts to create this dir
    # Thus, no failure message if it already exists
    try:
        os.makedirs('./correlations/'+
                    data_file.split('.fits')[0]+'_'+cfg_file.split('.txt')[0])
    except:
        FileExistsError
        
    for box_idx in range(total_boxes):
        # Loop through each partitioned region
        
        # Start file sum timer
        file_start_time = time.perf_counter()
        
        if ftype == 'npy':
            # If the user saved grids as npy files
        
            # Read in W and B grids
            W_ = np.load(tempDir+'W_p{}_of_{}.npy'.format(
                box_idx+1,total_boxes))
            B_ = (nD/nR)*np.load(tempDirRand+'B_p{}_of_{}.npy'.format(
                box_idx+1,total_boxes))
            
        elif ftype == 'fits':
            # If the user saved grids as fits files
            
            # Read in W and B grids
            W_ = u.fits_to_grid_unwrapper(tempDir+'W_p{}_of_{}.fits'.format(
                box_idx+1,total_boxes))
            B_ = (nD/nR)*u.fits_to_grid_unwrapper(
                tempDirRand+'B_p{}_of_{}.fits'.format(
                    box_idx+1,total_boxes))
        
        # End file sum timer and update
        file_end_time = time.perf_counter()
        timing['T_FILESM'] += file_end_time - file_start_time
        
        for s_idx in range(len(dist_dict['s1'])):
            # Loop through each unique s pair
            
            # Start file sum timer
            file_start_time = time.perf_counter()
            
            if ftype == 'npy':
                # If the user saved grids as npy files
                
                # Reaad in the B_1 and B_2 convolved grids
                # Use ell=0 and RE kernel for B
                B1 = (nD/nR)*np.load(
                    tempDirRand+'B_{}_0_0_RE_p{}_of_{}.npy'.format(
                        s1_step[s_idx],box_idx+1,total_boxes))
                B2 = (nD/nR)*np.load(
                    tempDirRand+'B_{}_0_0_RE_p{}_of_{}.npy'.format(
                        s2_step[s_idx],box_idx+1,total_boxes))
                
            elif ftype == 'fits':
                # If the user saved grids as fits files
                
                # Reaad in the B_1 and B_2 convolved grids
                # Use ell=0 and RE kernel for B
                B1 = (nD/nR)*u.fits_to_grid_unwrapper(
                    tempDirRand+'B_{}_0_0_RE_p{}_of_{}.fits'.format(
                        s1_step[s_idx],box_idx+1,total_boxes))
                B2 = (nD/nR)*u.fits_to_grid_unwrapper(
                    tempDirRand+'B_{}_0_0_RE_p{}_of_{}.fits'.format(
                        s2_step[s_idx],box_idx+1,total_boxes))
            
            # End file sum timer and update
            file_end_time = time.perf_counter()
            timing['T_FILESM'] += file_end_time - file_start_time
            
            # Start summation timer
            sum_start_time = time.perf_counter()
            
            # Set the B normalization
            B_norm = 4*np.pi
            
            # Sum over B grids for this kernel size
            dist_dict['B'][s_idx] += 2*np.pi*B_norm*np.sum(B_*B1*B2)
            
            # End summation timer and update
            sum_end_time = time.perf_counter()
            timing['T_SUM'] += sum_end_time - sum_start_time
            
            for ell_idx in range(len(ell_step)):
                # Loop through each ell step
                
                # Define the m_steps from -ell_max to ell_max in int steps
                m_step = np.linspace(-ell_step[ell_idx],
                                     ell_step[ell_idx],
                                     2*ell_step[ell_idx]+1,dtype=int)
                
                for m_idx in range(len(m_step)):
                    # Loop through all m values for ell
                    
                    # Start file timer
                    file_start_time = time.perf_counter()
                    
                    if ftype == 'npy':
                        # If the user saved grids as npy files
                    
                        # Read in the W_1 and W_2 convolved grids
                        W1_RE = np.load(
                            tempDir+'W_{}_{}_{}_RE_p{}_of_{}.npy'.format(
                                s1_step[s_idx],ell_step[ell_idx],
                                m_step[m_idx],box_idx+1,total_boxes))
                        W2_RE = np.load(
                            tempDir+'W_{}_{}_{}_RE_p{}_of_{}.npy'.format(
                                s2_step[s_idx],ell_step[ell_idx],
                                m_step[m_idx],box_idx+1,total_boxes))
                        
                        # Reaad in the B_1 and B_2 convolved grids
                        B1_RE = (nD/nR)*np.load(
                            tempDirRand+'B_{}_{}_{}_RE_p{}_of_{}.npy'.format(
                                s1_step[s_idx],ell_step[ell_idx],
                                m_step[m_idx],box_idx+1,total_boxes))
                        B2_RE = (nD/nR)*np.load(
                            tempDirRand+'B_{}_{}_{}_RE_p{}_of_{}.npy'.format(
                                s2_step[s_idx],ell_step[ell_idx],
                                m_step[m_idx],box_idx+1,total_boxes))
                        
                    elif ftype == 'fits':
                        # If the user saved grids as fits files
                        
                        # Read in the W_1 and W_2 convolved grids
                        W1_RE = u.fits_to_grid_unwrapper(
                            tempDir+'W_{}_{}_{}_RE_p{}_of_{}.fits'.format(
                                s1_step[s_idx],ell_step[ell_idx],
                                m_step[m_idx],box_idx+1,total_boxes))
                        W2_RE = u.fits_to_grid_unwrapper(
                            tempDir+'W_{}_{}_{}_RE_p{}_of_{}.fits'.format(
                                s2_step[s_idx],ell_step[ell_idx],
                                m_step[m_idx],box_idx+1,total_boxes))
                        
                        # Read in the B_1 and B_2 convolved grids
                        B1_RE = (nD/nR)*u.fits_to_grid_unwrapper(
                            tempDirRand+'B_{}_{}_{}_RE_p{}_of_{}.fits'.format(
                                s1_step[s_idx],ell_step[ell_idx],
                                m_step[m_idx],box_idx+1,total_boxes))
                        B2_RE = (nD/nR)*u.fits_to_grid_unwrapper(
                            tempDirRand+'B_{}_{}_{}_RE_p{}_of_{}.fits'.format(
                                s2_step[s_idx],ell_step[ell_idx],
                                m_step[m_idx],box_idx+1,total_boxes))
                        
                    if m_step[m_idx] == 0:
                        # If grids are only RE
                        
                        # End file timer and update
                        file_end_time = time.perf_counter()
                        timing['T_FILESM'] += file_end_time - file_start_time
                
                        # Start summation timer
                        sum_start_time = time.perf_counter()

                        # Create a normalization factor 4*pi/2*ell+1
                        norm = 4*np.pi/(2*ell_step[ell_idx]+1)

                        # Sum over W_1_ell, W_2_ell grids for these kernels
                        dist_dict['W'+str(
                            ell_step[ell_idx])][s_idx] += 2*np.pi*norm*np.sum(
                                W_*W1_RE*W2_RE)
                        
                        # Sum over B_1_ell, B_2_ell grids for these kernels
                        dist_dict['R'+str(
                            ell_step[ell_idx])][s_idx] += 2*np.pi*norm*np.sum(
                                B_*B1_RE*B2_RE)

                        # End summation timer and update
                        sum_end_time = time.perf_counter()
                        timing['T_SUM'] += sum_end_time - sum_start_time
                        
                    elif m_step[m_idx] != 0:
                        # If there will be IM and RE grids
                        
                        if ftype == 'npy':
                            # If the user saved grids as npy files

                            # Reaad in the W_1 and W_2 convolved grids
                            W1_IM = np.load(
                                tempDir+'W_{}_{}_{}_IM_p{}_of_{}.npy'.format(
                                    s1_step[s_idx],ell_step[ell_idx],
                                    m_step[m_idx],box_idx+1,total_boxes))
                            W2_IM = np.load(
                                tempDir+'W_{}_{}_{}_IM_p{}_of_{}.npy'.format(
                                    s2_step[s_idx],ell_step[ell_idx],
                                    m_step[m_idx],box_idx+1,total_boxes))
                            
                            # Reaad in the B_1 and B_2 convolved grids
                            B1_IM = (nD/nR)*np.load(
                                tempDirRand+
                                'B_{}_{}_{}_IM_p{}_of_{}.npy'.format(
                                    s1_step[s_idx],ell_step[ell_idx],
                                    m_step[m_idx],box_idx+1,total_boxes))
                            B2_IM = (nD/nR)*np.load(
                                tempDirRand+
                                'B_{}_{}_{}_IM_p{}_of_{}.npy'.format(
                                    s2_step[s_idx],ell_step[ell_idx],
                                    m_step[m_idx],box_idx+1,total_boxes))
                        
                        elif ftype == 'fits':
                            # If the user saved grids as fits files

                            # Reaad in the W_1 and W_2 convolved grids
                            W1_IM = u.fits_to_grid_unwrapper(
                                tempDir+
                                'W_{}_{}_{}_IM_p{}_of_{}.fits'.format(
                                    s1_step[s_idx],ell_step[ell_idx],
                                    m_step[m_idx],box_idx+1,total_boxes))
                            W2_IM = u.fits_to_grid_unwrapper(
                                tempDir+
                                'W_{}_{}_{}_IM_p{}_of_{}.fits'.format(
                                    s2_step[s_idx],ell_step[ell_idx],
                                    m_step[m_idx],box_idx+1,total_boxes))
                            
                            # Reaad in the B_1 and B_2 convolved grids
                            B1_IM = (nD/nR)*u.fits_to_grid_unwrapper(
                                tempDirRand+
                                'B_{}_{}_{}_IM_p{}_of_{}.fits'.format(
                                    s1_step[s_idx],ell_step[ell_idx],
                                    m_step[m_idx],box_idx+1,total_boxes))
                            B2_IM = (nD/nR)*u.fits_to_grid_unwrapper(
                                tempDirRand+
                                'B_{}_{}_{}_IM_p{}_of_{}.fits'.format(
                                    s2_step[s_idx],ell_step[ell_idx],
                                    m_step[m_idx],box_idx+1,total_boxes))
                            
                        # End file timer and update
                        file_end_time = time.perf_counter()
                        timing['T_FILESM'] += file_end_time - file_start_time
                        
                        # Start summation timer
                        sum_start_time = time.perf_counter()

                        # Create a normalization factor 4*pi/2*ell+1
                        norm = 4*np.pi/(2*ell_step[ell_idx]+1)

                        # Sum over W_1_ell, W_2_ell grids for these kernels
                        dist_dict['W'+str(
                            ell_step[ell_idx])][s_idx] += 2*np.pi*norm*np.sum(
                                W_*(W1_RE*W2_RE+W1_IM*W2_IM))
                        
                        # Sum over B_1_ell, B_2_ell grids for edge correction
                        dist_dict['R'+str(
                            ell_step[ell_idx])][s_idx] += 2*np.pi*norm*np.sum(
                                B_*(B1_RE*B2_RE+B1_IM*B2_IM))

                        # End summation timer and update
                        sum_end_time = time.perf_counter()
                        timing['T_SUM'] += sum_end_time - sum_start_time            
        
        # PS verb to mark the end of a partitioned region
        if verbose:
            print('Finished partition {} of {}'.format(box_idx+1,total_boxes))
    
    for ell_idx in range(len(ell_step)):
        # Loop through each ell step
                        
        # Define zeta and f terms for later edge correction
        dist_dict['zeta'+str(ell_step[ell_idx])]=dist_dict[
            'W'+str(ell_step[ell_idx])]/dist_dict['B']
        dist_dict['f'+str(ell_step[ell_idx])]=dist_dict[
            'R'+str(ell_step[ell_idx])]/dist_dict['B']
        
    # PS verb for file writing
    if verbose:
        print('Writing 3pcf to file...')
        
    # Create fits column list for the data
    corr_cols=[]
    
    for key in dist_dict.keys():
        # Loop through each column in the corr pkg
        
        # Write the data to a fits column and append
        corr_cols.append(fits.Column(
            name=key,array=dist_dict[key],format='E'))
        
    # Write the correlation data to a fits table
    corr_table = fits.BinTableHDU.from_columns(corr_cols)
    
    # Append the header with timing information
    # Keep file writing/reading time split between convolution and summation
    corr_table.header.set('T_DIVP',timing['T_DIVP'])
    corr_table.header.set('T_MAP',timing['T_MAP'])
    corr_table.header.set('T_CONV',timing['T_CONV'])
    corr_table.header.set('T_KERN',timing['T_KERN'])
    corr_table.header.set('T_FILE',timing['T_FILE'])
    corr_table.header.set('T_FILESM',timing['T_FILESM'])
    corr_table.header.set('T_SUM',timing['T_SUM'])
    
    # Write the corr pkg to file
    corr_table.writeto('./correlations/'+data_file.split('.fits')[0]+
                       '_'+cfg_file.split('.txt')[0]+'/corr_pkg_3pcf.fits',
                       overwrite=True)
    
    if plot_correlation == False:
        # If the user does not wish to plot results
        
        # PS end and timing breakdown
        print('ConKer 3pcf Summation Step took {} s CPU time'.format(
            timing['T_FILESM']+timing['T_SUM']))
        if verbose:
            print('   Summation time = {} CPU s'.format(timing['T_SUM']))
            print('   File reading time '+
                  '= {} CPU s'.format(timing['T_FILESM']))
        print('\n'+'================================='+
              '=================================='+'\n')
        return
    
    
# ==================== DIAGONAL NPCF SUMMATION STEP ========================= #


def SummationNPCFdiag(data_file: str, rand_file: str, cfg_file: str,
                      n_max: int, verbose: bool,
                      plot_correlation: bool = False, save_plot: bool = False):
    # A function to sum over grids and compute the diagonal elements of 
    #   the NPCF
    # Requires data and cfg to find the grids, rand to find the divide plen
    # n_max sets the max correlation order
    # TODO -> options for plotting and saving diagonal npcf
    
    # Specify the grid file directory and data name
    datafileString = data_file.split('.fits')[0]
    randfileString = rand_file.split('.fits')[0]
    tempDir = './grids/'+datafileString+'_'+cfg_file.split('.txt')[0]+'/'
    tempDirRand = './grids/'+randfileString+'_'+cfg_file.split('.txt')[0]+'/'

    # PS to begin routine
    print('\n==================== '+'CONKER DIAG. NPCF SUMMATION STEP'+
          ' ========================='+'\n')
    print('Summing over grids for '+datafileString)
            
    if (plot_correlation == False)&(save_plot == True):
        # Immediately trip a failure message if the user wishes to save 
        #   a plot but did not create it by specifying plot_result = True
        print('\n'+'==!==!==!==!==!==!==!==!==!== '+'FAILURE'+
              ' ==!==!==!==!==!==!==!==!==!=='+'\n')
        print('You must run the plotting routine '+
              'to save the correlation plot(s)')
        print('Retry SummationNPCFdiag() with plot_correlation = True')
        print('\n'+'!==!==!==!==!==!==!==!==!==!==!==!=='+
              '!==!==!==!==!==!==!==!==!==!==!'+'\n')
        return
    
    try:
        # Try to read in a kernel grid to check that the previous 
        #   step has been completed, Delete if successful
        ktest = np.load(tempDir+'K_0_0_0_RE.npy')
        del ktest
        
    except FileNotFoundError:
        # Trip a failure message if the grid files cannot be found
        print('\n'+'==!==!==!==!==!==!==!==!==!== '+'FAILURE'+
              ' ==!==!==!==!==!==!==!==!==!=='+'\n')
        print('ConKer cannot find your convolution files')
        print('Run the convolution step again!')
        print('\n'+'!==!==!==!==!==!==!==!==!==!==!==!=='+
              '!==!==!==!==!==!==!==!==!==!==!'+'\n')
        return
    
    try:
        # Looks for two test files to get the correct ftype (data and rand)
        btest = np.load(tempDirRand+'B_0_0_0_RE.npy')
        btest2 = np.load(tempDir+'W_0_0_0_RE.npy')
        del btest, btest2
        ftype = 'npy'
        
    except FileNotFoundError:
        # If it wasn't able to find the npy file, set ftype to fits
        ftype = 'fits'
        
    # Open the cfg file
    with open('./params/'+cfg_file) as cfgDict:
        cfg_set = json.load(cfgDict)

    # Read in the data and randoms (for normalization)
    data_full = fits.open('./data/'+data_file)[1].data
    rand_full = fits.open('./data/'+rand_file)[1].data
        
    # Get the nD and nR normalization values for the full catalogs
    nD = np.sum(data_full['wts'])
    nR = np.sum(rand_full['wts'])
    
    # Delete the large fits catalogs
    del data_full, rand_full
        
    # Initalize the correlation dictionary for data storage
    corr_dict = {}
    
    # Set s bins
    corr_dict['s'] = np.linspace(
        cfg_set['sMin'],cfg_set['sMax'],cfg_set['sBinN'])
    
    # Set n-step
    n_step = np.linspace(2,n_max,n_max-1,dtype='int')
    
    # Up to correlation order n_max, initialize W and B grids
    for n_idx in range(len(n_step)):
        corr_dict['W_n{}'.format(n_step[n_idx])]=np.zeros(len(corr_dict['s']))
        corr_dict['B_n{}'.format(n_step[n_idx])]=np.zeros(len(corr_dict['s']))
        
    # Initialize timing dict
    with open(tempDir+'timing_info.txt') as timeDic:
        timing = json.load(timeDic)
        
    # Grabs timing info from the divide plan
    timing['T_DIVP'] = fits.open('./divide/plans/'+
                                 rand_file.split('.fits')[0]+
                                 '_'+cfg_file.split('.txt')[0]+
                                 '.fits')[1].header['DIV_TIME']
        
    # Initialize file time for summation and file writing
    timing['T_FILESM'] = 0.
    
    # Initialize file time for summation
    timing['T_SUM'] = 0.
    
    # Grab the total partition box number
    _, _, _, total_boxes, _ = u.getLOSbox(rand_file, cfg_file, 0)
    
    # Try creating the necessary directory in case it doesn't exist
    # This will be used by all correlation summation processes
    # Each one attempts to create this dir
    # Thus, no failure message if it already exists
    try:
        os.makedirs('./correlations/'+
                    data_file.split('.fits')[0]+'_'+cfg_file.split('.txt')[0])
    except:
        FileExistsError
        
    for box_idx in range(total_boxes):
        # Loop through each partitioned region
        
        # Start file sum timer
        file_start_time = time.perf_counter()
        
        if ftype == 'npy':
            # If the user saved grids as npy files
        
            # Read in W and B grids
            W_ = np.load(tempDir+'W_p{}_of_{}.npy'.format(
                box_idx+1,total_boxes))
            B_ = (nD/nR)*np.load(tempDirRand+'B_p{}_of_{}.npy'.format(
                box_idx+1,total_boxes))
            
        elif ftype == 'fits':
            # If the user saved grids as fits files
            
            # Read in W and B grids
            W_ = u.fits_to_grid_unwrapper(tempDir+'W_p{}_of_{}.fits'.format(
                box_idx+1,total_boxes))
            B_ = (nD/nR)*u.fits_to_grid_unwrapper(
                tempDirRand+'B_p{}_of_{}.fits'.format(
                    box_idx+1,total_boxes))
        
        # End file sum timer and update
        file_end_time = time.perf_counter()
        timing['T_FILESM'] += file_end_time - file_start_time
        
        for s_idx in range(len(corr_dict['s'])):
            # Loop through each s step
            
            # Start file sum timer
            file_start_time = time.perf_counter()
            
            if ftype == 'npy':
                # If the user saved grids as npy files
                
                # Reaad in the W_1 and B_1 convolved grids
                W1 = np.load(tempDir+'W_{}_0_0_RE_p{}_of_{}.npy'.format(
                    s_idx,box_idx+1,total_boxes))
                B1 = (nD/nR)*np.load(
                    tempDirRand+'B_{}_0_0_RE_p{}_of_{}.npy'.format(
                        s_idx,box_idx+1,total_boxes))
                
            elif ftype == 'fits':
                # If the user saved grids as fits files
                
                # Reaad in the W_1 and B_1 convolved grids
                W1 = u.fits_to_grid_unwrapper(
                    tempDir+'W_{}_0_0_RE_p{}_of_{}.fits'.format(
                        s_idx,box_idx+1,total_boxes))
                B1 = (nD/nR)*u.fits_to_grid_unwrapper(
                    tempDirRand+'B_{}_0_0_RE_p{}_of_{}.fits'.format(
                        s_idx,box_idx+1,total_boxes))
            
            # End file sum timer and update
            file_end_time = time.perf_counter()
            timing['T_FILESM'] += file_end_time - file_start_time  

            for n_idx in range(len(n_step)):
                # Loop over all n_steps

                # Start summation timer
                sum_start_time = time.perf_counter()
                
                # Update the corr dict
                corr_dict['W_n{}'.format(n_step[n_idx])][s_idx] += np.sum(
                    W_*W1**(n_step[n_idx]-1))
                corr_dict['B_n{}'.format(n_step[n_idx])][s_idx] += np.sum(
                    B_*B1**(n_step[n_idx]-1))
                
                # End summation timer and update
                sum_end_time = time.perf_counter()
                timing['T_SUM'] += sum_end_time - sum_start_time

        # PS verb to mark the end of a partitioned region
        if verbose:
            print('Finished partition {} of {}'.format(box_idx+1,total_boxes))
            
    for n_idx in range(len(n_step)):
        # Loop through each ell step
        
        # Compute the correlation functions
        corr_dict['xi_n{}'.format(n_step[n_idx])]=corr_dict['W_n{}'.format(
            n_step[n_idx])]/corr_dict['B_n{}'.format(n_step[n_idx])]

    # PS verb for file writing
    if verbose:
        print('Writing diagonal npcf to file...')
        
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
    # Keep file writing/reading time split between convolution and summation
    corr_table.header.set('T_DIVP',timing['T_DIVP'])
    corr_table.header.set('T_MAP',timing['T_MAP'])
    corr_table.header.set('T_CONV',timing['T_CONV'])
    corr_table.header.set('T_KERN',timing['T_KERN'])
    corr_table.header.set('T_FILE',timing['T_FILE'])
    corr_table.header.set('T_FILESM',timing['T_FILESM'])
    corr_table.header.set('T_SUM',timing['T_SUM'])
    
    # Write the corr pkg to file
    # Overwrite is fine here because the routine is fast
    corr_table.writeto('./correlations/'+data_file.split('.fits')[0]+
                       '_'+cfg_file.split('.txt')[0]+
                       '/corr_pkg_diag_npcf.fits',overwrite=True)
    
    if plot_correlation == False:
        # If the user does not wish to plot results
        
        # PS end and timing breakdown
        print('ConKer Diag. npcf Summation Step took {} s CPU time'.format(
            timing['T_FILESM']+timing['T_SUM']))
        if verbose:
            print('   Summation time = {} CPU s'.format(timing['T_SUM']))
            print('   File reading time '+
                  '= {} CPU s'.format(timing['T_FILESM']))
        print('\n'+'================================='+
              '=================================='+'\n')
        return


