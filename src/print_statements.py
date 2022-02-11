# =========================================================================== #
"""
Created on Mon Feb  7 10:22:56 2022
@authors: Zachery Brown

Description: These routines make up the print statements used by the ConKer
algorithm. There are several types including failures, warnings, required, and
verbose.
"""
# =========================================================================== #

# numpy required
import numpy as np

# ======================= REQUIRED MESSAGES ================================= #


def req_corr_initialize(data_file: str):
    # The initial print statmeent
    print('\n'+'============================= '+'SET UP'+
          ' =============================='+'\n')
    print('Data file: '+data_file)
    return


def req_count_obj(n_obj: int):
    # The number of objects in the requested catalog
    print('This catalog contains {} objects'.format(n_obj))
    return


def req_catalog_type(data_type: str):
    # Print the catalog type
    print('Catalog type is '+data_type)
    return


def req_rand_file(rand_file: str):
    # Print the randoms file to be used
    print('Randoms file: '+rand_file)
    return


def req_cfg_file(cfg_file: str):
    # Print the cfg file to be used
    print('Configuration: '+cfg_file)
    return


def req_corr_init_done():
    # Mark that the correlation initialization has succeeded
    print('Correlation routines ready. Proceed...')
    print('\n'+'==================================='+
      '================================'+'\n')
    return


def req_div_begin(div_file: str):
    # Statement to mark the start of the divide step
    print('\n'+'=========================== '+'DIVIDE STEP'+
          ' ==========================='+'\n')
    print('Creating a divide plan for '+div_file)
    return


def req_partition_num(total_boxes: int):
    # Marks the total number of partitions
    print('Created {} total angular partitions'.format(total_boxes))
    return


def req_div_time_end(div_t: float):
    # Notes the overall timing of the divide step
    print('Divide Step took {} s CPU time'.format(div_t))
    print('\n'+'==================================='+
          '================================'+'\n')
    return


def req_begin_conv(data_file: str):
    # Statement to mark the start of any convolution routine
    print('\n'+'===================== '+'CONKER CONVOLUTION STEP'+
          ' ====================='+'\n')
    print('Convolving kernels with catalog '+data_file)
    return


def req_part_done(istep: int, nsteps: int):
    # PS to mark the end of a radial step
    print('Finished partition {} of {}'.format(istep+1,nsteps))
    return


def req_mu_bin_readout(mu_edges: np.array):
    # Readout to show selected mode
    # Displays the mu binning parameters
    print('2pcf mu wedge mode selected')
    print('Using {} mu wedges from {} to {}'.format(
        len(mu_edges)-1,mu_edges.min(),mu_edges.max()))
    return


def req_conv_tot_time(tot_time: float):
    # Returns the total time of the convolution step
    print('ConKer Convolution Step took'+
          ' {} s CPU time'.format(tot_time))
    return


def req_bar():
    # Calls the ending bar with nothing else
    print('\n'+'================================'+
          '==================================='+'\n')
    return


def req_begin_wrap(data_type: str, data_file: str):
    # Statement to mark the start of any wrapper sum/average routine
    # Type dependence
    if data_type == 'galaxy':
        print('\n'+'=================== '+'CONKER 2PCF SUMMATION STEP'+
              ' ===================='+'\n')
        print('Summing over grids for '+data_file)
    elif data_type == 'lyman_alpha_1D':
        print('\n'+'==================== '+'CONKER 2PCF AVERAGE STEP'+
              ' ====================='+'\n')
        print('Averaging over grids for '+data_file)
    return


def req_wrap_timer(data_type: str, tot_time: float):
    # Marks the total time of the summation/averaging step
    if data_type == 'galaxy':
        print('ConKer 2pcf Summation Step took {} s CPU time'.format(
            tot_time))
    elif data_type == 'lyman_alpha_1D':
        print('ConKer 2pcf Averaging Step took {} s CPU time'.format(
            tot_time))
    return


# ======================== VERBOSE MESSAGES ================================= #


def verb_cosmo_readout(cosmo: tuple):
    # Print the chosen cosmological paramters
    print('  H0 = {} km/s/Mpc'.format(cosmo[1]))
    print('  Om_m = {}'.format(cosmo[2][0]))
    print('  Om_K = {}'.format(cosmo[2][1]))
    print('  Om_L = {}'.format(cosmo[2][2]))
    return


def verb_binning_readout(s_bc: np.array, s_be: np.array, g_s: float):
    # Print the radial binning settings
    print('Using kernels of width: {} Mpc (or Mpc/h)'.format(
        np.round(g_s,decimals=2)))
    print('{} radial bins will probe correlations'.format(len(s_bc)))
    print('  from {} to {} Mpc (or Mpc/h)'.format(
        np.round(s_be.min(),decimals=2),np.round(s_be.max(),decimals=2)))
    return


def verb_divide_go():
    # Partition angles set
    print('Partitioning parameters ready')
    return


def verb_div_save():
    # If the plan will be saved
    print('Preparing the partition plan file')
    return


def verb_div_plot():
    # If the sky plot will be made
    print('Plotting angular regions now')
    return


def verb_div_plot_save():
    # If the sky plot will be saved
    print('Saving sky plot in ./divide/figures/')
    return


def verb_kernel_begin():
    # Mark the start of the kernel creation
    print('Creating and saving kernels')
    return


def verb_kernel_end():
    # Mark the end of the kernel creation
    print('Kernel files written successfully')
    return


def verb_kernel_mu_wedge_progress(iwedge: int, totwedge: int):
    # Prints the completed mu wedge kernel set
    print('Finished mu slice {} of {}'.format(iwedge+1,totwedge))
    return


def verb_LOS_readout(LOS_ra_readout: np.array, LOS: tuple,
                     map_box_lims: tuple, conv_box_lims: tuple):
    # Statment about the original partition in the series convolution mode
    print('Initial LOS at (RA: '+
          str(np.round(LOS_ra_readout[0],decimals=2))+
          ', DEC: '+str(np.round(LOS[1],decimals=2))+')')
    print('Initial mapping box is '+
          str(np.round(map_box_lims[0][1]-
                       map_box_lims[0][0],decimals=2))+
          ' deg. by '+str(np.round(map_box_lims[1][1]-
                                   map_box_lims[1][0],
                                   decimals=2))+' deg.')
    print('Initial convolution box is '+
          str(np.round(conv_box_lims[0][1]-
                       conv_box_lims[0][0],decimals=2))+
          ' deg. by '+str(np.round(conv_box_lims[1][1]-
                                   conv_box_lims[1][0],
                                   decimals=2))+' deg.')
    return


def verb_coord_transfer():
    # Marks completion of the coordinate transformation
    print('Successful initial coordinate transformation')
    return


def verb_hist_succ():
    # Marks completion of the tracer mapping step
    print('Tracer histograms complete in initial partition')
    return


def verb_grid_file_succ():
    # PS to mark that initial grid files have been saved
    print('Wrote initial region grid files')
    return


def verb_radial_progress(curr_rad: float):
    # Marks the end of a radial step
    # Useful for the 0th partition
    print('Finished writing files'+
          ' for radial step s1 = {} Mpc (or Mpc/h)'.format(
              np.round(curr_rad,decimals=2)))
    return


def verb_conv_time_breakdown(t_kern: float, t_map: float, t_conv: float,
                             t_file: float):
    # Prints a detailed breakdown of the times for convolution
    print('   Kernel creation time = {} CPU s'.format(t_kern))
    print('   Mapping time = {} CPU s'.format(t_map))
    print('   Convolution time = {} CPU s'.format(t_conv))
    print('   File writing time = {} CPU s'.format(t_file))
    return


def verb_file_write():
    # Marks that the 2pcf will be written to file
    print('Writing 2pcf to file')
    return


def verb_wrap_timing_breakdown(data_type: str, t_wrap: float,
                               t_wrap_file: float):
    # Prints a detailed breakdown of the times for wrapping 
    if data_type == 'galaxy':
        print('   Summation time = {} CPU s'.format(t_wrap))
    elif data_type == 'lyman_alpha_1D':
        print('   Averaging time = {} CPU s'.format(t_wrap))
    print('   File reading time '+
          '= {} CPU s'.format(t_wrap_file))
    return


# ======================= FAILURE MESSAGES ================================== #


def fail_unsupported_data_type(req_type: str, supported_types: list):
    # A failure print statement to trip if user has asked for a data type 
    #   not currently available
    # Repeats their requested type followed by a list of supported ones
    print('\n'+'==!==!==!==!==!==!==!==!==!== '+'FAILURE'+
          ' ==!==!==!==!==!==!==!==!==!=='+'\n')
    print('Your requested data type: "'+req_type+'"'+
          ' is not currently supported')
    print('Available data types are:')
    for stype in supported_types:
        print('  '+stype)
    print('Please try again with one of these')
    print('\n'+'!==!==!==!==!==!==!==!==!==!==!==!=='+
          '!==!==!==!==!==!==!==!==!==!==!'+'\n')
    return


def fail_no_file(missing_file: str):
    # A failure statement if the requested file is not in ./data/
    print('\n'+'==!==!==!==!==!==!==!==!==!== '+'FAILURE'+
          ' ==!==!==!==!==!==!==!==!==!=='+'\n')
    print('The file "{}" is not in ./data/'.format(missing_file))
    print('Please try again with an existing fits catalog')
    print('\n'+'!==!==!==!==!==!==!==!==!==!==!==!=='+
          '!==!==!==!==!==!==!==!==!==!==!'+'\n')
    return


def fail_no_randoms(data_type: str):
    # If the data type requires randoms, but none have been provided
    print('\n'+'==!==!==!==!==!==!==!==!==!== '+'FAILURE'+
          ' ==!==!==!==!==!==!==!==!==!=='+'\n')
    print('The data type you requested: "{}"'.format(data_type))
    print('  requires a randoms catalog')
    print('Please specify one corresponding to your data file')
    
    print('\n'+'!==!==!==!==!==!==!==!==!==!==!==!=='+
          '!==!==!==!==!==!==!==!==!==!==!'+'\n')
    return


def fail_no_cfg(cfg_file: str):
    # A failure statement if the requested cfg set is not in ./params/
    print('\n'+'==!==!==!==!==!==!==!==!==!== '+'FAILURE'+
          ' ==!==!==!==!==!==!==!==!==!=='+'\n')
    print('The file "{}" is not in ./params/'.format(cfg_file))
    print('Please try again with an existing configuration file')
    print('\n'+'!==!==!==!==!==!==!==!==!==!==!==!=='+
          '!==!==!==!==!==!==!==!==!==!==!'+'\n')
    return


def fail_plot_div():
    # Trip a failure message if the user wishes to save a plot but did
    #   not create it by specifying plot_result = True
    print('\n'+'==!==!==!==!==!==!==!==!==!== '+'FAILURE'+
          ' ==!==!==!==!==!==!==!==!==!=='+'\n')
    print('You must run the plotting routine to save the sky plot')
    print('Retry divide() with plot_results = True')
    print('\n'+'!==!==!==!==!==!==!==!==!==!==!==!=='+
          '!==!==!==!==!==!==!==!==!==!==!'+'\n')
    return


def fail_unsafe_partition(theta_p: float):
    # Message prints if partition limits are deemed too large
    print('\n'+'==!==!==!==!==!==!==!==!==!== '+'FAILURE'+
          ' ==!==!==!==!==!==!==!==!==!=='+'\n')
    print('Your partition edge LOS error is currently {}%'.format(
        np.round(100*(np.radians(theta_p)**2/2),decimals=2)))
    print('This is too large to safely partition!')
    print('Try again with smaller s-bins or increase your '+
          'minimum redshift cut')
    print('Reduce theta_p if you set it yourself')
    print('\n'+'!==!==!==!==!==!==!==!==!==!==!==!=='+
          '!==!==!==!==!==!==!==!==!==!==!'+'\n')
    return


def fail_no_div_plan():
    # Trip this failure if there is no divide plan at the begining of
    #   the convolution step
    print('\n'+'==!==!==!==!==!==!==!==!==!== '+'FAILURE'+
          ' ==!==!==!==!==!==!==!==!==!=='+'\n')
    print('ConKer cannot find the partitioning scheme!')
    print("Make sure you've run divide() "+
          "with save_plan = True")
    print('Requires a partition plan corresponding to both tracers '+
          'and cfg files')
    print('\n'+'!==!==!==!==!==!==!==!==!==!==!==!=='+
          '!==!==!==!==!==!==!==!==!==!==!'+'\n')
    return


def fail_conv_dir_overwrite():
    # A failure to prevent overwriting large files resulting from the 
    #   convolution step
    print('\n'+'==!==!==!==!==!==!==!==!==!== '+'FAILURE'+
          ' ==!==!==!==!==!==!==!==!==!=='+'\n')
    print('You already have a temporary directory for this ')
    print('   catalog, cfg, and mode!')
    print("Check ./conv/ to make sure you don't have ")
    print("   existing data")
    print('\n'+'!==!==!==!==!==!==!==!==!==!==!==!=='+
          '!==!==!==!==!==!==!==!==!==!==!'+'\n')
    return


def fail_plot_wrap():
    # Fails if the plot wasn't made but the user tried to save it
    print('\n'+'==!==!==!==!==!==!==!==!==!== '+'FAILURE'+
          ' ==!==!==!==!==!==!==!==!==!=='+'\n')
    print('You must run the plotting routine '+
          'to save the correlation plot(s)')
    print('Retry the wrapping step with plot_correlation = True')
    print('\n'+'!==!==!==!==!==!==!==!==!==!==!==!=='+
          '!==!==!==!==!==!==!==!==!==!==!'+'\n')
    return


def fail_no_files_wrap():
    # A failure if the convolution wasn't run properly or files are missing
    print('\n'+'==!==!==!==!==!==!==!==!==!== '+'FAILURE'+
          ' ==!==!==!==!==!==!==!==!==!=='+'\n')
    print('ConKer cannot find your convolution files')
    print('Run the convolution step for these tracers and cfg')
    print('\n'+'!==!==!==!==!==!==!==!==!==!==!==!=='+
          '!==!==!==!==!==!==!==!==!==!==!'+'\n')
    return


