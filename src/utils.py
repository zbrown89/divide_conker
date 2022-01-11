# =========================================================================== #
"""
Created on Wed Dec 15 12:23:40 2021
@authors: Zachery Brown, Gebri Mishtaku

Description: This file, utils.py, contains useful functions and routines
for the Divide+ConKer algorithm. Some of the following have been repurposed
from the original ConKer algorithm (see https://github.com/mishtak00/conker).
In general, functions specify required data types for their arguments. See 
comments for more details.
"""
# =========================================================================== #

# Required scipy functions
from scipy import integrate
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.special import sph_harm

# numpy required
import numpy as np

# astropy fits required for file reading/writing
from astropy.io import fits

# Wigner 3j needed for higher order edge correction
from sympy.physics.wigner import wigner_3j

# ======================== UTIL FUNCTIONS =================================== #


def z2r(z: float, cosmology: tuple):
    # Convert a single redshift to a radial distance given a flat cosmology
    # Uses numerical integration and returns distances in units of Mpc or 
    #   Mpc/h if H0=100 km/s/Mpc
    # TODO -> Non-flat cosmology options
    c_, H0_, (Omega_M, Omega_K, Omega_L) = cosmology
    # type: float
    return (c_/H0_)*integrate.quad(lambda z_:(
        Omega_M*(1+z_)**3 + Omega_K*(1+z_)**2 + Omega_L)**-0.5,0, z)[0]


def interpolate_r_z(redshift_min: float, redshift_max: float,
                    cosmology: tuple):
    # Creates tick values for the given range of redshifts
    # Creates lookup tables with interpolated radii values (k=1 spline)
    # WARNING -> Look up table currently hard-coded to 300 steps
    redshift_ticks = np.linspace(redshift_min, redshift_max, 300)
    radii_ticks = np.array([z2r(z, cosmology) for z in redshift_ticks])
    LUT_radii = InterpolatedUnivariateSpline(
        redshift_ticks, radii_ticks,k=1)
    LUT_redshifts = InterpolatedUnivariateSpline(
        radii_ticks, redshift_ticks,k=1)
    # type: InterpolatedUnivariateSpline
    # type: InterpolatedUnivariateSpline
    return LUT_radii, LUT_redshifts
  

def edge2cen(edges: np.array):
    # A simple routine to return the centers of an array of bin edges
    # Averages over each pair of bin edges
    # type: np.array
    return (edges[1:]+edges[:-1])/2


def cen2edge(centers: np.array):
    # A simple routine to return the edges of an array of bin centers
    # Finds the bin width, or step size
    # Writes an array of edges plus and minus step_size/2 from each center
    step_size = centers[1]-centers[0]
    # type: np.array
    return np.asarray([centers[0]+bin_idx*step_size-step_size/2 
                       for bin_idx in range(len(centers)+1)])


def shift_RA(ra: np.array):
    # A function designed to move SGC RA values to the other side of the sky
    # Will work for SDSS CMASS SGC and DESI LRG/ELG SGC footprints
    # RA values must fall within [0,360), no negative RA values!
    # TODO -> Make this work for arbitrary data fooprints
    shiftRA = np.zeros(len(ra))
    condRight = (ra<=180.)
    condLeft = (ra>180.)
    shiftRA[condRight] = ra[condRight] + 180.
    shiftRA[condLeft] = ra[condLeft] - 180.
    # type: np.array
    return shiftRA


def makeBinEdges(lims: tuple, bin_size: float):
    # A function to make bin edges given the angular min and max, 
    #   plus the desired bin size
    # Will begin at the lower edge and extend bin_size-(span%size)
    #   beyond the max
    # Returns the edges, to be passed through edge2cen() for the centers
    span = lims[1]-lims[0]
    num_edges = int((span//bin_size))+1
    edges = np.asarray([lims[0]+bin_size*bin_idx 
                        for bin_idx in range(num_edges+1)])
    # type: np.array
    return edges


def cosDeg(angle: float):
    # A function to compute the cosine of an angle in degrees
    # type: float
    return np.cos(np.radians(angle))


def sinDeg(angle: float):
    # A function to compute the sine of an angle in degrees
    # type: float
    return np.sin(np.radians(angle))
  

def makeAlphaBins(dec_lims: tuple, cos_dec: float,
                  randoms: fits, theta_P: float):
    # A function to make the bins in RA given the value of theta_P and
    #   the DEC limits of the row
    # Adjusts the RA bin width by cos(DEC)
    # Returns the edges
    condition = (randoms['dec']>=dec_lims[0])&(randoms['dec']<dec_lims[1])
    ra_lims_row = ((randoms['ra'][condition]).min()-1.,
                   (randoms['ra'][condition]).max()+1.)
    ra_bin_edges_row = makeBinEdges(ra_lims_row,(2*theta_P)/cos_dec)
    # type: np.array
    return ra_bin_edges_row


def reduceSample(sample_col: np.array, reduction_fac: float):
    # A function to reduce a catalog for making a scatter plot
    # Returns random indices with no conditions
    # Reduction factor may be increased or reduced
    rng = np.random.default_rng()
    indices = rng.choice(len(sample_col),
                         size=int(len(sample_col)/reduction_fac),
                         replace=False)
    # type: np.array
    return indices


def ticksAndOrg(shift: bool):
    # A function to return the proper tick values for the Mollweide plot
    # If the RA values have been shifted, this will account for it
    #   when plotting ticks
    # org: hard coded to 180 for now
    # Feel free to change at any time, use caution, boxes may wrap around
    # Returns tick labels as strings with degrees symbol attatched
    # org works best when defined to be an integer RA value
    # TODO -> Test org behavior for plotting
    org = 180
    if shift == False:
        tick_label_values = np.array(
            [150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210])
        tick_label_values = np.remainder(tick_label_values+360+org,360)
    elif shift == True:
        tick_label_values = np.array(
            [330, 300, 270, 240, 210, 180, 150, 120, 90, 60, 30])
        tick_label_values = np.remainder(tick_label_values+360+org,360)
    tick_labels = []
    for tick in tick_label_values:
        tick_labels.append(str(tick).split('.')[0]+r'$^\circ$')
    # type: list
    # type: int
    return tick_labels, org
  
  
  def shiftForMoll(ra: np.array, dec: np.array, org: float):
    # A function to shift RA and DEC values of an array
    # Used for making a Mollwiede plot
    # Shift RA values
    # Scale conversion to [-180, 180]
    # Reverse the scale: east to the left
    x = np.remainder(ra+360-org,360)
    ind = x>180
    x[ind] -=360
    x=-x
    # type: np.array
    # type: np.array
    return np.radians(x), np.radians(dec)


def shiftForMollPoint(ra: float, dec: float, org: float):
    # A function to shift RA and DEC values of one point
    # Used for making a Mollwiede plot
    # Shift RA values
    # Scale conversion to [-180, 180]
    # Reverse the scale: east to the left
    x = np.remainder(ra+360-org,360)
    if x>180:
        x-=360
    x=-x
    # type: float
    # type: float
    return np.radians(x), np.radians(dec)


def plotBox(limits: tuple):
    # A function to return arrays suitable for plotting boxes
    # Creates a "box" based on angular limits tuple
    x_vals = np.asarray([limits[0][0],limits[0][1],
                         limits[0][1],limits[0][0],limits[0][0]])
    y_vals = np.asarray([limits[1][0],limits[1][0],
                         limits[1][1],limits[1][1],limits[1][0]])
    # type: np.array
    # type: np.array
    return x_vals,y_vals
  
  
def getLOSbox(rand_file: str, cfg_file: str, box_idx: int):
    # A function to return the LOS box limits 
    # Reads in the divide plan and unwraps the .fits file
    # For use in series LOS boxes
    # Returns useful info about the divide process
    div_plan = fits.open('./divide/plans/'+rand_file.split('.fits')[0]+
                         '_'+cfg_file.split('.txt')[0]+'.fits')[1]
    LOS = (div_plan.data['LOS_ra'][box_idx],
           div_plan.data['LOS_dec'][box_idx])
    conv_box_lims = ((div_plan.data['cb_ra_min'][box_idx],
                      div_plan.data['cb_ra_max'][box_idx]),
                     (div_plan.data['cb_dec_min'][box_idx],
                      div_plan.data['cb_dec_max'][box_idx]))
    map_box_lims = ((div_plan.data['mb_ra_min'][box_idx],
                     div_plan.data['mb_ra_max'][box_idx]),
                    (div_plan.data['mb_dec_min'][box_idx],
                     div_plan.data['mb_dec_max'][box_idx]))
    total_boxes = div_plan.header['BCTOT']
    shift_RA = div_plan.header['SHIFT_RA']
    # type: tuple
    # type: tuple
    # type: tuple
    # type: int
    # type: bool
    return LOS, conv_box_lims, map_box_lims, total_boxes, shift_RA 


def coordBoxSlice(catalog: fits, box_lims: tuple):
    # A function to slice a catalog into angular boxes
    # Returns a fits catalog corresponding to the box lim tuple
    output = catalog
    cond = (
        output['ra']>=box_lims[0][0])&(
        output['ra']<box_lims[0][1])&(
        output['dec']>=box_lims[1][0])&(
        output['dec']<box_lims[1][1])
    # type: fits
    return output[cond]
  
  
def sky2localCart(angular_coords: tuple, LOS: tuple):
    # Transforms the coordinates from sky to local cartesian
    # Defined with respect to a line of sight (LOS)
    # x-axis == LOS !
    # WARNING -> Expects the [2] column of angular_coordinates to be 
    #   expressed as a radial distance from Earth!
    nx_p = cosDeg(angular_coords[1])*cosDeg(angular_coords[0]-LOS[0])
    ny_p = cosDeg(angular_coords[1])*sinDeg(angular_coords[0]-LOS[0])
    nz_p = sinDeg(angular_coords[1])
    nx_pp = nx_p*cosDeg(LOS[1])+nz_p*sinDeg(LOS[1])
    ny_pp = ny_p
    nz_pp = -nx_p*sinDeg(LOS[1])+nz_p*cosDeg(LOS[1])
    ret_stack = (angular_coords[2]*nx_pp,
                 angular_coords[2]*ny_pp,
                 angular_coords[2]*nz_pp)
    # type: np.array
    # type: np.array
    # type: np.array
    return ret_stack[0], ret_stack[1], ret_stack[2]


def phi_theta_leg(x_v: np.array, y_v: np.array, z_v: np.array):
    # A function to return the theta and phi of a kernel cell
    # Used to populate legendre and ylm kernels
    # Return phi is shifted by \pi to use the scipy ylm function
    phi_v = np.arctan2(y_v, x_v)
    th2_v = np.arccos(z_v/np.sqrt(x_v**2+y_v**2+z_v**2))
    theta_v = th2_v
    # type: np.array
    # type: np.array
    return phi_v+np.pi, theta_v
  
  
def make_calibrated_kernel(radius: float, grid_spacing: float):
    # A function to make a calibrated kernel
    # Cells within \sqrt(3)/2 of the radial bin center are tagged
    # For each tagged cell, it places 1000 randoms points
    # The cell is given the value of the number of points 
    #   between the radius \pm g_S/2
    # TODO -> Investigate behavior of this new kernel as a function of N
    N_per_cell = 1000
    kr_idx_units = int(radius//grid_spacing)+1
    kg_size_idx_units = 2*kr_idx_units+1
    kernel_grid = np.zeros((kg_size_idx_units,
                            kg_size_idx_units,
                            kg_size_idx_units))
    center_idx_units = (kg_size_idx_units//2,
                        kg_size_idx_units//2,
                        kg_size_idx_units//2)
    potential_nonzero_cells = []
    for i in range(kg_size_idx_units):
        for j in range(kg_size_idx_units):
            for k in range(kg_size_idx_units):
                cell_rad_idx_units = np.sqrt(
                    (i-center_idx_units[0])**2+
                    (j-center_idx_units[1])**2+
                    (k-center_idx_units[2])**2)
                if (cell_rad_idx_units>=(radius/grid_spacing)-np.sqrt(3)/2.)&(
                        cell_rad_idx_units <= (
                            radius/grid_spacing)+np.sqrt(3)/2.): 
                    potential_nonzero_cells.append((i,j,k))
    for cell_id in potential_nonzero_cells:
        cal_rad_idx_units = np.sqrt((
            np.random.uniform(cell_id[0]-0.5,cell_id[0]+0.5,N_per_cell)-
            center_idx_units[0])**2+(
                np.random.uniform(cell_id[1]-0.5,cell_id[1]+0.5,N_per_cell)-
                center_idx_units[1])**2+(
                    np.random.uniform(cell_id[2]-0.5,cell_id[2]+0.5,
                                      N_per_cell)-center_idx_units[2])**2)
        kernel_grid[cell_id] = len(np.where((cal_rad_idx_units>=(
            radius/grid_spacing-0.5))&(cal_rad_idx_units<(
                radius/grid_spacing+0.5)))[0])/N_per_cell
    # type: np.array
    return kernel_grid
  
  
def ylmKernel(R_: float, g_S: float, n_: int, m_: int):
    # A function to make the ylm kernels
    # Finds all nonzero cells in the original kernel
    # Multiplies the value by y_\ell^m(phi,theta)
    # Theta is the polar angle! Phi is the azimuthal angle!
    # Due to the LOS orientation, phi and theta are computed with the 
    #   phi_theta_leg() function of the Y,Z,X cyclic coord permutation
    # Returns two kernels, one with the real, one with the imaginary part
    k_G = make_calibrated_kernel(R_,g_S)
    kernel_shape = k_G.shape
    kX,kY,kZ = np.array([(
        kernel_shape[0]//2-idx)*g_S 
        for idx in range(kernel_shape[0])]),np.array([(
                kernel_shape[0]//2-idx)*g_S 
                for idx in range(kernel_shape[1])]),np.array([(
                        kernel_shape[0]//2-idx)*g_S 
                        for idx in range(kernel_shape[2])])
    ylm_kernel_RE = np.zeros(kernel_shape)
    ylm_kernel_IM = np.zeros(kernel_shape)
    for i in range(kernel_shape[0]):
        for j in range(kernel_shape[1]):
            for k in range(kernel_shape[2]):
                if k_G[i,j,k] != 0:
                    azimuth_cell,polar_cell = phi_theta_leg(
                        kY[j],kZ[k],kX[i])
                    kvalue = sph_harm(m_, n_, azimuth_cell, polar_cell)
                    ylm_kernel_RE[i,j,k] = np.real(kvalue) * k_G[i,j,k]
                    ylm_kernel_IM[i,j,k] = np.imag(kvalue) * k_G[i,j,k]
    # type: np.array
    # type: np.array
    return ylm_kernel_RE, ylm_kernel_IM


def grid_to_fits_wrapper(grid: np.array, mask: np.array, save_path: str):
    # A function to save a sparse grid using astropy FITS tables
    # Finds the locations and values of non-masked cells
    nz_cell_X,nz_cell_Y,nz_cell_Z = np.where(mask!=0.)
    nz_cell_val = np.zeros(len(nz_cell_X))
    for c_id in range(len(nz_cell_X)):
        nz_cell_val[c_id] = grid[
            nz_cell_X[c_id],nz_cell_Y[c_id],nz_cell_Z[c_id]]
    grid_table = fits.BinTableHDU.from_columns([
        fits.Column(name='grid_val',array=nz_cell_val,format='E')])
    grid_table.writeto(save_path,overwrite=True)
    # type: None
    return
  
  
def fits_to_grid_unwrapper(save_path: str):
    # A function to read a sparse table grid data using astropy FITS
    # Returns array populated with entries from table
    tab = fits.open(save_path)[1].data
    flattened_grid = tab['grid_val']
    # type: np.array
    return flattened_grid


def make_grid_mask(grid: np.array):
    # A function to make a mask for a 3D grid
    # Used to "sparsify matrices for saving"
    # Finds non-zero entries and populates a new grid with 1s there
    shape = grid.shape
    nz_cell_X,nz_cell_Y,nz_cell_Z = np.where(grid!=0.)
    mask = np.zeros(shape)
    for c_id in range(len(nz_cell_X)):
        mask[nz_cell_X[c_id],nz_cell_Y[c_id],nz_cell_Z[c_id]] = 1.
    # type: np.array
    return mask


def ylm_norm_m0(ell: int):
    # A function to return the normalization of yl0
    norm = np.sqrt( (2*ell+1)/(4*np.pi) )
    # type: float
    return norm


def squareRadialBins(bin_centers: np.array):
    # A function to square the radial bins for the 3pcf
    # Returns an array of bins for s1 and s2
    # Correponds only to unique pairs
    squarebinsS1 = []
    squarebinsS2 = []
    for bi in range(len(bin_centers)):
        for bj in range(len(bin_centers)):
            if bin_centers[bi] >= bin_centers[bj]:
                squarebinsS1.append(bin_centers[bi])
                squarebinsS2.append(bin_centers[bj])
    # type: np.array
    # type: np.array
    return np.asarray(squarebinsS1),np.asarray(squarebinsS2)
  
  
def edge_correct_3pcf(path: str):
    # A function to open the 3pcf files and perform the edge correction
    # Loop through all pairs
    # Create and invert matrix M_kl
    # Use R_ell factors computed during summation step
    # Then square matrices for plotting
    # Includes correlation function zeta and a scaling grid
    # Scaling grid is set to s1^2*s2^2/(100 Mpc/h)^4
    # See https://arxiv.org/abs/1506.02040 for details
    # Returns a dict with egde corrected 3pcf for each ell
    # Automatically grabs ell_max from file
    # path should specify a corr_pkg_3pcf.fits file
    file = fits.open(path)[1].data
    dist_dict = {}
    for cid in range(len(file.columns)):
        dist_dict[file.columns[cid].name] = file[file.columns[cid].name]
    ell_max = int(list(dist_dict.keys())[-1].split('f')[1])
    ell_step = np.linspace(0,ell_max,ell_max+1,dtype='int')
    lp_step = np.linspace(1,ell_max,ell_max,dtype='int')
    corr_dict = {}
    corr_dict['s1'] = dist_dict['s1']
    corr_dict['s2'] = dist_dict['s2']
    for ell_idx in range(len(ell_step)):
        corr_dict['zeta'+str(ell_step[ell_idx])] = np.zeros(
            len(corr_dict['s1']))
    for s_idx in range(len(dist_dict['s1'])):
        M_kl = np.zeros((len(ell_step),len(ell_step)))
        for k_idx in range(len(ell_step)):
            for l_idx in range(len(ell_step)):
                sum_lp = 0.
                norm_lp = (2*ell_step[k_idx]+1)
                for lp_idx in range(len(lp_step)):
                    sum_lp += np.float64(
                        wigner_3j(ell_step[l_idx],lp_step[lp_idx],
                                  ell_step[k_idx],0,0,0))**2*dist_dict[
                                      'f'+str(lp_step[lp_idx])][s_idx]
                M_kl[k_idx,l_idx] = sum_lp*norm_lp
        edge_corr_mat = np.linalg.inv(np.identity(len(ell_step))+M_kl)
        corr_vec = np.asarray([dist_dict['zeta'+str(ell_step[ell_idx])][s_idx] 
                               for ell_idx in range(len(ell_step))])
        fixed_corr_vec = np.matmul(edge_corr_mat, corr_vec)
        for ell_idx in range(len(ell_step)):
            if ell_idx ==0:
                corr_dict['zeta'+str(ell_step[ell_idx])][s_idx] = dist_dict[
                    'zeta'+str(ell_step[ell_idx])][s_idx]
            if ell_idx >0:
                corr_dict['zeta'+str(ell_step[ell_idx])][
                    s_idx] = fixed_corr_vec[ell_idx]
    stp = corr_dict['s1'][1] - corr_dict['s1'][0]
    s1_idx = np.asarray(
        (corr_dict['s1']-corr_dict['s1'].min())/stp,dtype='int')
    s2_idx = np.asarray(
        (corr_dict['s2']-corr_dict['s2'].min())/stp,dtype='int')
    out = {}
    out['s1'] = corr_dict['s1']
    out['s2'] = corr_dict['s2']
    out['sc'] = np.zeros((s2_idx.max()+1,s2_idx.max()+1))
    for ell_idx in range(ell_max+1):
        out['z'+str(ell_step[ell_idx])] = np.zeros(
            (s2_idx.max()+1,s2_idx.max()+1))
        for s_id in range(len(s1_idx)):
            if s1_idx[s_id] != s2_idx[s_id]:
                out['z'+str(ell_step[ell_idx])][s1_idx[s_id],s2_idx[
                    s_id]]=corr_dict['zeta'+str(ell_step[ell_idx])][s_id]
                out['z'+str(ell_step[ell_idx])][s2_idx[s_id],s1_idx[
                    s_id]]=corr_dict['zeta'+str(ell_step[ell_idx])][s_id]
                out['sc'][s1_idx[s_id],s2_idx[s_id]] = corr_dict['s1'][
                    s_id]**2*corr_dict['s2'][s_id]**2/100000000
                out['sc'][s2_idx[s_id],s1_idx[s_id]] = corr_dict['s1'][
                    s_id]**2*corr_dict['s2'][s_id]**2/100000000
    # type: dict
    return out


def CIC_grid(coords: np.array, grid_edges: list, grid_centers: 
             list, bin_width: float, weights: np.array):
    # A function to use CIC grid interpolation when painting the density field
    # If the user wishes to use this option, replace the np.histogram step in
    #   conker_series.py for all four grids, N, R, N_mp, and R_mp
    # TODO -> Incorporation into driver routine as optional parameters
    mesh_ijk = np.array((np.floor((coords.T[0] - grid_edges[0][0])/bin_width),
                    np.floor((coords.T[1] - grid_edges[1][0])/bin_width),
                    np.floor((coords.T[2] - grid_edges[2][0])/bin_width))).T.astype(dtype=int)
    mesh_ijk_plus_one = mesh_ijk + 1
    data_grid = np.zeros((len(grid_centers[0]),len(grid_centers[1]),len(grid_centers[2])))
    for g_id in range(len(coords)):
        dg_i,dg_j,dg_k = mesh_ijk[g_id][0],mesh_ijk[g_id][1],mesh_ijk[g_id][2]
        if (dg_i+1<data_grid.shape[0])&(dg_j+1<data_grid.shape[1])&(dg_k+1<data_grid.shape[2]):
            data_grid[dg_i,dg_j,dg_k] += (weights[g_id]/(bin_width**3))*(
                grid_centers[0][dg_i+1]-coords[g_id][0])*(
                grid_centers[1][dg_j+1]-coords[g_id][1])*(
                grid_centers[2][dg_k+1]-coords[g_id][2])
            data_grid[dg_i+1,dg_j,dg_k] += (weights[g_id]/(bin_width**3))*(
                coords[g_id][0]-grid_centers[0][dg_i])*(
                grid_centers[1][dg_j+1]-coords[g_id][1])*(
                grid_centers[2][dg_k+1]-coords[g_id][2])
            data_grid[dg_i,dg_j+1,dg_k] += (weights[g_id]/(bin_width**3))*(
                grid_centers[0][dg_i+1]-coords[g_id][0])*(
                coords[g_id][1]-grid_centers[1][dg_j])*(
                grid_centers[2][dg_k+1]-coords[g_id][2])
            data_grid[dg_i,dg_j,dg_k+1] += (weights[g_id]/(bin_width**3))*(
                grid_centers[0][dg_i+1]-coords[g_id][0])*(
                grid_centers[1][dg_j+1]-coords[g_id][1])*(
                coords[g_id][2]-grid_centers[2][dg_k])
            data_grid[dg_i,dg_j+1,dg_k+1] += (weights[g_id]/(bin_width**3))*(
                grid_centers[0][dg_i+1]-coords[g_id][0])*(
                coords[g_id][1]-grid_centers[1][dg_j])*(
                coords[g_id][2]-grid_centers[2][dg_k])
            data_grid[dg_i+1,dg_j,dg_k+1] += (weights[g_id]/(bin_width**3))*(
                coords[g_id][0]-grid_centers[0][dg_i])*(
                grid_centers[1][dg_j+1]-coords[g_id][1])*(
                coords[g_id][2]-grid_centers[2][dg_k])
            data_grid[dg_i+1,dg_j+1,dg_k] += (weights[g_id]/(bin_width**3))*(
                coords[g_id][0]-grid_centers[0][dg_i])*(
                coords[g_id][1]-grid_centers[1][dg_j])*(
                grid_centers[2][dg_k+1]-coords[g_id][2])
            data_grid[dg_i+1,dg_j+1,dg_k+1] += (weights[g_id]/(bin_width**3))*(
                coords[g_id][0]-grid_centers[0][dg_i])*(
                coords[g_id][1]-grid_centers[1][dg_j])*(
                coords[g_id][2]-grid_centers[2][dg_k])
    # type: np.array
    return data_grid
