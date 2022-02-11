# =========================================================================== #
"""
Created on Mon Feb  7 11:29:47 2022
@authors: Zachery Brown, Gebri Mishtaku

Description: This file, utils.py, contains useful functions and routines
for the ConKer algorithm. Some of the following have been repurposed
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

# ======================== UTIL FUNCTIONS =================================== #


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


def z2r(z: float, cosmology: tuple):
    # Convert a single redshift to a radial distance given a flat cosmology
    # Uses numerical integration and returns distances in units of Mpc or 
    #   Mpc/h if H0=100 km/s/Mpc
    # TODO -> Non-flat cosmology options
    c_, H0_, (Omega_M, Omega_K, Omega_L) = cosmology
    # type: float
    return (c_/H0_)*integrate.quad(lambda z_:(
        Omega_M*(1+z_)**3 + Omega_K*(1+z_)**2 + Omega_L)**-0.5,0, z)[0]


def edge2cen(edges: np.array):
    # A simple routine to return the centers of an array of bin edges
    # Averages over each pair of bin edges
    # type: np.array
    return (edges[1:]+edges[:-1])/2


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


def mu_wedge_kernel(R_: float, g_S: float, mu_lims: tuple):
    # A function to make a kernel populated only for a given mu-wedge
    # Finds the non-zero cells in the original kernel
    # Populates only cells within a given range of mu values
    k_G, _ = ylmKernel(R_,g_S,n_=0,m_=0)
    kernel_shape = k_G.shape
    kX,kY,kZ = np.array([(
        kernel_shape[0]//2-idx)*g_S 
        for idx in range(kernel_shape[0])]),np.array([(
                kernel_shape[0]//2-idx)*g_S 
                for idx in range(kernel_shape[1])]),np.array([(
                        kernel_shape[0]//2-idx)*g_S 
                        for idx in range(kernel_shape[2])])
    mu_kernel = np.zeros(kernel_shape)
    for i in range(kernel_shape[0]):
        for j in range(kernel_shape[1]):
            for k in range(kernel_shape[2]):
                if k_G[i,j,k] != 0:
                    azimuth_cell,polar_cell = phi_theta_leg(
                        kY[j],kZ[k],kX[i])
                    if (np.abs(np.cos(polar_cell)) >= mu_lims[0])&(
                            np.abs(np.cos(polar_cell)) <= mu_lims[1]):
                        mu_kernel[i,j,k] = k_G[i,j,k]
    # type: np.array
    return mu_kernel


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


