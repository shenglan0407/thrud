import numpy as np

def q_grid_as_xyz(q_values, num_phi, k):
    """
    Generate a q-grid in cartesian space: (q_x, q_y, q_z).
    Parameters
    ----------
    q_values : ndarray/list, float
        The values of |q| to extract rings at (in Ang^{-1}).
    num_phi : int
        The number of equally spaced points around the azimuth to
        interpolate onto (e.g. `num_phi`=360 means 1 deg spacing).
        
    k : float
        The wavenumber of the indicent beam.
    Returns
    -------
    qxyz : ndarray, float
        An N x 3 array of (q_x, q_y, q_z)
    """
    
    q_values = np.array(q_values)

    phi_values = np.linspace( 0.0, 2.0*np.pi, num=num_phi )
    num_q = len(q_values)

    # q_h is the magnitude projection of the scattering vector on (x,y)
    q_z = - np.power(q_values, 2) / (2.0 * k)
    q_h = q_values * np.sqrt( 1.0 - np.power( q_values / (2.0 * k), 2 ) )

    # construct the polar grid:
    qxyz = np.zeros(( num_q * num_phi, 3 ))
    qxyz[:,0] = np.repeat(q_h, num_phi) * np.cos(np.tile(phi_values, num_q)) # q_x
    qxyz[:,1] = np.repeat(q_h, num_phi) * np.sin(np.tile(phi_values, num_q)) # q_y
    qxyz[:,2] = np.repeat(q_z, num_phi)                                      # q_z

    return qxyz

def get_constant_solid_angle_q_values(qmin,qmax,
    k, delta_omega, num_phi):
    """
    Generate a list a of q values that produces polar pixels
    that span a constant solid angle. 

    i.e. sin(2*theta) d(2*theta) d(phi) = delta_omega
    where q = 2k sin(theta) and phi evenly spans 0 to 2pi.

    ----------
    q_min : float
        minimum q value, inclusive.
    q_max: float
        maximum q value, exclusive in general
    k : float
        The wavenumber of the indicent beam.
    delta_omega: float
        solid angle of each polar pixel. 
    num_phi : int
        The number of equally spaced points around the azimuth to
        interpolate onto (e.g. `num_phi`=360 means 1 deg spacing).
        

    Returns
    -------
    q_mags : ndarray, float
        a list of q value magnitudes that produces polar pixels
        that span a constant solid angle. 
    """

    wavlen = 2 * np.pi / k
    bmin = np.arcsin( qmin * wavlen / (4*np.pi) ) *2 # 2 * theta, min
    bmax = np.arcsin( qmax * wavlen / (4*np.pi) ) *2 # 2 * theta, max

    phis = np.linspace(0,np.pi*2,180)
    dphi = phis[1]-phis[0]

    betas = [bmin] # 2 * theta
    d_betas = [] # d( 2 * theta)

    while True:
        d_beta = delta_omega/dphi/np.sin(betas[-1])
        beta_next = betas[-1] + \
        d_beta
        
        d_betas.append(d_beta)
        
        if beta_next >= bmax:
            break
        betas.append(beta_next)
        
    betas = np.array(betas)
    d_betas = np.array(d_betas)

    q_mags = np.sin( betas/2.0 ) *4*np.pi/wavlen

    return q_mags
