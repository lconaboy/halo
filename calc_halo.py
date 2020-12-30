from __future__ import print_function

import seren3
import numpy as np
# from seren3.array import SimArray
from seren3.core.snapshot import Family
from pymses.utils.regions import Sphere


def M_to_R(m, f, cosmo):
    """Converts from a halo mass defined as f wrt the critical density
    to the radius of the same definition.

    :param m: (float) the mass of the halo in the definition specified
              by f in units of Msol
    :param f: (str or float or int) the mass definition, either 'vir'
              or the overdensity times the critical density e.g. 200

    :param cosmo: (dict) dictionary of cosmology parameters compatible
        with cosmolopy

    :returns: radius of the halo in that mass definition, in comoving
              kpc/h

    :rtype: (float)
    """
    from cosmolopy.distance import e_z
    from cosmolopy.density import cosmo_densities, omega_M_z

    assert(f == 'vir' or (isinstance(f, int) or isinstance(f, float))), \
        "f should be 'vir' or number"

    # rho_crit(0) = 3 * H0**2 / 8 * \pi * G
    # H(z) = H0 * E(z)
    # rho_crit(z) = rho_crit(0) * E(z)**2
    
    rho_crit, _ = cosmo_densities(**cosmo)
    E_z = e_z(**cosmo)
    rho_crit_z = rho_crit * E_z**2
    omega_m_z = omega_M_z(**cosmo)
    
    if f == 'vir':
        X = 18 * np.pi**2 + (82. * (omega_m_z - 1.)) \
            - (39 * (omega_m_z - 1.)**2)
    else:
        X = float(f)

    r = ((3. * m) / (4. * np.pi * X * rho_crit_z)) ** (1. / 3.)  # Mpc
    r = r * 1e3 * cosmo['h'] / cosmo['aexp']  # a kpc h**-1
    
    return r


def get_pos(h):
    """Gets the halo position in code units (i.e. [0., 1.])

    :param h: (Halo) halo object
    :returns: halo position in code units
    :rtype: (array)

    """

    return h.pos.view(type=np.ndarray)


def get_r(R, h):
    """Converts a radius R from comoving kpc into code units 
    (i.e. [0., 1.])

    Test (for get_pos, too)

        pos_s, r_s = h.pos_r_code_units
        print('---- calc:', t_Rvir_rs * 1e-3 * subsnap.cosmo['h'] / 
               float(h.boxsize))
        print('---- me:', pos, r)
        print('---- rs:', pos_s, r_s)

    :param R: (float) radius in comoving kpc (i.e. a * kpc)
    :param h: (Halo) halo object
    :returns: halo radius in code units
    :rtype: (float)

    """
    
    boxsize = h.boxsize * 1e3 / h.base.cosmo['h']  # comoving kpc
    
    return float(R / boxsize)


def get_sphere(R, h):
    """Gives a spherical filter centred on the halo h, with a radius of
    R (in comoving kpc)

    :param R: (float) radius in comoving kpc (i.e. a * kpc)
    :param h: (Halo) halo object

    :returns: halo radius in code units

    :rtype: (float)
    """
    # Radius and position of haloes in code units
    r = get_r(R, h)
    pos = get_pos(h)

    return Sphere(pos, r)


def get_masses(snap, R, h):
    """Returns the dark matter, gas and stellar mass (in Msol) of the halo h
    within the radius R (in comoving kpc)

    :param snap: (Snapshot) snapshot object containing the relevant
        dark matter, gas and stellar fields
    :param R: (float) radius within which to calculate the mass
    :param h: (Halo) halo object

    :returns: dark matter, gas and stellar halo mass (in Msol)

    :rtype: (float, float, float)
    """
    # Get the spherical filter
    sphere = get_sphere(R, h)

    # Filter the Snapshot object and return a spherical sub-region
    sub = snap[sphere]

    # Use the Family class to access the relevant fields
    d = Family(sub, 'dm')
    g = Family(sub, 'amr')
    s = Family(sub, 'star')

    # Get the total mass in Msol (checking if there's anything in the
    # dset is handled in tot_mass)
    m_d = tot_mass(d["mass"].flatten())  # Msol
    m_g = tot_mass(g["mass"].flatten())  # Msol
    m_s = tot_mass(s["mass"].flatten())  # Msol

    # There might not be stars, so be a little more careful
    # s_ds = s[["age", "mass"]].flatten()
    # if (len(s_ds["age"]) > 0):
    #     m_s = tot_mass(s_ds)  # Msol
    # else:
    #     m_s = 0.0

    return [m_d, m_g, m_s]


def tot_mass(flattened_dset):
    """Returns the total mass in flattened_dset in a vanilla numpy
    array.

    :param flattened_dset: (source) must contain the 'mass' field and
        have had .flatten() applied

    :returns: total mass in Msol

    :rtype: (float or ndarray) 
    """
    # Check if there's anything in this dset, useful for cases when no
    # stars have formed
    if len(flattened_dset['mass'] > 0):
    
        # Get the total mass in the flattened dset in a vanilla numpy
        # array
        m = flattened_dset["mass"].in_units("Msol").sum().view(type=np.ndarray)
    
        # Sometimes if there is no mass, then this will return an array
        # containing one value, so check for this
        if type(m) == type(np.array(0.)):
            print('---- found no mass')
            m = float(m)
    else:
        m = 0.0
        
    return m

        
def main(path, ioutput):
    from mpi4py import MPI

    # MPI stuff
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    barrier = comm.Barrier
    finalize = MPI.Finalize

    # Convenience variables
    root = 0
    host = rank == root
    
    if host:
        print('---- working on output', ioutput)
        print(path)

    sim = seren3.init(path)
    snap = sim[ioutput]
    subsnap = snap

    snap.set_nproc(1)
    halos = subsnap.halos(finder='rockstar')

    nhalos = len(halos)

    halo_ix = None
    chunks = None
    
    if host:
        halo_ix = halos.halo_ix(shuffle=True)
        chunks = np.array_split(halo_ix, size)

    halo_ixs = comm.scatter(chunks, root)

    # Field names for HDF5 datasets
    fields = ['halo_id', 'mvir_rs', 'rvir_rs', 'm200c_rs', 'r200c_rs',
              'mvir_d', 'mvir_g', 'mvir_s', 'm200c_d', 'm200c_g', 'm200c_s']
    # Field units for HDF5 dataset attributes
    units = ['', 'Msol', 'kpc (comoving)', 'Msol', 'kpc (comoving)',
             'Msol', 'Msol', 'Msol', 'Msol', 'Msol', 'Msol']

    assert (len(fields) == len(units))
    nvar = len(fields)

    # Local buffers for holding data
    data = np.zeros(shape=(len(halo_ixs), nvar))

    # Here i is the loop counter and ih is the halo index
    i = 0
    for ih in halo_ixs:
        # Current halo object
        h = halos[ih]

        # Quick check to make sure we're looking at interesting haloes
        # (trust 1000 particle haloes for PM codes)
        npart = len(h.d)
        # print('-------- {0:d} halo particles'.format(npart))
        if npart < 999:
            continue

        # Rockstar halo ID
        data[i, 0] = h['id']

        # Virial quantities returned by rockstar
        rvir = h.properties['rvir']/subsnap.cosmo['h']   # a * kpc
        data[i, 1] = h.properties['mvir']/subsnap.cosmo['h']   # Msol
        data[i, 2] = rvir

        # Quantities at 200*\rho_crit returned by rockstar
        m200c = h.properties['m200c']/subsnap.cosmo['h']  # Msol
        data[i, 3] = m200c
        r200c = M_to_R(m200c, f=200, cosmo=subsnap.cosmo) \
            / subsnap.cosmo['h']  # a * kpc
        data[i, 4] = r200c
        # Masses at the virial radius recalculated from the data
        data[i, 5:8] = get_masses(snap, rvir, h)   # Msol

        # Masses at the 200*\rho_crit radius recalculated from the data
        data[i, 8:11] = get_masses(snap, r200c, h)  # Msol

        i += 1

    # Trim down the local data array
    data = data[0:i, :]
    
    # Gather all of the local array sizes
    # data_shape = data.shape
    # data_ravel = data.ravel()
    # data_unravel = data.reshape(data_shape)
    # assert np.all(data_unravel == data)
    # datac = np.array(comm.gather(data.shape[0] * data.shape[1], root))

    # Temporarily flatten the data array
    data = data.ravel()
    # Gather all of the local array sizes
    datac = np.array(comm.gather(len(data), root))
    
    if host:
        databuf = np.empty(shape=(sum(datac)))
    else:
        databuf = None

    # Use the vector Gather to collect all of the flattened local
    # arrays
    comm.Gatherv(sendbuf=data, recvbuf=(databuf, datac), root=root)
        
    if host:
        import os, h5py

        # Reshape the flattened array to its original shape
        ncols = nvar
        nrows = len(databuf)//ncols
        databuf = np.reshape(databuf, (nrows, ncols))
        fn = os.path.join(path, 'out_{0:d}.h5'.format(ioutput))

        print('-------- kept {0:d} haloes'.format(nrows))
        
        with h5py.File(fn, 'w') as f:
            # Store cosmological parameters
            g = f.create_group('cosmo')
            for key in snap.cosmo:
                g.create_dataset(key, data=snap.cosmo[key])

            # Store actual results
            g = f.create_group('results')
            for i in range(nvar):
                d = g.create_dataset(fields[i], data=databuf[:, i])
                d.attrs['units'] = units[i]

if __name__ == '__main__':
    path = '/lustre/scratch/astro/lc589/tot-halo-test'
    iout = 99

    main(path, iout)
