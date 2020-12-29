import seren3
import numpy as np
from seren3.array import SimArray

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


    assert(f == 'vir' or (type(f) == type(1.) or
           type(f) == type(1))), "f should be 'vir' or number"

    # rho_crit(0) = 3 * H0**2 / 8 * \pi * G
    # H(z) = H0 * E(z)
    # rho_crit(z) = rho_crit(0) * E(z)**2
    
    # FIXME this doesn't work in this instance for some reason, e_z
    # throws an error
    # z is a kwarg in **cosmo

    print(cosmo)
    rho_crit, _ = cosmo_densities(**cosmo)
    E_z = e_z(**cosmo)
    rho_crit_z = rho_crit * E_z**2
    omega_m_z = omega_M_z(**cosmo)
    
    if f == 'vir':
        X = 18*np.pi**2 + 82.*(omega_m_z - 1.) - 39*(omega_m_z - 1.)**2
    else:
        X = float(f)

    r = ((3. * m)/(4. * np.pi * X * rho_crit_z)) ** (1./3.)  # Mpc
    r = r * 1e3 * cosmo['h'] / cosmo['aexp']  # a kpc h**-1
    
    return r


def tot_mass(flattened_dset):
    """Returns the total mass in flattened_dset in a vanilla numpy
    array.

    :param flattened_dset: (source) must contain the 'mass' field and
        have had .flatten() applied

    :returns: total mass in Msol

    :rtype: (float or ndarray) 
    """
    # Get the total mass in the flattened dset in a SimArray
    m = flattened_dset["mass"].in_units("Msol").sum().view(type=np.ndarray)
    
    # Sometimes if there is no mass, then this will return an array
    # containing one value, so check for this
    if type(m) == type(np.array(0.)):
        print('---- found no mass')
        m = float(m)
    
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
        halo_ix = halos.halo_ix(shuffle=True)[0:12]
        chunks = np.array_split(halo_ix, size)

    halo_ixs = comm.scatter(chunks, root)


    # hid = []
    # Mvir_rs = []
    # Rvir_rs = []
    # M200c_rs = []
    # R200c_rs = []
    # Mvir_dm = []
    # Mvir_gas = []
    # Mvir_star = []
    # M200c_dm = []
    # M200c_gas = []
    # M200c_star = []

    fields = ['halo_id', 'mvir_rs', 'rvir_rs', 'm200c_rs', 'r200c_rs',
              'mvir_dm', 'mvir_gas', 'mvir_star']
    units = ['', 'Msol', 'kpc (comoving)', 'Msol', 'kpc (comoving)',
             'Msol', 'Msol', 'Msol']
    nvar = 8

    assert (nvar == len(fields)) and (len(fields) == len(units))
    
    data = np.zeros(shape=(len(halo_ixs), nvar))
    
    # Here i is the loop counter and ih is the halo index
    for i, ih in enumerate(halo_ixs):
        h = halos[ih]

        # The quantities returned by rockstar
        t_Mvir_rs = h.properties['mvir']/subsnap.cosmo['h']  # Msol
        t_Rvir_rs = h.properties['rvir']/subsnap.cosmo['h']  # a * kpc

        # The quantities calculated at the virial radius calculated by rockstar
        t_Mvir_gas = tot_mass(h.g["mass"].flatten())  # Msol
        t_Mvir_dm = tot_mass(h.d["mass"].flatten())   # Msol
        s_dset = h.s[["age", "mass"]].flatten()
        if (len(s_dset["age"]) > 0):
            t_Mvir_star = tot_mass(s_dset)  # Msol
        else:
            t_Mvir_star = 0.0

        # The quantities calculated at R200c, as determined by the
        # M200c returned by rockstar
        t_M200c_rs = h.properties['m200c']/subsnap.cosmo['h']        # Msol
        t_R200c_rs = M_to_R(t_M200c_rs, f=200, cosmo=subsnap.cosmo) \
            / subsnap.cosmo['h']  # a * kpc

        t_hid = h['id']

        # sto.idx = h["id"]
        # # sto.result = {"M_tot" : M_tot, "R_vir" : R_vir.in_units('km'), "M_gas" : M_gas, \
        # #                   "M_star" : s_dset["mass"].in_units("Msol").sum()}

        # sto.result = {'Mvir_rs': Mvir_rs, 'Rvir_rs': Rvir_rs,
        #               'Mvir_dm': Mvir_dm, 'Mvir_gas': Mvir_gas, 
        #               'Mvir_star': Mvir_star}

        # hid.append(t_hid)
        # Mvir_rs.append(t_Mvir_rs)
        # Rvir_rs.append(t_Rvir_rs)
        # M200c_rs.append(t_M200c_rs)
        # R200c_rs.append(t_R200c_rs)
        # Mvir_dm.append(t_Mvir_dm)
        # Mvir_gas.append(t_Mvir_gas)
        # Mvir_star.append(t_Mvir_star)

        data[i, 0] = t_hid
        data[i, 1] = t_Mvir_rs
        data[i, 2] = t_Rvir_rs
        data[i, 3] = t_M200c_rs
        data[i, 4] = t_R200c_rs
        data[i, 5] = t_Mvir_dm
        data[i, 6] = t_Mvir_gas
        data[i, 7] = t_Mvir_star

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

        # Mvir_rs = comm.gather(Mvir_rs, root=0)
        # Rvir_rs = comm.gather(Rvir_rs, root=0)
        # M200c_rs = comm.gather(M200c_rs, root=0)
        # R200c_rs = comm.gather(R200c_rs, root=0)
        # Mvir_dm = comm.gather(Mvir_dm, root=0)
        # Mvir_gas = comm.gather(Mvir_gas, root=0)
        # Mvir_star = comm.gather(Mvir_star, root=0)

        with h5py.File(fn, 'w') as f:
            g = f.create_group('results')
            for i in range(nvar):
                d = g.create_dataset(fields[i], data=databuf[:, i])
                d.attrs['units'] = units[i]

if __name__ == '__main__':
    path = '/lustre/scratch/astro/lc589/tot-halo-test'
    iout = 99

    main(path, iout)
