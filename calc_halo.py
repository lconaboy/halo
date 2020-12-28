import numpy as np
from seren3.array import SimArray
from cosmolopy.distance import e_z
from cosmolopy.density import cosmo_densities, omega_M_z

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

    assert(f == 'vir' or (type(f) == type(1.) or
           type(f) == type(1))), "f should be 'vir' or number"

    # rho_crit(0) = 3 * H0**2 / 8 * \pi * G
    # H(z) = H0 * E(z)
    # rho_crit(z) = rho_crit(0) * E(z)**2
    
    rho_crit, rho_bar = cosmo_densities(**cosmo)
    E_z = e_z(cosmo['z'], **cosmo)
    rho_crit_z = rho_crit * E_z**2
    omega_m_z = omega_M_z(z, **cosmo)
    
    if f == 'vir':
        X = 18*np.pi**2 + 82.*(omega_m_z - 1.) - 39*(omega_m_z - 1.)**2
    else:
        X = float(f)

    r = ((3 .* m)/(4. * np.pi * X * rho_crit_z)) ** 1./3.  # Mpc
    r = r * 1e3 * cosmo['h'] / cosmo['aexp']  # a kpc h**-1
    
    return r

        
def main(path, ioutput)
    from seren3.analysis.parallel import mpi
    
    if mpi.host:
        print('Working on output', ioutput)
        print(path)

    sim = seren3.init(path)
    snap = sim[ioutput]
    subsnap = snap

    snap.set_nproc(1)
    halos = subsnap.halos(finder='rockstar')
    nhalos = len(halos)

    halo_ix = None
    if mpi.host:
        halo_ix = halos.halo_ix(shuffle=True)[0:3]

    dest = {}
    for i, sto in mpi.piter(halo_ix, storage=dest, print_stats=False):
        h = halos[i]

        # The quantities returned by rockstar
        Mvir_rs = h.properties['mvir']/subsnap.cosmo['h']  # Msol
        Rvir_rs = h.properties['rvir']/subsnap.cosmo['h']  # a * kpc

        # The quantities calculated at the virial radius calculated by rockstar
        Mvir_gas = h.g["mass"].flatten()["mass"].in_units("Msol").sum()   # Msol
        Mvir_dm = h.d["mass"].flatten()["mass"].in_units("Msol").sum()    # Msol
        s_dset = h.s[["age", "mass"]].flatten()
        if (len(s_dset["age"]) > 0):
            Mvir_star = s_dset["mass"].in_units("Msol").sum()  # Msol
        else:
            Mvir_star = 0.0

        # The quantities calculated at R200c, as determined by the
        # M200c returned by rockstar
        M200c_rs = h.properties['m200c']/subsnap.cosmo['h']        # Msol
        R200c_rs = M_to_R(M200c_rs, f=200,
                          cosmo=subsnap.cosmo)/subsnap.cosmo['h']  # a * kpc


        sto.idx = h["id"]
        sto.result = {"M_tot" : M_tot, "R_vir" : R_vir.in_units('km'), "M_gas" : M_gas, \
                          "M_star" : s_dset["mass"].in_units("Msol").sum()}

            


    if mpi.host:
        import os, h5py
        
        fn = os.path.join(path, 'out_{0:d}.h5'.format(ioutput))
        res = mpi.unpack(dest)
        with h5py.File(, 'w') as f:
            g = f.create_group('Results')
            for key in res:
                g.create_dataset(key, res[key])
                

if __name__ == '__main__':
    path = '/lustre/scratch/astro/lc589/tot-halo-test'
    iout = 99

    main(path, iout)
