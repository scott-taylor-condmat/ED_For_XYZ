import math
import cmath
import numpy as np
from scipy import sparse, special
import scipy.sparse.linalg as splinalg
import time
import sys

def Getrs(Es):
    """Takes a set of energies and returns the gap ratio values"""
    Evals = np.sort(Es)
    dEs = []
    for x in range (0, len(Es)-1):
        dEs.append(Evals[x+1] - Evals[x])
    rs = []
    for x in range (0, len(dEs)-1):
        r0 = min(dEs[x+1], dEs[x])
        r0 /= max(dEs[x+1], dEs[x])
        rs.append(r0)
    return rs

def Entropy_Shannon(v):
    """Calculates the Shannon entropy of a state"""
    mask = abs(v) > 1e-14 #Explicitly remove zero components
    SumThis = abs(v[mask])**2 * np.log(abs(v[mask])**2)
    return -np.sum(SumThis)

def Entropy_vNs(L, v, map):
    """
    Calculates the von Neumann entropy of a state for all partitions
    of the form 1|23...L, 12|3...L, ..., 123...(L-1)|L
    """
    ents = []
    for cut in range (1, L):
        Nl = 2**cut
        Nr = 2**(L - cut)
        A = np.zeros((Nl, Nr), dtype=complex)
        #Put the vector into the full 2**L Fock space and reshape into A
        A[(np.array(map)//Nr).tolist(),(np.array(map)%Nr).tolist()] = np.array(v)
        s = np.linalg.svd(A, compute_uv=0)
        ws = abs(s)**2
        ents.append(np.dot(ws, -np.log(ws)))
    return ents

def Diagonalise_XYZ(L, Jx, Jy, Jz, hs, eps, Nst, **kwargs):
    """
    Diagonalise (using shift-invert to target specific eigenvalues) an 
    XYZ Hamiltonian with parameters:

    L         :  Length of system
    Jx,Jy,Jz  :  Components of nearest-neighbour coupling strengths
    hs        :  Onsite disorder values (array of length L)
    eps       :  Target energy density, defined as eps = (E - E_min) / (E_max - E_min)
    Nst       :  Number of states to find around eps
    Sector    :  Which symmetry sector to solve, based on number of up spins.
                 Options: 'even' and 'odd' when Jx!=Jy, and str(N) for 0<=N<=L when 
                 Jx=Jy. Auto is 'even' or str(L//2)
    BCs       :  Boundary conditions for the system. Options: 'periodic' and 'open'

    Returning as output the eigenvalues, eigenvectors, and extremal eigenvalues
    """
    params = { 'Sector' : 'Auto',
               'BCs'    : 'periodic'}
    params.update(**kwargs)
    Sector = params['Sector']
    bc = params['BCs']

    if Sector == 'Auto':
        if abs(Jx - Jy) < 1e-6:
            sec = str(L//2)
        else:
            sec = 'even'
    else: sec = Sector

    HOps = GetOps(L)
    print('Building Hamiltonian...')
    sys.stdout.flush()
    H = Jz * HOps['ZZ'][bc][sec] / 4.0
    H += (Jx + Jy) * HOps['XY_sym'][bc][sec] / 2.0 / 4.0
    if abs(Jx - Jy) > 1e-6:
        H += (Jx - Jy) * HOps['XY_asym'][bc][sec] / 2.0 / 4.0
    for ii in range (0, L): H += hs[ii] * HOps['Z'][sec][ii] / 2.0
    Map = HOps['Maps'][sec]
    del HOps
    if abs(H.imag).max() < 1e-10: H = H.real
    else: print('Large imaginary elements in Hamiltonian!')
    print('Hamiltonian build complete')
    sys.stdout.flush()

    print('Finding extremal states...')
    sys.stdout.flush()
    Emin = splinalg.eigsh(H, 1, which='SA')[0][0]
    Emax = splinalg.eigsh(H, 1, which='LA')[0][0]
    print('Extremal states found')
    sys.stdout.flush()
    
    print('Performing diagonalisation...')
    sys.stdout.flush()
    t0 = time.time()
    Etarg = Emin + eps * (Emax - Emin)
    Es, vs = splinalg.eigsh(H, Nst, sigma=Etarg, which='LM')
    #Ensure eigenvectors are real
    for ii in range (0, len(vs[0])):
        vs[:,ii] *= abs(vs[0,ii]) / vs[0,ii]
    t1 = time.time()
    print('Diagonalisation complete in %g s' %(t1-t0))

    return Es, vs, (Emin, Emax)

def GetOps(L):
    """
    Load or build the operators requires to build a length L XYZ Hamiltonian.
    Returning a dictionary with the following dictionary entries:
    'ZZ'      : sum_n sigma^z_{n} @ sigma^z_{n+1}
    'XY_sym'  : sum_n sigma^x_{n} @ sigma^x_{n+1} + sigma^y_{n} @ sigma^y_{n+1} 
    'XY_asym' : sum_n sigma^x_{n} @ sigma^x_{n+1} - sigma^y_{n} @ sigma^y_{n+1} 
    'Z'       : sigma^z_{n}
    'Maps'    : list indicating the position of the component of this subspace
                in the full 2**L Fock space

    'ZZ', 'XY_sym' and 'XY_asym' take keywords 'open' and 'periodic' to choose
    boundary condition, then the keyword of the sector
    e.g. Ops['ZZ']['open']['even'] gives the ZZ coupling for a chain with open
    boundary conditions in the even number of up spins sector

    'Z' and 'Maps' take keyword of the sector, giving a list of sigma^z operators
    e.g. Ops['Z']['even'][0] gives the 0th sigma^z operator in the even number of
    up spins sector
    """
    nm = 'L=%d' %L                                                               
    try:
        Ops = np.load(nm+'.npy').item()
        print(nm + ' loaded from file')
        sys.stdout.flush()
    except:
        print(nm + ' cannot be loaded. Generating...')
        sys.stdout.flush()
        eye2 = sparse.lil_matrix([[1.,0.],[0.,1.]])
        Sig_x = sparse.csr_matrix([[0.,1.],[1.,0.]])
        Sig_y = sparse.csr_matrix([[0.,-1j],[1j,0.]])
        Sig_z = sparse.csr_matrix([[1.,0.],[0.,-1.]])

        print('Building Full System Matrices...')
        sys.stdout.flush()
        Sig_x_list = []
        Sig_y_list = []
        Sig_z_list = []

        for ii in range (0, L):
            I_left = sparse.identity(2**ii, format='csr')
            I_right = sparse.identity(2**(L-ii-1), format='csr')

            if ii != 0:
                fullsig_x = sparse.kron(I_left, Sig_x)
                fullsig_y = sparse.kron(I_left, Sig_y)
                fullsig_z = sparse.kron(I_left, Sig_z)
            else:
                fullsig_x = Sig_x
                fullsig_y = Sig_y
                fullsig_z = Sig_z
            if ii != L-1:
                fullsig_x = sparse.kron(fullsig_x, I_right)
                fullsig_y = sparse.kron(fullsig_y, I_right)
                fullsig_z = sparse.kron(fullsig_z, I_right)

            Sig_x_list += [fullsig_x]
            Sig_y_list += [fullsig_y]
            Sig_z_list += [fullsig_z]

        Sig_z_tot = np.sum(Sig_z_list)

        ZZ_nn = []
        XY_sym_nn = []
        XY_asym_nn = []

        for ii in range (0, L-1):
            SigxSigx = Sig_x_list[ii].dot(Sig_x_list[ii+1])
            SigySigy = Sig_y_list[ii].dot(Sig_y_list[ii+1])
            SigzSigz = Sig_z_list[ii].dot(Sig_z_list[ii+1])
            ZZ_nn += [SigzSigz]
            XY_sym_nn += [SigxSigx + SigySigy]
            XY_asym_nn += [SigxSigx - SigySigy]

        SigxSigx = Sig_x_list[-1].dot(Sig_x_list[0])
        SigySigy = Sig_y_list[-1].dot(Sig_y_list[0])
        SigzSigz = Sig_z_list[-1].dot(Sig_z_list[0])
        ZZ_nn += [SigzSigz]
        XY_sym_nn += [SigxSigx + SigySigy]
        XY_asym_nn += [SigxSigx - SigySigy]

        ZZ_open_nn = np.sum(ZZ_nn[:L-1:])
        ZZ_per_nn = ZZ_open_nn + ZZ_nn[-1]
        XY_sym_open_nn = np.sum(XY_sym_nn[:L-1:])
        XY_sym_per_nn = XY_sym_open_nn + XY_sym_nn[-1]
        XY_asym_open_nn = np.sum(XY_asym_nn[:L-1:])
        XY_asym_per_nn = XY_asym_open_nn + XY_asym_nn[-1]

        print('Splitting Into Symmetry Sectors...')
        sys.stdout.flush()
        Sig_z_vals = Sig_z_tot.diagonal()

        ZZ_open = {}
        ZZ_periodic = {}
        XY_open_sym = {}
        XY_periodic_sym = {}
        XY_open_asym = {}
        XY_periodic_asym = {}
        Z = {}
        Maps = {}

        iieven = []
        iiodd = []

        print('Projecting Into Magnetisation Subspaces...')
        sys.stdout.flush()
        for UpN in range (0, L+1):
            Sig_z_targ = np.float(UpN - (L - UpN))
            print('Targeting sigma_z_tot = %d' %Sig_z_targ)
            sys.stdout.flush()
            iis = np.where(abs(Sig_z_vals-Sig_z_targ)<1e-3)[0]
            if UpN % 2 == 0: iieven += iis.tolist()
            else: iiodd += iis.tolist()
            Maps[str(UpN)] = iis

            d = np.size(iis)
            print('Reducing from %d to %d' %(2**L, d))
            sys.stdout.flush()
            Proj = sparse.lil_matrix((d, 2**L))
            for ii in range (0, len(iis)):
                Proj[ii,iis[ii]] = 1.0

            ZZ_open[str(UpN)] = Proj*ZZ_open_nn*Proj.T
            ZZ_periodic[str(UpN)] = Proj*ZZ_per_nn*Proj.T
            XY_open_sym[str(UpN)] = Proj*XY_sym_open_nn*Proj.T
            XY_periodic_sym[str(UpN)] = Proj*XY_sym_per_nn*Proj.T

            small_Sigzs = []
            for ii in range (0, L):
                small_Sigzs.append(Proj*Sig_z_list[ii]*Proj.T)
            Z[str(UpN)] = small_Sigzs

        iiodd = np.sort(iiodd)
        iieven = np.sort(iieven)

        print('Projecting Into Odd Parity Subspace...')
        sys.stdout.flush()
        Projodd = sparse.lil_matrix((len(iiodd), 2**L))
        for ii in range (0, len(iiodd)):
            Projodd[ii,iiodd[ii]] = 1.0
        ZZ_open['odd'] = Projodd*ZZ_open_nn*Projodd.T
        ZZ_periodic['odd'] = Projodd*ZZ_per_nn*Projodd.T
        XY_open_sym['odd'] = Projodd*XY_sym_open_nn*Projodd.T
        XY_periodic_sym['odd'] = Projodd*XY_sym_per_nn*Projodd.T
        XY_open_asym['odd'] = Projodd*XY_asym_open_nn*Projodd.T
        XY_periodic_asym['odd'] = Projodd*XY_asym_per_nn*Projodd.T
        small_Sigzs = []
        for ii in range (0, L):
            small_Sigzs.append(Projodd*Sig_z_list[ii]*Projodd.T)
        Z['odd'] = small_Sigzs
        Maps['odd'] = iiodd

        print('Projecting Into Even Parity Subspace...')
        sys.stdout.flush()
        Projeven = sparse.lil_matrix((len(iieven), 2**L))
        for ii in range (0, len(iieven)):
            Projeven[ii,iieven[ii]] = 1.0
        ZZ_open['even'] = Projeven*ZZ_open_nn*Projeven.T
        ZZ_periodic['even'] = Projeven*ZZ_per_nn*Projeven.T
        XY_open_sym['even'] = Projeven*XY_sym_open_nn*Projeven.T
        XY_periodic_sym['even'] = Projeven*XY_sym_per_nn*Projeven.T
        XY_open_asym['even'] = Projeven*XY_asym_open_nn*Projeven.T
        XY_periodic_asym['even'] = Projeven*XY_asym_per_nn*Projeven.T
        small_Sigzs = []
        for ii in range (0, L):
            small_Sigzs.append(Projeven*Sig_z_list[ii]*Projeven.T)
        Z['even'] = small_Sigzs
        Maps['even'] = iieven

        Ops = {
            'ZZ' : {'open': ZZ_open,
                    'periodic': ZZ_periodic},
            'XY_sym' : {'open': XY_open_sym,
                        'periodic': XY_periodic_sym},
            'XY_asym' : {'open': XY_open_asym,
                         'periodic': XY_periodic_asym},
            'Z' : Z,
            'Maps' : Maps
            }
        np.save(nm, Ops)
        print('Operators Saved To File')
        sys.stdout.flush()

    return Ops


if __name__ == "__main__":
    if False:
        print('Checking the structure of the Operator dictionary')
        mydic = GetOps(6)
        for k in mydic.keys():
            for k2 in mydic[k].keys():
                try:
                    print(k, k2, mydic[k][k2].keys())
                except:
                    print(k, k2, np.shape(mydic[k][k2]))
                    
    if True:
        L = 10
        OpDict = mydic = GetOps(L)
        Jx, Jy, Jz = 0.4, 1.6, 1.2
        hs = (np.random.random(L) * 2.0 - 1.0) * 10.0
        eps = 0.5
        Nst = 50
        sec = 'odd'
        bc = 'open'
        map = OpDict['Maps'][sec]
        Es, vs, bnds = Diagonalise_XYZ(L, Jx, Jy, Jz, hs, eps, Nst, Sector=sec, BCs=bc)

        rs = Getrs(Es)
        AllShannons, AllvNs_mid = [], []
        for ii in range (0, len(vs[0])):
            AllShannons.append(Entropy_Shannon(vs[:,0]))
            MyvNs = Entropy_vNs(L, vs[:,ii], map)
            AllvNs_mid.append((MyvNs[(L-1)//2]))

        print('Average r:', np.average(rs))
        print('Average Shannon entropy:', np.average(AllShannons))
        print('Avergae half-chain entanglement:', np.average(AllvNs_mid))
