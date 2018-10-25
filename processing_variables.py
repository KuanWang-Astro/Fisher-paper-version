import numpy as np

p19p0 = np.array((1.04029, 12.80315, 0.51193, 10.25010, 11.64354, 0., 0.))
p19p5 = np.array((1.11553, 13.06008, 0.44578, 11.29134, 11.75068, 0., 0.))
p20p0 = np.array((1.14385, 13.28584, 0.34846, 11.30750, 11.97186, 0., 0.))
p20p5 = np.array((1.19652, 13.59169, 0.18536, 11.20134, 12.25470, 0., 0.))
p21p0 = np.array((1.33738, 13.98811, 0.55950, 11.95797, 12.82356, 0., 0.))
p0_list = {'-19.0':p19p0,'-19.5':p19p5,'-20.0':p20p0,'-20.5':p20p5,'-21.0':p21p0}
from seedlist_102218 import *
seed_list = {'-19.0':seed_19p0,'-19.5':seed_19p5,'-20.0':seed_20p0,'-20.5':seed_20p5,'-21.0':seed_21p0}

oneside_ranges = np.array([[0.15, 0.4, 0.4,   2, 0.3, 1, 1],
                           [0.15, 0.4, 0.4,   2, 0.3, 1, 1],
                           [0.15, 0.4, 0.34846,  2, 0.3, 1, 1],
                           [0.15, 0.4, 0.18536, 2, 0.3, 1, 1],
                           [0.15, 0.4, 0.4,   2, 0.3, 1, 1]])

oneside_dict = {'-19.0':np.array([0.15, 0.4, 0.4,   2, 0.3, 1, 1]),
                '-19.5':np.array([0.15, 0.4, 0.4,   2, 0.3, 1, 1]),
                '-20.0':np.array([0.15, 0.4, 0.34846,  2, 0.3, 1, 1]),
                '-20.5':np.array([0.15, 0.4, 0.18536, 2, 0.3, 1, 1]),
                '-21.0':np.array([0.15, 0.4, 0.4,   2, 0.3, 1, 1])}

Nparam = 200

param_list = [r'$\alpha$', r'$\log{M_\mathrm{1}}$', r'$\sigma_{\log{M}}$', r'$\log{M_\mathrm{0}}$', r'$\log{M_{\rm{min}}}$',\
              r'$A_{\rm{cen}}$', r'$A_{\rm{sat}}$']

r_wp = np.logspace(-1,1.5,30)
r_wp = (r_wp[1:]+r_wp[:-1])/2.

r_ds = np.logspace(-1,1.5,31)
r_ds = np.sqrt(0.5*(r_ds[1:]**2 + r_ds[:-1]**2))

r_vpf = np.logspace(0,1,30)

cic_bin = np.concatenate([np.arange(10),np.around(np.logspace(1,np.log10(150),30-10)).astype(np.int)])
cia_bin = np.concatenate([np.arange(10),np.around(np.logspace(1,np.log10(200),30-10)).astype(np.int)])

ratio_bin = np.linspace(0,1,31)

obs_list = [r'$n_{\mathrm{gal}}$',]+\
            list([r'$w_{\mathrm{p}}(r_{\mathrm{p}}= %.2f h^{-1}\mathrm{Mpc})$'%r for r in r_wp])+\
            list([r'$\Delta \Sigma(r_{\mathrm{p}}=%.2f h^{-1}\mathrm{Mpc})$'%r for r in r_ds])+\
            list([r'$\mathrm{VPF}(r=%.2f h^{-1}\mathrm{Mpc})$'%r for r in r_vpf])+\
            list([r'$P(N_{\mathrm{CIC}}=%d)$'%i for i in range(10)])+\
            list([r'$P(%d\leq N_{\mathrm{CIC}}<%d)$'%(cic_bin[i],cic_bin[i+1]) for i in range(10,29)])+\
            [r'$P(N_{\mathrm{CIC}} \geq %d)$'%cic_bin[-1],]+\
            list([r'$P(N_{\mathrm{CIA}}=%d)$'%i for i in range(10)])+\
            list([r'$P(%d\leq N_{\mathrm{CIA}}<%d)$'%(cia_bin[i],cia_bin[i+1]) for i in range(10,29)])+\
            [r'$P(N_{\mathrm{CIA}} \geq %d)$'%cia_bin[-1],]+\
            list([r'$P(%.2f<N_2/N_5<%.2f)$'%(ratio_bin[i],ratio_bin[i+1]) for i in range(30)])