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
                           [0.15, 0.4, 0.35,  2, 0.3, 1, 1],
                           [0.15, 0.4, 0.185, 2, 0.3, 1, 1],
                           [0.15, 0.4, 0.4,   2, 0.3, 1, 1]])

oneside_dict = {'-19.0':np.array([0.15, 0.4, 0.4,   2, 0.3, 1, 1]),
                '-19.5':np.array([0.15, 0.4, 0.4,   2, 0.3, 1, 1]),
                '-20.0':np.array([0.15, 0.4, 0.35,  2, 0.3, 1, 1]),
                '-20.5':np.array([0.15, 0.4, 0.185, 2, 0.3, 1, 1]),
                '-21.0':np.array([0.15, 0.4, 0.4,   2, 0.3, 1, 1])}

Nparam = 200

ntex = r'$n_\mathrm{g}$'
wtex = r'$w_\mathrm{p}(r_\mathrm{p})$'
dtex = r'$\Delta\Sigma(r_\mathrm{p})$'
vtex = r'$VPF(r)$'
ctex = r'$P(N_{\rm{cic}})$'
atex = r'$P(N_{\rm{cia}})$'
rtex = r'$P(N_{\mathrm{1}}/N_{\mathrm{2}})$'

ftex_dict = {'n':ntex,'w':wtex,'d':dtex,'v':vtex,'c':ctex,'a':atex,'r':rtex}

param_list = [r'$\alpha$', r'$\log{M_\mathrm{1}}$', r'$\sigma_{\log{M}}$', r'$\log{M_\mathrm{0}}$', r'$\log{M_{\rm{min}}}$',\
              r'$A_{\rm{cen}}$', r'$A_{\rm{sat}}$']