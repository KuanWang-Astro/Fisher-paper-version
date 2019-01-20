import numpy as np

p19p0 = np.array((1.04029, 12.80315, 0.51193, 11.64354, 0., 0.))
p19p5 = np.array((1.11553, 13.06008, 0.44578, 11.75068, 0., 0.))
p20p0 = np.array((1.14385, 13.28584, 0.34846, 11.97186, 0., 0.))
p20p5 = np.array((1.19652, 13.59169, 0.18536, 12.25470, 0., 0.))
p21p0 = np.array((1.33738, 13.98811, 0.55950, 12.82356, 0., 0.))
p0_list = {'-19.0':p19p0,'-19.5':p19p5,'-20.0':p20p0,'-20.5':p20p5,'-21.0':p21p0}
from seedlist_102218 import *
seed_list = {'-19.0':seed_19p0,'-19.5':seed_19p5,'-20.0':seed_20p0,'-20.5':seed_20p5,'-21.0':seed_21p0}

oneside_ranges = np.array([[0.15, 0.4, 0.4,   0.3, 1, 1],
                           [0.15, 0.4, 0.4,   0.3, 1, 1],
                           [0.15, 0.4, 0.34846,  0.3, 1, 1],
                           [0.15, 0.4, 0.18536, 0.3, 1, 1],
                           [0.15, 0.4, 0.4,   0.3, 1, 1]])

oneside_dict = {'-19.0':np.array([0.15, 0.4, 0.4,   0.3, 1, 1]),
                '-19.5':np.array([0.15, 0.4, 0.4,   0.3, 1, 1]),
                '-20.0':np.array([0.15, 0.4, 0.34846,  0.3, 1, 1]),
                '-20.5':np.array([0.15, 0.4, 0.18536, 0.3, 1, 1]),
                '-21.0':np.array([0.15, 0.4, 0.4,   0.3, 1, 1])}

Nparam = 200

keylist = ['w','d','v','c','a','r','wd','wv','wc','wa','wr','dv','dc','da','dr','vc','va','vr','ca','cr','ar',\
        'wdv','wdc','wda','wdr','wvc','wva','wvr','wca','wcr','war','dvc','dva','dvr','dca','dcr','dar',\
         'vca','vcr','var','car','wdvc','wdva','wdvr','wdca','wdcr','wdar','wvca','wvcr','wvar','wcar',\
        'dvca','dvcr','dvar','dcar','vcar','wdvca','wdvcr','wdvar','wdcar','wvcar','dvcar','wdvcar']