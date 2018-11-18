import sys
import argparse

parser = argparse.ArgumentParser(description='#')

parser.add_argument('--Lbox',type=float,default=250.,dest='Lbox')
parser.add_argument('--simname',default='bolplanck',dest='simname')
parser.add_argument('--version',default='halotools_v0p4',dest='version')
parser.add_argument('--redshift',type=float,default=0.,dest='redshift')
parser.add_argument('--halofinder',default='rockstar',dest='halofinder')

parser.add_argument('--seed',type=int,default=-1,dest='seed')

parser.add_argument('--Nparam',type=int,required=True,dest='Nparam')

parser.add_argument('--funcbool',nargs=6,type=int,required=True,dest='funcbool')

parser.add_argument('--Nbins_w',type=int,default=30,dest='Nbins_w')
parser.add_argument('--Nbins_d',type=int,default=30,dest='Nbins_d')
parser.add_argument('--Nbins_v',type=int,default=30,dest='Nbins_v')
parser.add_argument('--Nbins_c',type=int,default=30,dest='Nbins_c')
parser.add_argument('--Nbins_a',type=int,default=30,dest='Nbins_a')
parser.add_argument('--Nbins_r',type=int,default=30,dest='Nbins_r')

parser.add_argument('--vpfcen',default='vpf_centers_250_1e5.txt',dest='vpfcen')
parser.add_argument('--ptclpos',default='bolplanck_1of10_ptcl.txt',dest='ptclpos')
parser.add_argument('--threshold',required=True,type=float,dest='threshold')
parser.add_argument('--outfile',required=True,dest='outfile')
parser.add_argument('--parallel',type=int,default=55,dest='nproc')

args = parser.parse_args()


import collections
import gc
import numpy as np
from concurrent.futures import ProcessPoolExecutor as Pool

from datetime import datetime
print str(datetime.now())


from HOD_models_fix import decorated_hod_model

from halotools.sim_manager import CachedHaloCatalog
from halotools.mock_observables import return_xyz_formatted_array

from calc_jk_real import calc_jk_real

##########################################################

param_names = ('alpha','logM1','sigma_logM','logM0','logMmin','mean_occupation_centrals_assembias_param1','mean_occupation_satellites_assembias_param1')
output_names = ('func_all','param')

##########################################################

p19p0 = np.array((1.04029, 12.80315, 0.51193, 10.25010, 11.64354, 0., -0.6))
p19p5 = np.array((1.11553, 13.06008, 0.44578, 11.29134, 11.75068, 0., -0.6))
p20p0 = np.array((1.14385, 13.28584, 0.34846, 11.30750, 11.97186, 0., -0.6))
p20p5 = np.array((1.19652, 13.59169, 0.18536, 11.20134, 12.25470, 0., -0.6))
p21p0 = np.array((1.33738, 13.98811, 0.55950, 11.95797, 12.82356, 0., -0.6))

if args.threshold==-19.0:
    fiducial_p = p19p0
elif args.threshold==-19.5:
    fiducial_p = p19p5
elif args.threshold==-20.0:
    fiducial_p = p20p0
elif args.threshold==-20.5:
    fiducial_p = p20p5
elif args.threshold==-21.0:
    fiducial_p = p21p0
    
#########################################################

if args.seed==-1:
    seed = np.random.randint(100000)
else:
    seed = args.seed

Lbox = args.Lbox

pi_max = 60
r_wp = np.logspace(-1, 1.5, args.Nbins_w)
##wp

r_vpf = np.logspace(0,1.,args.Nbins_v)
num_sphere = int(1e5)
vpf_centers = np.loadtxt(args.vpfcen)
##vpf

proj_search_radius1 = 2.0         ##a cylinder of radius 2 Mpc/h
proj_search_radius2 = 5.0         ##a cylinder of radius 5 Mpc/h
cylinder_half_length = 10.0      ##half-length 10 Mpc/h

cyl_sum_at = np.concatenate([np.arange(10),np.around(np.logspace(1,np.log10(299),args.Nbins_c-9)).astype(np.int)])[:-1]
ann_sum_at = np.concatenate([np.arange(10),np.around(np.logspace(1,np.log10(499),args.Nbins_a-9)).astype(np.int)])[:-1]
rat_bin = np.linspace(0,1,args.Nbins_r+1)
##cic

ptclpos = np.loadtxt(args.ptclpos)
rp_bins_ggl = np.logspace(-1, 1.5, args.Nbins_d+1)
num_ptcls_to_use = len(ptclpos)
##ggl


##########################################################


def calc_all_observables(param,seed=seed):

    model.param_dict.update(dict(zip(param_names, param)))    ##update model.param_dict with pairs (param_names:params)
    
    try:
        model.mock.populate(seed=seed)
    except:
        model.populate_mock(halocat, seed=seed)
    
    gc.collect()
    
    output = []
    
    pos_gals_d = return_xyz_formatted_array(*(model.mock.galaxy_table[ax] for ax in 'xyz'), \
            velocity=model.mock.galaxy_table['vz'], velocity_distortion_dimension='z',\
                                          period=Lbox)             ##redshift space distorted
    pos_gals_d = np.array(pos_gals_d,dtype=float)
    
    pos_gals = return_xyz_formatted_array(*(model.mock.galaxy_table[ax] for ax in 'xyz'), period=Lbox)
    pos_gals = np.array(pos_gals,dtype=float)
    particle_masses = halocat.particle_mass
    total_num_ptcls_in_snapshot = halocat.num_ptcl_per_dim**3
    downsampling_factor = total_num_ptcls_in_snapshot/float(num_ptcls_to_use)
    
    func = calc_jk_real(pos_gals_d, Lbox, wbool=args.funcbool[0], dbool=args.funcbool[1], vbool=args.funcbool[2], cbool=args.funcbool[3], abool=args.funcbool[4], rbool=args.funcbool[5], jackknife_nside=0,\
                 rbins_wp=r_wp, zmax=pi_max,\
                 r_vpf=r_vpf, vpf_cen=vpf_centers,\
                 galpos_non_rsd=pos_gals, ptclpos=ptclpos, ptcl_mass=particle_masses, ptcl_per_dim=halocat.num_ptcl_per_dim, r_ds=rp_bins_ggl,\
                 cyl_r1=proj_search_radius1, cyl_halflen=cylinder_half_length, cyl_bin=cyl_sum_at,\
                 cyl_r2=proj_search_radius2,  ann_bin=ann_sum_at,\
                 rat_bin=rat_bin)
    
    output.append(func) 

    
    # parameter set
    output.append(param)
    
    return output


############################################################

def main(model_gen_func, fiducial, output_fname):
    global model
    model = model_gen_func()
    global fid
    fid = np.array(fiducial)
    params = fid*np.ones((7*args.Nparam,7))
    dp_range = np.array((0.15,0.4,0.4,2,0.3))

    for i in range(5):
        params[args.Nparam*i:args.Nparam*(i+1),i] += np.linspace(-1,1,args.Nparam)*min(dp_range[i],fid[i])
    params[args.Nparam*5:args.Nparam*6,5] = np.linspace(-1,1,args.Nparam)
    params[args.Nparam*6:args.Nparam*7,6] = np.linspace(-1,-0.2,args.Nparam)


    
    output_dict = collections.defaultdict(list)
    nproc = args.nproc
    
    global halocat
    
    with Pool(nproc) as pool:
        if 1:
            halocat = CachedHaloCatalog(simname = args.simname, version_name = args.version, redshift = args.redshift, \
                                halo_finder = args.halofinder)
            model.populate_mock(halocat)
            for i, output_data in enumerate(pool.map(calc_all_observables, params)):
                #if i%nproc == nproc-1:
                #    print i
                #    print str(datetime.now())
                for name, data in zip(output_names, output_data):
                    output_dict[name].append(data)
    
    for name in output_names:
        output_dict[name] = np.array(output_dict[name])

    np.savez(output_fname, **output_dict)


if __name__ == '__main__':
    main(decorated_hod_model, fiducial_p, args.outfile+'_%05d'%(seed,))
    with open(args.outfile+'_%05d_log'%(seed,),'w') as f:
        f.write(sys.argv[0]+'\n')
        f.write('seed:'+str(seed)+'\n')
        for arg in vars(args):
            f.write(str(arg)+':'+str(getattr(args, arg))+'\n')
    print str(datetime.now())


