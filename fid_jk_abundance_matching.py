import sys
import argparse

parser = argparse.ArgumentParser(description='#')

parser.add_argument('--Lbox',type=float,default=250.,dest='Lbox')
parser.add_argument('--simname',default='bolplanck',dest='simname')
parser.add_argument('--version',default='halotools_v0p4',dest='version')
parser.add_argument('--redshift',type=float,default=0.,dest='redshift')
parser.add_argument('--halofinder',default='rockstar',dest='halofinder')

parser.add_argument('--Nsidejk',type=int,required=True,dest='Nsidejk')
parser.add_argument('--losdir',required=True,dest='losdir')
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
parser.add_argument('--parallel',type=int,default=3,dest='nproc')
args = parser.parse_args()


import collections
import gc
import numpy as np
from concurrent.futures import ProcessPoolExecutor as Pool

from datetime import datetime
print str(datetime.now())

from halotools.sim_manager import CachedHaloCatalog

from halotools.mock_observables import return_xyz_formatted_array
from calc_jk_real import calc_jk_real

##########################################################

output_names = ('jkcov','jkfunc')

##########################################################

if args.threshold==-19.0:
    Ng = int(0.017004429312*args.Lbox**3)
elif args.threshold==-19.5:
    Ng = int(0.011267999744*args.Lbox**3)
elif args.threshold==-20.0:
    Ng = int(0.006515886336*args.Lbox**3)
elif args.threshold==-20.5:
    Ng = int(0.003186637824*args.Lbox**3)
elif args.threshold==-21.0:
    Ng = int(0.001170136576*args.Lbox**3)
    
#########################################################


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

cyl_sum_at = np.concatenate([np.arange(10),np.around(np.logspace(1,np.log10(150),args.Nbins_c-10)).astype(np.int)])
ann_sum_at = np.concatenate([np.arange(10),np.around(np.logspace(1,np.log10(200),args.Nbins_a-10)).astype(np.int)])
rat_bin = np.linspace(0,1,args.Nbins_r+1)

##cic

ptclpos = np.loadtxt(args.ptclpos)
rp_bins_ggl = np.logspace(-1, 1.5, args.Nbins_d+1)
num_ptcls_to_use = len(ptclpos)
##ggl

##########################################################

def calc_all_observables(los_direction):
    
    output = []
    
    if los_direction=='x':
        xyz = 'yzx'
    elif los_direction=='y':
        xyz = 'xzy'
    elif los_direction=='z':
        xyz = 'xyz'
    
    table = halocat.halo_table[np.argsort(-halocat.halo_table['halo_vpeak'])[:Ng]]
    
    pos_gals_d = return_xyz_formatted_array(*(table['halo_'+ax] for ax in xyz), \
            velocity=table['halo_v'+los_direction], velocity_distortion_dimension='z', period=Lbox) ##redshift space distorted
    pos_gals_d = np.array(pos_gals_d,dtype=float)
       
    pos_gals = return_xyz_formatted_array(*(table['halo_'+ax] for ax in xyz), period=Lbox)
    pos_gals = np.array(pos_gals,dtype=float)
    total_num_ptcls_in_snapshot = halocat.num_ptcl_per_dim**3
    downsampling_factor = total_num_ptcls_in_snapshot/float(num_ptcls_to_use)
    particle_masses = halocat.particle_mass*downsampling_factor
    
    
    jkcov, jkfunc = calc_jk_real(pos_gals_d, Lbox, wbool=args.funcbool[0], dbool=args.funcbool[1], vbool=args.funcbool[2], cbool=args.funcbool[3], abool=args.funcbool[4], rbool=args.funcbool[5], jackknife_nside=args.Nsidejk,\
                 rbins_wp=r_wp, zmax=pi_max,\
                 r_vpf=r_vpf, vpf_cen=vpf_centers,\
                 galpos_non_rsd=pos_gals, ptclpos=ptclpos, ptcl_mass=particle_masses, r_ds=rp_bins_ggl,\
                 cyl_r1=proj_search_radius1, cyl_halflen=cylinder_half_length, cyl_bin=cyl_sum_at,\
                 cyl_r2=proj_search_radius2,  ann_bin=ann_sum_at,\
                 rat_bin=rat_bin)
    
    output.append(jkcov)
    output.append(jkfunc)
    
    return output


############################################################

def main(output_fname):
    
    output_dict = collections.defaultdict(list)
    nproc = args.nproc
    
    global halocat
    
    
    halocat = CachedHaloCatalog(simname = args.simname, version_name = args.version, redshift = args.redshift, \
                                halo_finder = args.halofinder)
    output_data = calc_all_observables(args.losdir)
    for name, data in zip(output_names, output_data):
        output_dict[name] = np.array(data)

    np.savez(output_fname, **output_dict)


if __name__ == '__main__':
    main(args.outfile)
    with open(args.outfile+'_log','w') as f:
        f.write(sys.argv[0]+'\n')
        for arg in vars(args):
            f.write(str(arg)+':'+str(getattr(args, arg))+'\n')
    print str(datetime.now())
