from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import collections
import itertools

from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
locfit = importr('locfit')
fANCOVA = importr('fANCOVA')
from rpy2.robjects import Formula

from processing_variables import *

#*----------------------------------------------------------------------------------------------------------------*#

def apply_to_dict(function,dictionary,dictionary2=None,**kwargs):
    outs = dict()
    if dictionary2==None:
        for key in dictionary.keys():
            outs[key] = function(dictionary[key],**kwargs)
    else:
        for key in dictionary.keys():
            outs[key] = function(dictionary[key],dictionary2[key],**kwargs)
    return outs

#*----------------------------------------------------------------------------------------------------------------*#

def mean_of_list(ls):
    arr = np.array(ls)
    return np.mean(arr,axis=0)

def mean_of_dict(dic,key2=None):
    ls = []
    for item in dic.values():
        if type(item)==dict:
            ls.append(item[key2])
        else:
            ls.append(item)
    return mean_of_list(ls)

#*----------------------------------------------------------------------------------------------------------------*#

def calc_covariance(jkcov, fidreal, fidvd=None, funcidx={'w':range(30),'d':range(30,60),'v':range(60,90),\
                                           'c':range(90,120),'a':range(120,150),'r':range(150,180)}):
    cov = jkcov+np.cov(fidreal.T)
    if fidvd!=None:
        cov[funcidx['v']][:,funcidx['v']] += np.cov(fidvd['vpf'].T)
        cov[funcidx['d']][:,funcidx['d']] += np.cov(fidvd['deltasigma'].T)
    return cov

#*----------------------------------------------------------------------------------------------------------------*#

def cov_to_corr(cov):
    stddev = np.sqrt(np.diag(cov))
    return (cov.T/stddev).T/stddev

def stddev_from_cov(cov):
    return np.sqrt(cov.diagonal())

#*----------------------------------------------------------------------------------------------------------------*#

def inv_cov(cov,rcond=0):
    corr = cov_to_corr(cov)
    stddev = stddev_from_cov(cov)
    invcorr = np.linalg.pinv(corr,rcond=rcond)
    return (invcorr/stddev).T/stddev

#*----------------------------------------------------------------------------------------------------------------*#

def cut_by_func_1D(vec, axis=0, funcnames='wdvcar', funcidx={'w':range(30),'d':range(30,60),'v':range(60,90),\
                                           'c':range(90,120),'a':range(120,150),'r':range(150,180)}):
    vecs = dict()
    for key in funcnames:
        vecs[key] = vec.take(funcidx[key],axis=axis)
    for l in range(2,len(funcnames)+1):
        for key in [''.join(comb) for comb in itertools.combinations(funcnames,l)]:
            vecs[key] = np.concatenate(list(vec.take(funcidx[k],axis=axis) for k in key),axis=axis)
    return vecs

def cut_by_func_2D(mat, funcnames='wdvcar', funcidx={'w':range(30),'d':range(30,60),'v':range(60,90),\
                                           'c':range(90,120),'a':range(120,150),'r':range(150,180)}):
    mats = dict()
    for key in funcnames:
        mats[key] = mat[funcidx[key]][:,funcidx[key]]
    for l in range(2,len(funcnames)+1):
        for key in [''.join(comb) for comb in itertools.combinations(funcnames,l)]:
            mats[key] = np.block([[mat[funcidx[k1]][:,funcidx[k2]] for k2 in key] for k1 in key])
    return mats

#*----------------------------------------------------------------------------------------------------------------*#

def load_pert(threshold):
    df_dict = dict()
    dp_list = np.zeros((7,Nparam))
    for seed in seed_list[threshold]:
        data = np.load('Run_102218/pert_'+threshold[1:3]+'p'+threshold[4]+'_'+seed+'.npz')
        df_dict[seed] = data['func_all']
    for i in range(7):
        dp_list[i] = data['param'][Nparam*i:(i+1)*Nparam,i]
    return df_dict, dp_list

#*----------------------------------------------------------------------------------------------------------------*#

def gcv_alpha_singley(x, y):
    loessfit = fANCOVA.loess_as(x,y,degree=2,criterion='gcv')
    alpha = list(loessfit[10][0])[0]
    return alpha

def min_alphas(func, param):
    return np.array([np.array([gcv_alpha_singley(param[i],func[Nparam*i:(i+1)*Nparam,j])\
                      for j in range(func.shape[1])]) for i in range(7)])

#*----------------------------------------------------------------------------------------------------------------*#

def sigma_to_alpha(sigma, oneside):
    return sigma/oneside

#*----------------------------------------------------------------------------------------------------------------*#

def locfit_deriv_singley(y, alpha, x, x0): #x is perturbed values of one parameter
    formula = Formula('y~x')
    env = formula.environment
    env['x'] = x
    env['y'] = y
    fit = locfit.locfit(formula=formula,deg=2,deriv=1,alpha=alpha)
    dfdp = list(locfit.preplot_locfit(fit,x0)[1])[0]
    return dfdp

def locfit_comb(ys, alphas, xs, x0s):
    return np.array([np.array([locfit_deriv_singley(ys[Nparam*i:(i+1)*Nparam,j],alphas[i],xs[i],x0s[i]) for j in range(ys.shape[1])]) for i in range(7)])

#*----------------------------------------------------------------------------------------------------------------*#

def calc_fisher(dfdp, invcov):
    fmatrix = np.dot(dfdp,np.dot(invcov,dfdp.T))
    return fmatrix

#*----------------------------------------------------------------------------------------------------------------*#

def calc_1sigma(fisher):
    return np.sqrt(np.linalg.inv(fisher).diagonal())

#*----------------------------------------------------------------------------------------------------------------*#
#*----------------------------------------------------------------------------------------------------------------*#
#*----------------------------------------------------------------------------------------------------------------*#
#*----------------------------------------------------------------------------------------------------------------*#
#*----------------------------------------------------------------------------------------------------------------*#
#*----------------------------------------------------------------------------------------------------------------*#
#*----------------------------------------------------------------------------------------------------------------*#


#*----------------------------------------------------------------------------------------------------------------*#
def apply_maxmin_alpha(alphas, min_alphas):
    alphas = np.max((alphas,min_alphas),axis=0)
    return np.min((alphas,np.ones(7)),axis=0)

def plot_average_f_p_single(average_f_p,pertparam,i,j,save=0):
    plt.plot(pertparam[i],average_f_p[Nparam*i:Nparam*i+Nparam,j],'.')