from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import collections
from scipy import stats
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
#returns dictionary with same keys as dictionary and dictionary2

#*----------------------------------------------------------------------------------------------------------------*#

def mean_of_list(ls):
    arr = np.array(ls)
    return np.mean(arr,axis=0)
#returns np array with same dimensions as list elements

def mean_of_dict(dic,key2=None):
    ls = []
    for item in dic.values():
        if type(item)==dict:
            ls.append(item[key2])
        else:
            ls.append(item)
    return mean_of_list(ls)
#returns np array with same dimensions as dictionary items

def trimmean_of_list(ls):
    arr = np.array(ls)
    return stats.trim_mean(arr,0.16,axis=0)
#returns np array with same dimensions as list elements

def trimmean_of_dict(dic,key2=None):
    ls = []
    for item in dic.values():
        if type(item)==dict:
            ls.append(item[key2])
        else:
            ls.append(item)
    return trimmean_of_list(ls)
#returns np array with same dimensions as dictionary items

def median_of_list(ls):
    arr = np.array(ls)
    return np.median(arr,axis=0)
#returns np array with same dimensions as list elements

def median_of_dict(dic,key2=None):
    ls = []
    for item in dic.values():
        if type(item)==dict:
            ls.append(item[key2])
        else:
            ls.append(item)
    return median_of_list(ls)
#returns np array with same dimensions as dictionary items

def std_of_list(ls):
    arr = np.array(ls)
    return np.std(arr,axis=0)
#returns np array with same dimensions as list elements

def std_of_dict(dic,key2=None):
    ls = []
    for item in dic.values():
        if type(item)==dict:
            ls.append(item[key2])
        else:
            ls.append(item)
    return std_of_list(ls)
#returns np array with same dimensions as dictionary items

#*----------------------------------------------------------------------------------------------------------------*#

def calc_covariance(jkcov, fidreal, fidvd=None, funcidx={'w':range(30),'d':range(30,60),'v':range(60,90),\
                                           'c':range(90,120),'a':range(120,150),'r':range(150,180)}):
    cov = jkcov+np.cov(fidreal.T)
    if fidvd!=None:
        cov[funcidx['v']][:,funcidx['v']] += np.cov(fidvd['vpf'].T)
        cov[funcidx['d']][:,funcidx['d']] += np.cov(fidvd['deltasigma'].T)
    return cov
#returns nbin*nbin np array

#*----------------------------------------------------------------------------------------------------------------*#

def cov_to_corr(cov):
    stddev = np.sqrt(np.diag(cov))
    return (cov.T/stddev).T/stddev
#returns 2D np array with same dimensions as cov

def stddev_from_cov(cov):
    return np.sqrt(cov.diagonal())
#returns 1D np array with same length as cov

#*----------------------------------------------------------------------------------------------------------------*#

def inv_cov(cov,rcond=0):
    corr = cov_to_corr(cov)
    stddev = stddev_from_cov(cov)
    invcorr = np.linalg.pinv(corr,rcond=rcond)
    return (invcorr/stddev).T/stddev
#returns 2D np array with same dimensions as cov

#*----------------------------------------------------------------------------------------------------------------*#

def cut_by_func_1D(vec, axis=0, funcnames='wdvcar', funcidx={'w':range(1,30),'d':range(30,60),'v':range(60,90),\
                                           'c':range(90,120),'a':range(120,150),'r':range(150,180)}):
    vecs = dict()
    for key in funcnames:
        vecs[key] = vec.take(np.concatenate((np.zeros(1).astype(int),funcidx[key])),axis=axis)
    for l in range(2,len(funcnames)+1):
        for key in [''.join(comb) for comb in itertools.combinations(funcnames,l)]:
            vecs[key] = np.concatenate((vec.take([0,],axis=axis),np.concatenate(list(vec.take(funcidx[k],axis=axis) for k in key),axis=axis)),axis=axis)
    return vecs
#returns dictionary

def cut_by_func_2D(mat, funcnames='wdvcar', funcidx={'w':range(1,30),'d':range(30,60),'v':range(60,90),\
                                           'c':range(90,120),'a':range(120,150),'r':range(150,180)}):
    mats = dict()
    funcidx['n'] = [0,]
    for key in funcnames:
        mats[key] = mat[np.concatenate((np.zeros(1).astype(int),funcidx[key]))][:,np.concatenate((np.zeros(1).astype(int),funcidx[key]))]
    for l in range(2,len(funcnames)+1):
        for key in [''.join(comb) for comb in itertools.combinations(funcnames,l)]:
            mats[key] = np.block([[mat[funcidx[k1]][:,funcidx[k2]] for k2 in 'n'+key] for k1 in 'n'+key])
    return mats
#returns dictionary

#*----------------------------------------------------------------------------------------------------------------*#
idx_0 = np.concatenate((np.arange(600),np.arange(800,1400)))
def load_pert(threshold):
    df_dict = dict()
    dp_list = np.zeros((7,Nparam))
    for seed in seed_list[threshold]:
        data = np.load('Run_102218/pert_'+threshold[1:3]+'p'+threshold[4]+'_'+seed+'.npz')
        df_dict[seed] = data['func_all'][idx_0]
    for i in range(7):
        dp_list[i] = data['param'][Nparam*i:(i+1)*Nparam,i]
    dp_list = dp_list[[0,1,2,4,5,6]]
    return df_dict, dp_list
#returns function dictionary of seeds, and 7*Nparam perturbed parameter np array

#*----------------------------------------------------------------------------------------------------------------*#

def gcv_alpha_singley(x, y):
    loessfit = fANCOVA.loess_as(x,y,degree=2,criterion='gcv')
    alpha = list(loessfit[10][0])[0]
    return alpha
#returns single value

def gcv_alphas(func, param, apply_min=0, min_a=None):
    if apply_min:
        return np.array([np.array([max(gcv_alpha_singley(param[i],func[Nparam*i:(i+1)*Nparam,j]),min_a)\
                      for j in range(func.shape[1])]) for i in range(6)])

    else:
        return np.array([np.array([gcv_alpha_singley(param[i],func[Nparam*i:(i+1)*Nparam,j])\
                      for j in range(func.shape[1])]) for i in range(6)])
#returns 7*len(func) 2D np array

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
#returns single value

def locfit_comb(ys, alphas, xs, x0s):
    return np.array([np.array([locfit_deriv_singley(ys[Nparam*i:(i+1)*Nparam,j],alphas[i,j],xs[i],x0s[i]) for j in range(ys.shape[1])]) for i in range(6)])
#returns 7*len(func) 2D np array 

#*----------------------------------------------------------------------------------------------------------------*#

def calc_fisher(dfdp, invcov):
    fmatrix = np.dot(dfdp,np.dot(invcov,dfdp.T))
    return fmatrix

#*----------------------------------------------------------------------------------------------------------------*#

def calc_1sigma(fisher):
    return np.sqrt(np.linalg.inv(fisher).diagonal())

#*----------------------------------------------------------------------------------------------------------------*#

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
            list([r'$P(%.3f<N_2/N_5<%.3f)$'%(ratio_bin[i],ratio_bin[i+1]) for i in range(30)])
            
def plot_fi_pj(i,j,pertfunc_dict,gcv_alphas,save=0):
    fig = plt.figure(figsize=(13,10))
    plt.xlabel(param_list[i],fontsize=30)
    plt.ylabel(obs_list[j],fontsize=30)
    for seed in pertfunc_dict.keys():
        plt.plot(pertparam[i],pertfunc_dict[seed][i*200:i*200+200,j],'.',markersize=2,alpha=0.4)
    plt.plot(pertparam[i],average_curves[200*i:200*i+200,j],'b-',label='averaged curve')
    dfdp = pf.locfit_deriv_singley(average_curves[200*i:200*i+200,j],gcv_alphas[i,j],pertparam[i],p0[i])
    plt.plot(pertparam[i],(pertparam[i]-p0[i])*dfdp+average_curves[200*i+100,j],'r--',lw=2,label='fit with smoothing from GCV')
    plt.axvline(p0[i],c='k',ls='--')
    plt.axvspan(p0[i]-gcv_alphas[i,j]*oneside[i],p0[i]+gcv_alphas[i,j]*oneside[i],color='b',alpha=0.1,label='smoothing scale from GCV')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(frameon=False,fontsize=20)
    if save:
        fig.savefig('Plots_102218/'+threshold+'_p'+str(i)+'_f'+str(j)+'.png')
        plt.close(fig)

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