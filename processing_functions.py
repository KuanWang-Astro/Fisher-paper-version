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

########################################################################################
# Calculate the total covariance from separate files ############## calc_covariance ####
# Load pert files ################################################# load_pert ##########
# Calculate smoothing parameter from gcv ########################## gcv_alpha ##########
# Calculate minimal smoothing parameter for 7 parameters ########## min_alphas #########
# Ensure the alphas returned are >= min_alphas and <= 1 ########### apply_maxmin_alpha #
# Calculate updated smoothing parameter from sigma feedback ####### sigma_to_alpha #####
# Calculate derivative given file and alpha ####################### locfit_deriv #######
# Cut and reassemble array by function in 1 dimension ############# cut_by_func_1D #####
# Cut and reassemble matrix by function in 2 dimensions ########### cut_by_func_2D #####
# Calculate correlation matrix from covariance matrix ############# cov_to_corr ########
# Calculate diagonal standard deviations from covariance matrix ### stddev_from_cov ####
# Plot eigenvalues of correlation from covariance ################# plot_eigvals #######
# Calculate inverse covariance matrix as a function of rcond ###### inv_cov ############
# Calculate Fisher matrix from dfdp and inverse covariance ######## calc_fisher ########
# Calculate posterior diagonal variance from Fisher matrix ######## calc_sigma2 ########
# Calculate posterior constraint from Fisher matrix ############### calc_1sigma ########
# Apply a function to items in a dictionary, return a dictionary ## apply_to_dict ######
# Calculate mean across a list #################################### mean_of_list #######
# Calculate standard deviation across a list ###################### stddev_of_list #####
# Calculate mean across all keys of a dictionary ################## mean_of_dict #######
# Calculate standard deviation across all keys of a dictionary #### stddev_of_dict #####
# Calculate 2D covariance for ellipse from Fisher matrix ########## cov2d_from_fisher ##
# Plot ellipse from numbers ####################################### plot_ellipse #######
# Plot contour from covariance matrix ############################# ellipse_from_cov ###
########################################################################################

def calc_covariance(fidjk, fidvd, funcidx={'w':range(20),'d':range(20,40),'v':range(40,60),\
                                           'c':range(60,100),'a':range(100,140),'r':range(140,165)}):
    cov = fidjk['func_all_cov'][0]+np.cov(fidjk['func_all'].T)
    cov[funcidx['v']][:,funcidx['v']] += np.cov(fidvd['vpf'].T)
    cov[funcidx['d']][:,funcidx['d']] += np.cov(fidvd['deltasigma'].T)
    return cov

def load_pert(threshold):
    df_dict = dict()
    dp_dict = dict()
    for seed in seed_list[threshold]:
        data = np.load('bolp_'+threshold[1:3]+'p'+threshold[4]+'_um0_'+seed+'.npz')
        dp_dict[seed] = data['param']
        df_dict[seed] = data['func_all']
    return df_dict, dp_dict

def gcv_alpha(x, ys): #each column of ys should be one f(x)
    alphas = np.zeros(len(ys.T))
    for i,y in enumerate(ys.T):
        loessfit = fANCOVA.loess_as(x,y,degree=2,criterion='gcv')
        alphas[i] = list(loessfit[10][0])[0]
    return alphas

def min_alphas(func,param): #returns array of 7
    return np.array([4.*min(gcv_alpha(param[Nparam*i:(i+1)*Nparam,i],func[Nparam*i:(i+1)*Nparam]))for i in range(7)])

def apply_maxmin_alpha(alphas,min_alphas):
    alphas = np.max((alphas,min_alphas),axis=0)
    return np.min((alphas,np.ones(7)),axis=0)

def sigma_to_alpha(sigma,oneside):
    return 2.*sigma/oneside

def locfit_deriv(ys,alpha,x,x0): #x is perturbed values of one parameter
    formula = Formula('y~x')
    env = formula.environment
    env['x'] = x
    dfdp = np.zeros(len(ys.T))
    for i,y in enumerate(ys.T):
        env['y'] = y
        fit = locfit.locfit(formula=formula,deg=2,deriv=1,alpha=alpha)
        dfdp[i] = list(locfit.preplot_locfit(fit,x0)[1])[0]
    return dfdp

def locfit_comb(ys,alphas,xs,x0s):
    return np.array([locfit_deriv(ys[Nparam*i:(i+1)*Nparam],alphas[i],xs[Nparam*i:(i+1)*Nparam,i],x0s[i]) for i in range(7)])

def cut_by_func_1D(vec, axis=0, funcnames='wdvcar', funcidx={'w':range(20),'d':range(20,40),'v':range(40,60),\
                                                           'c':range(60,100),'a':range(100,140),'r':range(140,165)}):
    vecs = dict()
    for key in funcnames:
        vecs[key] = vec.take(funcidx[key],axis=axis)
    vecs[funcnames] = vec
    for l in range(2,len(funcnames)):
        for key in [''.join(comb) for comb in itertools.combinations(funcnames,l)]:
            vecs[key] = np.concatenate(list(vec.take(funcidx[k],axis=axis) for k in key),axis=axis)
    return vecs

def cut_by_func_2D(mat, funcnames='wdvcar', funcidx={'w':range(20),'d':range(20,40),'v':range(40,60),\
                                                     'c':range(60,100),'a':range(100,140),'r':range(140,165)}):
    mats = dict()
    for key in funcnames:
        mats[key] = mat[funcidx[key]][:,funcidx[key]]
    mats[funcnames] = mat
    for l in range(2,len(funcnames)):
        for key in [''.join(comb) for comb in itertools.combinations(funcnames,l)]:
            mats[key] = np.block([[mat[funcidx[k1]][:,funcidx[k2]] for k2 in key] for k1 in key])
    return mats

def cov_to_corr(cov):
    stddev = np.sqrt(np.diag(cov))
    return (cov.T/stddev).T/stddev

def stddev_from_cov(cov):
    return np.sqrt(cov.diagonal())

def plot_eigvals(cov):
    eigval_sort = -np.sort(-np.linalg.eigvals(cov_to_corr(cov)))
    #if eigval_sort[-1]/eigval_sort[0]>1e-4:
    if 1:
        print eigval_sort[-1]/eigval_sort[0]
    #else:
    #    plt.figure()
    #    plt.plot(eigval_sort,'.')
    #    plt.semilogy()
    #    plt.xlabel(r'$N$')
    #    plt.ylabel(r'$N\rm{th}\ \rm{eigenvalue}$')

def inv_cov(cov,rcond=0):
    corr = cov_to_corr(cov)
    stddev = stddev_from_cov(cov)
    invcorr = np.linalg.pinv(corr,rcond=rcond)
    return (invcorr/stddev).T/stddev

def calc_fisher(dfdp, invcov):
    fmatrix = np.dot(dfdp,np.dot(invcov,dfdp.T))
    return fmatrix

def calc_1sigma(fisher):
    return np.sqrt(np.linalg.inv(fisher).diagonal())

def apply_to_dict(function,dictionary,dictionary2=None,**kwargs):
    outs = dict()
    if dictionary2==None:
        for key in dictionary.keys():
            outs[key] = function(dictionary[key],**kwargs)
    else:
        for key in dictionary.keys():
            outs[key] = function(dictionary[key],dictionary2[key],**kwargs)
    return outs

def mean_of_list(ls):
    arr = np.array(ls)
    return np.mean(arr,axis=0)

def stddev_of_list(ls):
    arr = np.array(ls)
    return np.std(arr,axis=0)

def mean_of_dict(dic,key2=None):
    ls = []
    for item in dic.values():
        if type(item)==dict:
            ls.append(item[key2])
        else:
            ls.append(item)
    return mean_of_list(ls)

def stddev_of_dict(dic,key2=None):
    ls = []
    for item in dic.values():
        if type(item)==dict:
            ls.append(item[key2])
        else:
            ls.append(item)
    return stddev_of_list(ls)

def cov2d_from_fisher(fisher, i, j):
    covij = []
    cov = np.linalg.inv(fisher)
    covij = np.array(((cov[i,i],cov[i,j]),(cov[j,i],cov[j,j])))
    return covij

def plot_ellipse(x_cen, y_cen, half_x, half_y, cclockwise_angle_x, ax, color='b', label='', linestyle='-',npoints=200): 
    if ax==None:
        _,ax = plt.subplots()
    #rotation angle in rad
    t = np.linspace(0, 2*np.pi, npoints)
    Ell = np.array([half_x*np.cos(t) , half_y*np.sin(t)])
    R_rot = np.array([[np.cos(cclockwise_angle_x) , -np.sin(cclockwise_angle_x)],\
                      [np.sin(cclockwise_angle_x) , np.cos(cclockwise_angle_x)]])  #2-D rotation matrix
    Ell_rot = np.dot(R_rot,Ell)
    line, = ax.plot(x_cen+Ell_rot[0] , y_cen+Ell_rot[1], linestyle=linestyle, color=color, label=label)
    return line

def ellipse_from_cov(mean, cov, ax=None, color='b', label='', linestyle='-',npoints=200):
    x_cen = mean[0]
    y_cen = mean[1]
    if cov[1][0]==0:
        if np.abs(cov[0][1])>1e-5:
            raise ValueError('covariance matrix must be symmetric')
    elif cov[0][1]/cov[1][0]<1.-1e-5 or cov[0][1]/cov[1][0]>1.+1e-5:
        raise ValueError('covariance matrix must be symmetric')
    sxsx = cov[0][0]
    sysy = cov[1][1]
    sxy = cov[0][1]
    half_x = np.sqrt((sxsx+sysy)/2.+np.sqrt(((sxsx-sysy)**2)/4.+sxy*sxy))
    half_y = np.sqrt((sxsx+sysy)/2.-np.sqrt(((sxsx-sysy)**2)/4.+sxy*sxy))
    if sxsx==sysy:
        cclockwise_angle_x = 0.5*np.arctan(np.inf)
    else:
        cclockwise_angle_x = 0.5*np.arctan(2.*sxy/(sxsx-sysy))
    if sxsx<sysy:
        cclockwise_angle_x += np.pi/2.
    return plot_ellipse(x_cen,y_cen,half_x,half_y,cclockwise_angle_x,ax,color,label,linestyle=linestyle,npoints=npoints)

####################################################################################################################