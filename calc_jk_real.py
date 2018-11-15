from __future__ import division, print_function
import math
from builtins import zip, range
import numpy as np
from scipy.integrate import quad
from fast3tree import fast3tree, get_distance
from scipy import spatial
from halotools.mock_observables import surface_density_helpers

from halotools.mock_observables import void_prob_func
from halotools.mock_observables import wp
from halotools.mock_observables import counts_in_cylinders
from halotools.mock_observables import delta_sigma
from datetime import datetime
import sys
sys.stdout.flush()

def _fast_histogram1d(a, bins):
    """
    Note: `a` is modified in place!
    """
    a.sort()
    return np.ediff1d(np.searchsorted(a, bins))


def _yield_periodic_points(center, dcorner1, dcorner2, box_size):
    cc = np.array(center)
    flag = (cc+dcorner1 < 0).astype(int) - (cc+dcorner2 >= box_size).astype(int)
    cp = cc + flag*box_size
    for j in range(1 << len(cc)):
        for i in range(len(cc)):
            if j >> i & 1 == 0:
                cc[i] = center[i]
            elif flag[i]:
                cc[i] = cp[i]
            else:
                break
        else:
            yield cc.copy()


def _corner_area(x, y):
    a = math.sqrt(1.0-x*x)-y
    b = math.sqrt(1.0-y*y)-x
    theta = math.asin(math.sqrt(a*a+b*b)*0.5)*2.0
    return (a*b + theta - math.sin(theta))*0.5

def _segment_without_corner_area(x, r):
    half_chord = math.sqrt(1.0-x*x)
    return math.acos(x) - x*half_chord \
            - quad(_corner_area, 0, min(half_chord, 1.0/r), (x,))[0]*r

def _overlapping_circular_areas(r):
    if r*r >= 2.0:
        return 1.0
    elif r <= 0:
        return 0.0
    return (math.pi - quad(_segment_without_corner_area, 0, min(1, 1.0/r), \
            (r,))[0]*4.0*r)*r*r

_overlapping_circular_areas_vec = np.vectorize(_overlapping_circular_areas, [float])


def _jackknife_2d_random(rbins, box_size, jackknife_nside):
    side_length = box_size/float(jackknife_nside)
    square_area = 1.0/float(jackknife_nside*jackknife_nside)
    rbins_norm = rbins/side_length
    annulus_areas = np.ediff1d(_overlapping_circular_areas_vec(rbins_norm))
    annulus_areas /= np.ediff1d(rbins_norm*rbins_norm)*math.pi
    return 1.0 - square_area * (2.0 - annulus_areas)


def _get_pairs_max_sphere(points1, points2, max_radius, periodic_box_size=None, indices1=None, indices2=None):

    assert max_radius > 0

    is_periodic = False
    box_size = -1
    if periodic_box_size is not None:
        is_periodic = True
        box_size = float(periodic_box_size)
        assert box_size > 0
        if max_radius*2.0 > box_size:
            print('[Warning] box too small!')

    with fast3tree(points2, indices2) as tree:
        del points2, indices2
        if box_size > 0:
            tree.set_boundaries(0, box_size)
        iter_points1 = (enumerate(points1) if indices1 is None else zip(indices1, points1))
        del points1, indices1
        for i, p in iter_points1:
            indices, pos = tree.query_radius(p, max_radius, is_periodic, 'both')
            if len(indices):
                pos = get_distance(p, pos, box_size)
                yield i, indices, pos


def _get_pairs_max_box(points1, points2, max_distances, periodic_box_size=None, indices1=None, indices2=None):

    max_distances = np.asanyarray(max_distances)
    max_distances_neg = max_distances * (-1.0)
    assert np.all(max_distances - max_distances_neg > 0)

    is_periodic = False
    if periodic_box_size is not None:
        is_periodic = True
        box_size = float(periodic_box_size)
        assert box_size > 0
        assert np.all(max_distances - max_distances_neg < box_size)

    with fast3tree(points2, indices2) as tree:
        del points2, indices2
        iter_points1 = (enumerate(points1) if indices1 is None else zip(indices1, points1))
        del points1, indices1
        for i, p in iter_points1:
            p_iter = _yield_periodic_points(p, max_distances_neg, max_distances, box_size) if is_periodic else [p]
            for pp in p_iter:
                indices, pos = tree.query_box(pp + max_distances_neg, pp + max_distances, output='both')
                if len(indices):
                    pos -= pp
                    yield i, indices, pos


def _reduce_2d_distance_square(pos):
    """
    Note: `pos` is modified in place.
    """
    pos = pos[:,:2]
    pos *= pos
    pos[:,0] += pos[:,1]
    return pos[:,0]


def get_pairs(points1, points2, max_radius, max_dz=None, periodic_box_size=None,
              id1_label='id1', id2_label='id2', dist_label='d', can_swap_points=True,
              indices1=None, indices2=None, wrapper_function=None):
    """
    Identify all pairs within a sphere or a cylinder.

    Parameters
    ----------
    points1 : array_like
        Must be a 2-d array whose last dimension is 3 (i.e. has 3 columns)
    points2 : array_like or None
        Must be a 2-d array whose last dimension is 3 (i.e. has 3 columns)
        If set to None, do auto-correlation
    max_radius : float
        Find pairs within this radius
    max_dz : float or None
    periodic_box_size : float or None
    """

    if points2 is not None and can_swap_points and len(points2) > len(points1):
        points1, points2 = points2, points1
        id1_label, id2_label = id2_label, id1_label


    is_autocorrelation = False
    if points2 is None:
        points2 = points1
        if indices2 is None and indices1 is not None:
            indices2 = indices1
        is_autocorrelation = True


    if max_dz is None:
        function_to_call = _get_pairs_max_sphere
        max_distances = max_radius
        if wrapper_function is None:
            def wrapper_function(iter_input):
                for i, j_arr, d_arr in iter_input:
                    if is_autocorrelation:
                        flag = (j_arr > i)
                        j_arr = j_arr[flag]
                        d_arr = d_arr[flag]
                        del flag
                    for j, d in zip(j_arr, d_arr):
                        yield i, j, d
    else:
        function_to_call = _get_pairs_max_box
        max_distances = np.array([max_radius, max_radius, max_dz])
        if wrapper_function is None:
            def wrapper_function(iter_input):
                for i, j_arr, d_arr in iter_input:
                    d_arr = _reduce_2d_distance_square(d_arr)
                    np.sqrt(d_arr, out=d_arr)
                    flag = (d_arr < max_radius)
                    if is_autocorrelation:
                        flag &= (j_arr > i)
                    j_arr = j_arr[flag]
                    d_arr = d_arr[flag]
                    del flag
                    for j, d in zip(j_arr, d_arr):
                        yield i, j, d

    iterator = wrapper_function(function_to_call(points1, points2, max_distances,
            periodic_box_size, indices1, indices2))
    del points1, points2, indices1, indices2

    output_dtype = np.dtype([(id1_label, np.int), (id2_label, np.int), (dist_label, np.float)])
    return np.fromiter(iterator, output_dtype)


def _check_points(points):
    points = np.asarray(points)
    s = points.shape
    if len(s) != 2 or s[1] != 3:
        raise ValueError('`points` must be a 2-d array with last dim=3')
    return points, s[0]


def _check_rbins(rbins):
    rbins = np.asarray(rbins)
    assert (np.ediff1d(rbins) > 0).all(), '`rbins` must be an increasing array'
    assert rbins[0] >= 0, '`rbins must be all positive'
    return rbins


def get_random_pair_counts(n_points, box_size, rbins, zmax=None):
    rbins = np.asarray(rbins)
    assert (np.ediff1d(rbins) > 0).all()

    density = n_points * n_points / (box_size**3.0)

    if zmax is None:
        volume = np.ediff1d(rbins**3.0) * (np.pi*4/3)
    else:
        volume = np.ediff1d(rbins*rbins) * np.pi * zmax * 2.0

    return density * volume


def get_jackknife_ids(points, box_size, jackknife_nside):
    jackknife_nside = int(jackknife_nside)
    jack_ids  = np.floor(np.remainder(points[:,0], box_size)\
            / box_size * jackknife_nside).astype(int)
    jack_ids += np.floor(np.remainder(points[:,1], box_size)\
            / box_size * jackknife_nside).astype(int) * jackknife_nside
    return(jack_ids)


def projected_correlation(points, rbins, zmax, box_size, jackknife_nside=0, **kwargs):
    """
    Calculate the projected correlation function wp(rp) and its covariance
    matrix for a periodic box, with the plane-parallel approximation and
    the Jackknife method.

    Parameters
    ----------
    points : array_like
        Must be a 2-d array whose last dimension is 3 (i.e. has 3 columns)
        The last column will be used as the redshift distance.
    rbins : array_like
        A 1-d array that has the edges of the rp bins. Must be sorted.
    zmax : float
        The integral of \pi goes from -zmax to zmax (redshift distance).
    box_size : float
        The side length of the periodic box.
    jackknife_nside : int, optional (Default: 0)
        If <= 1 , it will not do Jackknife.

    Returns
    -------
    wp : ndarray
        A 1-d array that has wp. The length of this returned array would be
        len(rbins) - 1.
    wp_cov : ndarray (returned if jackknife_nside > 1)
        The len(wp) by len(wp) covariance matrix of wp.
    """

    if 'bias_correction' in kwargs:
        print('`bias_correction` is obsolete. No correction is applied.')

    points, N = _check_points(points)
    rbins = _check_rbins(rbins)
    rbins_sq = rbins*rbins

    max_distances = np.array([rbins[-1], rbins[-1], zmax])
    pairs_rand = get_random_pair_counts(N, box_size, rbins, zmax)
    jackknife_nside = int(jackknife_nside or 0)

    if jackknife_nside <= 1: #no jackknife
        pairs = np.zeros(len(rbins)-1, dtype=np.int)
        for _, _, pos in _get_pairs_max_box(points, points, max_distances, periodic_box_size=box_size):
            pairs += _fast_histogram1d(_reduce_2d_distance_square(pos), rbins_sq)
        return (pairs/pairs_rand - 1.0) * zmax*2.0

    else: #do jackknife
        jack_ids = get_jackknife_ids(points, box_size, jackknife_nside)
        n_jack = jackknife_nside*jackknife_nside
        jack_counts = np.bincount(jack_ids, minlength=n_jack)
        jack_pairs_rand_scale = (N-jack_counts)*(N-jack_counts)/float(N*N)
        del jack_counts

        pairs = np.zeros((n_jack, len(rbins)-1), dtype=np.int)
        auto_pairs = np.zeros_like(pairs)

        for i, j, pos in _get_pairs_max_box(points, points, max_distances, periodic_box_size=box_size):
            jid = jack_ids[i]
            pos = _reduce_2d_distance_square(pos)
            pos_auto = pos[jack_ids[j] == jid]
            pairs[jid] += _fast_histogram1d(pos, rbins_sq)
            auto_pairs[jid] += _fast_histogram1d(pos_auto, rbins_sq)
        del i, j, pos, jack_ids, points

        pairs_sum = pairs.sum(axis=0)
        wp_full = (pairs_sum/pairs_rand - 1.0) * zmax*2.0

        pairs = pairs_sum - pairs*2 + auto_pairs
        wp_jack = (pairs / pairs_rand / jack_pairs_rand_scale[:, np.newaxis] \
                / _jackknife_2d_random(rbins, box_size, jackknife_nside) \
                - 1.0) * zmax*2.0
        wp_cov = np.cov(wp_jack, rowvar=0, bias=1)*(n_jack-1)

        return wp_full, wp_cov


def correlation3d(points, rbins, box_size):
    """
    Calculate the 3D correlation function xi(r) for a periodic box.

    Parameters
    ----------
    points : array_like
        Must be a 2-d array whose last dimension is 3 (i.e. has 3 columns).
    rbins : array_like
        A 1-d array that has the edges of the rp bins. Must be sorted.
    box_size : float
        The side length of the periodic box.

    Returns
    -------
    xi : ndarray
        A 1-d array that has wp. The length of this returned array would be
        len(rbins) - 1.
    """

    points, N = _check_points(points)
    rbins = _check_rbins(rbins)

    pairs = np.zeros(len(rbins)-1, dtype=np.int)

    for _, _, d in _get_pairs_max_sphere(points, points, rbins[-1], periodic_box_size=box_size):
        pairs += _fast_histogram1d(d, rbins)

    return pairs / get_random_pair_counts(N, box_size, rbins) - 1.0


#######################################################
#######################################################
def apply_periodic(points,period):
    x,y,z = points.T
    length = points.shape[0]
    extended_box = np.zeros((length*27,3))
    xp = np.array((2,5,8,11,14,17,20,23,26))
    xo = xp-1
    xm = xp-2
    yp = np.array((6,7,8,15,16,17,24,25,26))
    yo = yp-3
    ym = yp-6
    zp = np.array((18,19,20,21,22,23,24,25,26))
    zo = zp-9
    zm = zp-18
    for i in xp:
        extended_box[i*length:(i+1)*length,0] = x+period
    for i in xo:
        extended_box[i*length:(i+1)*length,0] = x
    for i in xm:
        extended_box[i*length:(i+1)*length,0] = x-period
    for i in yp:
        extended_box[i*length:(i+1)*length,1] = y+period
    for i in yo:
        extended_box[i*length:(i+1)*length,1] = y
    for i in ym:
        extended_box[i*length:(i+1)*length,1] = y-period
    for i in zp:
        extended_box[i*length:(i+1)*length,2] = z+period
    for i in zo:
        extended_box[i*length:(i+1)*length,2] = z
    for i in zm:
        extended_box[i*length:(i+1)*length,2] = z-period
    return extended_box

def apply_periodic_xy(points,period):
    x,y,z = points.T
    length = points.shape[0]
    extended_box = np.zeros((length*9,3))
    xp = np.array((2,5,8))
    xo = xp-1
    xm = xp-2
    yp = np.array((6,7,8))
    yo = yp-3
    ym = yp-6
    for i in xp:
        extended_box[i*length:(i+1)*length,0] = x+period
    for i in xo:
        extended_box[i*length:(i+1)*length,0] = x
    for i in xm:
        extended_box[i*length:(i+1)*length,0] = x-period
    for i in yp:
        extended_box[i*length:(i+1)*length,1] = y+period
    for i in yo:
        extended_box[i*length:(i+1)*length,1] = y
    for i in ym:
        extended_box[i*length:(i+1)*length,1] = y-period
    return extended_box

def voidsize(centers, points):
    voidsizes = np.zeros(len(centers))
    
    tree = spatial.cKDTree(points)

    voidsizes = tree.query(centers)[0]
    
    return voidsizes

def cicyl(points, periopoints, radius, halflen):
    with fast3tree(periopoints[:,:2]) as tree_xy:

        cnts = []
        for pt in points:
            idx_xy = tree_xy.query_radius(pt[:2], radius)
            cnt = -1
            for j in idx_xy:
                if np.abs(pt[2]-periopoints[j,2])<halflen:
                    cnt += 1
            cnts.append(cnt)
    return np.array(cnts)

def Ncyl(gals, perioptcl, r_ds):
    with fast3tree(perioptcl[:,:2]) as tree_xy:
        N_cyl = []
        for pt in gals:
            n_cyl = []
            for r in r_ds:
                n_cyl.append(len(tree_xy.query_radius(pt[:2], r)))
            N_cyl.append(n_cyl)
    return np.array(N_cyl)   ##len(gals)*len(r)

def calc_jk_real(points, box_size, wbool=1, dbool=1, vbool=1, cbool=1, abool=1, rbool=1, jackknife_nside=0,\
                 rbins_wp=None, zmax=None,\
                 r_vpf=None, vpf_cen=None,\
                 galpos_non_rsd=None, ptclpos=None, ptcl_mass=None, ptcl_per_dim=None, r_ds=None,\
                 cyl_r1=None, cyl_halflen=None, cyl_bin=None,\
                 cyl_r2=None,  ann_bin=None,\
                 rat_bin=None):
    """
    Calculate the observables and their covariance
    matrix for a periodic box, with the plane-parallel approximation and
    the Jackknife method.

    Parameters
    ----------
    points: array_like
        Must be a 2-d array whose last dimension is 3 (i.e. has 3 columns)
        The last column will be used as the redshift distance.
    box_size: float
        The side length of the periodic box.
    w/d/v/c/a/rbool: integer or boolean
        Whether to calculate wp, delta sigma, vpf, cic, cia, ratio.
    jackknife_nside: int, optional (Default: 0)
        If <= 1 , it will not do Jackknife. Else, Ncell = nside^2
    rbins_wp: array_like
        A 1-d array that has the edges of the rp bins for wp. Must be sorted.
    zmax: float
        The integral of \pi goes from -zmax to zmax (redshift distance).
    galpos_non_rsd: array_like
        Galaxy positions without redshift space distortion, for delta sigma calculation.
        Must be a 2-d array whose last dimension is 3 (i.e. has 3 columns)
    ptclpos: array_like
        Dark matter particle positions.
        Must be a 2-d array whose last dimension is 3 (i.e. has 3 columns)
    ptcl_mass: float
        Mass of each dark matter particle.
    r_ds: array_like
        A 1-d array that has the edges of the rp bins for delta sigma. Must be sorted.
    r_vpf: array_like
        1-d array of vpf sphere radii
    vpf_cen: array_like
        Pre-stored random centers, must be a 2-d array whose last dimension is 3 (i.e. has 3 columns)
    cyl_r1: float
        The radius of the first cylinder to do count in
    cyl_halflen: float
        The half length of cylinders
    cyl_bin: array_like
        Must be a 1-D array, the bins at which to sum counts-in-cylinders histogram.
    cyl_r2: float
        The radius of the second cylinder to do count in
    ann_bin: array_like
        Must be a 1-D array, the bins at which to sum counts-in-annuli histogram.
    rat_bin: int
        Must be a 1-D array, the bin edges for which to compute ratio histogram.

    Returns
    -------
    func_full: ndarray (returned if jackknife_nside <= 1)
        A 1-d array that has the calculated observables among [number density of galaxies, wp, delta sigma, vpf, cic, cia, 
        cic ratio], concatenated.
    func_cov: ndarray (returned if jackknife_nside > 1)
        The jackknife covariance matrix of func.
    """

    points, N = _check_points(points)
    jackknife_nside = int(jackknife_nside or 0)
    
                          
    if jackknife_nside <= 1: #no jackknife
        func_full = []
        if wbool:
            func_full.append(np.array((N/float(box_size**3),)))
            func_full.append(wp(points, rbins_wp, zmax, period=box_size))
        if dbool:
            total_num_ptcls_in_snapshot = ptcl_per_dim**3
            downsampling_factor = total_num_ptcls_in_snapshot/float(len(ptclpos))
            func_full.append(delta_sigma(galpos_non_rsd, ptclpos, particle_masses=ptcl_mass, downsampling_factor=downsampling_factor,\
                      rp_bins=r_ds, period=box_size)[1]/1e12)
        if vbool:                   
            func_full.append(void_prob_func(points, r_vpf, random_sphere_centers=vpf_cen, period=box_size))
        if cbool:
            cic1_h = counts_in_cylinders(points, points, cyl_r1, cyl_halflen, period=box_size)
            func_full.append(np.add.reduceat(np.bincount((cic1_h-1), minlength=300)[:300]/float(N),cyl_bin))
        if abool:
            cic2_h = counts_in_cylinders(points, points, cyl_r2, cyl_halflen, period=box_size)
            func_full.append(np.add.reduceat(np.bincount((cic2_h-cic1_h), minlength=500)[:500]/float(N),ann_bin))
        if rbool:
            ratio_h = ((cic1_h-1)/(cic2_h-1)).astype('float')
            ratio_h[~np.isfinite(ratio_h)] = 0.
            Pratio_h = np.histogram(ratio_h,bins=rat_bin)[0]/float(N)
            func_full.append(Pratio_h)
    
        func_full = np.array([item for sublist in func_full for item in sublist])
        return func_full

    else: #do jackknife
        jack_ids = get_jackknife_ids(points, box_size, jackknife_nside)
        n_jack = jackknife_nside*jackknife_nside
        func_jk = []
        if wbool:
            rbins_wp = _check_rbins(rbins_wp)
            rbins_sq = rbins_wp*rbins_wp
            max_distances = np.array([rbins_wp[-1], rbins_wp[-1], zmax])
            pairs_rand = get_random_pair_counts(N, box_size, rbins_wp, zmax)
            jack_counts = np.bincount(jack_ids, minlength=n_jack)    
            ngal_jack = (N-jack_counts)/(box_size**3*(n_jack-1)/n_jack)
            jack_pairs_rand_scale = (N-jack_counts)*(N-jack_counts)/float(N*N)
            del jack_counts
            pairs = np.zeros((n_jack, len(rbins_wp)-1), dtype=np.int)
            auto_pairs = np.zeros_like(pairs)
            for i, j, pos in _get_pairs_max_box(points, points, max_distances, periodic_box_size=box_size):
                jid = jack_ids[i]
                pos = _reduce_2d_distance_square(pos)
                pos_auto = pos[jack_ids[j] == jid]
                pairs[jid] += _fast_histogram1d(pos, rbins_sq)
                auto_pairs[jid] += _fast_histogram1d(pos_auto, rbins_sq)
            pairs_sum = pairs.sum(axis=0)
            pairs = pairs_sum - pairs*2 + auto_pairs
            wp_jack = (pairs / pairs_rand / jack_pairs_rand_scale[:, np.newaxis] \
                    / _jackknife_2d_random(rbins_wp, box_size, jackknife_nside) \
                    - 1.0) * zmax*2.0
            func_jk.append(ngal_jack)
            func_jk.append(wp_jack.T)
        if dbool:
            perioptclpos = apply_periodic_xy(ptclpos, box_size)
            N_cyl = Ncyl(galpos_non_rsd, perioptclpos, r_ds)
            rp_mids = surface_density_helpers.annular_area_weighted_midpoints(r_ds)
            ggl_jack = np.zeros((n_jack,len(rp_mids)))
            for i in range(n_jack):
                mask = jack_ids!=i
                N_cyl_i = N_cyl[mask]
                Sigma_cyl_i = ptcl_mass*np.mean(np.array(N_cyl_i),axis=0)/np.pi/(r_ds**2)
                Sigma_annuli_i = ptcl_mass*np.diff(np.mean(np.array(N_cyl_i),axis=0))/np.pi/np.diff(r_ds**2)
                Sigma_interp_mid_i = surface_density_helpers.log_interpolation_with_inner_zero_masking(Sigma_cyl_i,r_ds, rp_mids)
                ggl_jack[i] = (Sigma_interp_mid_i-Sigma_annuli_i)/1e12
            func_jk.append(ggl_jack.T)
        if vbool or cbool or wbool or abool:                      
            periopos = apply_periodic(points, box_size)
        if vbool:
            center_jkids = get_jackknife_ids(vpf_cen, box_size, jackknife_nside)
            vs = voidsize(vpf_cen, periopos)
            vpf_jack = np.zeros((n_jack,len(r_vpf)))
            for i in range(n_jack):
                mask = center_jkids!=i
                for j,r in enumerate(r_vpf):
                    vpf_jack[i,j] = np.sum(vs[mask]>r)/float(mask.sum())
            func_jk.append(vpf_jack.T)
        if cbool or abool or rbool:    
            cic1 = cicyl(points, periopos, cyl_r1, cyl_halflen)
        if abool or rbool:
            cic2 = cicyl(points, periopos, cyl_r2, cyl_halflen)
        if cbool:
            Pcic_jack = np.zeros((n_jack,len(cyl_bin)))
            for i in range(n_jack):
                mask = jack_ids!=i
                Pcic_jack[i] = np.add.reduceat(np.bincount(cic1[mask],\
                                                        minlength=300)[:300]/float(len(points[mask])),cyl_bin)
            func_jk.append(Pcic_jack.T)
        if abool:
            Pcia_jack = np.zeros((n_jack,len(ann_bin)))
            for i in range(n_jack):
                mask = jack_ids!=i
                Pcia_jack[i] = np.add.reduceat(np.bincount((cic2-cic1)[mask], minlength=500)[:500]/float(len(points[mask])),ann_bin)
            func_jk.append(Pcia_jack.T)
        if rbool:
            ratio = (cic1/cic2).astype('float')
            ratio[~np.isfinite(ratio)] = 0.
            Pratio_jack = np.zeros((n_jack,len(rat_bin)-1))
            for i in range(n_jack):
                mask = jack_ids!=i
                Pratio_jack[i] = np.histogram(ratio[mask],bins=rat_bin)[0]/float(len(points[mask]))
            func_jk.append(Pratio_jack.T)
        func_cov = np.cov(np.vstack(func_jk), bias=1)*(n_jack-1)

        return func_cov, np.vstack(func_jk)
