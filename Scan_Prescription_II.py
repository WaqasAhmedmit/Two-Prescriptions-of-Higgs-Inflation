"""STANDALONE SCRIPT - Prescription II nonminimal Higgs inflation.

Run: python Scan_Prescription_II.py --samples 300000 --workers 4
Test: python Scan_Prescription_II.py --samples 30 --grid 400 --output test_II
Run separately from the other file: no shared solver module is required.
"""

"""All-detected-branch slow-roll scan: two one-loop nonminimal Higgs prescriptions.

This searches *both* rolling directions for all resolved end-of-inflation
crossings in a finite user-defined field interval. It is not a proof that no
ultra-narrow branches exist: increase n_grid and x_max for convergence checks.

x=phi/Mp, U=V/(lambda*Mp**4), beta=kappa/lambda.
V_I=lambda*Mp^4*x^4*[1+24 beta log(x/sqrt(F))]/(24 F**2)
V_II=lambda*Mp^4*x^4*[1+24 beta log(x)]/(24 F**2)
F=1+xi*x*x; (d sigma/d phi)**2=(1+xi*(1+6xi)*x*x)/F**2.
"""
import numpy as np
from scipy.integrate import solve_ivp,quad
from scipy.optimize import brentq

MP_GEV=2.435515e18
AS_TARGET=2.43e-9
X_MIN=1e-10
X_MAX=200.


def ingredients(x,xi,beta,prescription):
    x=np.asarray(x,dtype=float)
    F=1+xi*x*x
    if np.any(x<=0) or np.any(F<=0): raise ValueError('invalid x or F')
    h=2*xi*x/F
    h1=2*xi/F-h*h
    h2=-12*xi*xi*x/F**2+2*h**3 # derivative of h1 = - 3*(2xi)*(2xi x)/F^2 + 2h^3
    c=.5 if int(prescription)==1 else 0.
    logarg=np.log(x)-c*np.log(F)
    b=1+24*beta*logarg
    b1=24*beta*(1/x-c*h)
    b2=24*beta*(-1/x**2-c*h1)
    b3=24*beta*(2/x**3-c*h2)
    t=xi*(1+6*xi)
    K=1+t*x*x
    a=np.sqrt(K)/F
    q=t*x/K-h
    q1=t*(1-t*x*x)/K**2-h1
    with np.errstate(divide='ignore',invalid='ignore',over='ignore',under='ignore'):
        B=b1/b
        E=4/x-2*h+B
        E1=-4/x**2-2*h1+b2/b-B*B
        E2=8/x**3-2*h2+b3/b-3*b2*b1/b**2+2*B**3
        u=x**4*b/(24*F*F)
        eps=.5*(E/a)**2
        eta=(E1+E*E-q*E)/a**2
        chi3=(E2+3*E*E1+E**3-3*q*(E1+E*E)+(2*q*q-q1)*E)/a**3
        zeta2=(E/a)*chi3
        alpha=16*eps*eta-24*eps**2-2*zeta2
    return dict(F=F,b=b,logarg=logarg,u=u,a=a,E=E,eps=eps,eta=eta,zeta2=zeta2,alpha=alpha)


def scalar(x,xi,beta,prescription):
    return {k:float(v) for k,v in ingredients(float(x),xi,beta,prescription).items()}


def list_ends(xi,beta,prescription,x_min=X_MIN,x_max=X_MAX,n_grid=600):
    """Locate end crossings where max(eps,|eta|) rises to unity downhill.

    Uphill/rolling-to-smaller-x (positive slope): high-x slow side, low-x end.
    Rolling-to-larger-x (negative slope): low-x slow side, high-x end.
    """
    if x_min<=0 or x_min>=x_max: raise ValueError('bad search window')
    xs=np.geomspace(x_min,x_max,n_grid)
    v=ingredients(xs,xi,beta,prescription)
    g=np.maximum(v['eps'],np.abs(v['eta']))-1.
    good=np.isfinite(g)&np.isfinite(v['E'])&(np.abs(v['b'])>1e-13)&(np.abs(v['u'])>0)
    ends=[]
    # Vectorized candidate location is much faster for large (50k) scans.
    rising=(g[:-1]>0)&(g[1:]<0)&(v['E'][1:]>0)
    falling=(g[:-1]<0)&(g[1:]>0)&(v['E'][:-1]<0)
    idx=np.flatnonzero(good[:-1]&good[1:]&(rising|falling))
    for i in idx:
        direction=1 if rising[i] else -1
        def fun(z):
            w=scalar(np.exp(z),xi,beta,prescription)
            return max(w['eps'],abs(w['eta']))-1
        try:
            z=brentq(fun,np.log(xs[i]),np.log(xs[i+1]),xtol=1e-13)
            xe=float(np.exp(z)); w=scalar(xe,xi,beta,prescription)
            if not np.isfinite(w['eps']+w['eta']) or abs(w['b'])<1e-13: continue
            end_by='epsilon' if w['eps']>=abs(w['eta']) else 'abs_eta'
            # Guard against duplicate numerical crossings on adjacent cells.
            if any(abs(np.log(e['x_end']/xe)) < 5e-8 for e in ends): continue
            ends.append(dict(x_end=xe,roll_direction=direction,end_by=end_by,eps_end=w['eps'],eta_end=w['eta']))
        except (ValueError,FloatingPointError,OverflowError): continue
    ends.sort(key=lambda item:item['x_end'])
    for i,e in enumerate(ends,1): e['end_id']=i
    return ends


def solve_all_branches(xi,beta,N,prescription,*,x_min=X_MIN,x_max=X_MAX,n_grid=600,check_path=True):
    """One record per detected end branch, or one labelled failure record.

    Normalization lambda from As is determined for every trajectory.
    Both signs of renormalized lambda are stored. Negative lambda demands
    separate global-stability checks; it is NOT considered automatically viable.
    No observational cuts are used to discard solutions.
    """
    base=dict(xi=float(xi),beta=float(beta),N=float(N),prescription=int(prescription),
              x_min=float(x_min),x_max=float(x_max),success=False,status='unknown')
    try: ends=list_ends(xi,beta,prescription,x_min,x_max,n_grid)
    except (ValueError,OverflowError,FloatingPointError):
        return [dict(base,status='end_search_error',n_end_candidates=0)]
    if not ends: return [dict(base,status='no_resolved_end_crossing',n_end_candidates=0)]
    rows=[]
    for e in ends:
        row=dict(base,**e,n_end_candidates=len(ends))
        xe=e['x_end']; direction=e['roll_direction']
        interior_bound=x_max if direction==1 else x_min
        # Integrate *backwards* in efolds, dx/dN = dlnV/dx /(d sigma/dphi)^2.
        rhs_calls=[0]
        def rhs(n,y):
            rhs_calls[0]+=1
            if rhs_calls[0]>2000: raise RuntimeError('ODE near-critical work budget exceeded; high-precision retry needed')
            x=float(y[0])
            if x<=x_min or x>=x_max: return [0.]
            w=scalar(x,xi,beta,prescription)
            if not np.isfinite(w['E']) or abs(w['b'])<1e-13: return [0.]
            return [w['E']/(w['a']*w['a'])]
        def boundary(n,y): return direction*(interior_bound-float(y[0]))
        boundary.terminal=True; boundary.direction=-1
        try:
            sol=solve_ivp(rhs,(0,float(N)),[xe],method='DOP853',rtol=3e-9,atol=1e-11,
                          max_step=1.5,events=boundary,dense_output=bool(check_path))
        except (ValueError,OverflowError,ZeroDivisionError,RuntimeError) as ex:
            rows.append(dict(row,status='integration_exception',details=str(ex)[:120]));continue
        if not sol.success:
            rows.append(dict(row,status='integration_failure')); continue
        if sol.t[-1]<N-1e-6:
            rows.append(dict(row,status='horizon_outside_domain')); continue
        xs=float(sol.y[0,-1]); w=scalar(xs,xi,beta,prescription)
        if (not np.isfinite(w['u']+w['eps']+w['eta']+w['alpha']) or abs(w['u'])==0 or abs(w['b'])<1e-13
              or w['eps']>=1 or abs(w['eta'])>=1 or direction*w['E']<=0):
            rows.append(dict(row,status='invalid_horizon')); continue
        if check_path:
            trajectory=sol.sol(np.linspace(0,N,65))[0]
            v=ingredients(trajectory,xi,beta,prescription)
            good=(np.isfinite(trajectory)&(np.sign(v['b'])==np.sign(w['b']))&(np.abs(v['u'])>0)&(direction*v['E']>0))
            good[0]=True
            if (not np.all(good) or not np.all(v['eps'][1:]<1+1e-6)
                or not np.all(np.abs(v['eta'][1:])<1+1e-6)):
                rows.append(dict(row,status='invalid_trajectory')); continue
        lam=24*np.pi**2*AS_TARGET*w['eps']/w['u']
        if not np.isfinite(lam) or abs(lam)==0:
            rows.append(dict(row,status='bad_As_normalization'));continue
        logg=ingredients(np.geomspace(min(xe,xs),max(xe,xs),50),xi,beta,prescription)['logarg']
        loop_max=float(np.max(np.abs(24*beta*logg)))
        rows.append(dict(row,success=True,status='ok',x_star=xs,
            phi_end_GeV=xe*MP_GEV,phi_star_GeV=xs*MP_GEV,
            ns=1-6*w['eps']+2*w['eta'],r=16*w['eps'],alpha_s=w['alpha'],
            eps_star=w['eps'],eta_star=w['eta'],zeta2_star=w['zeta2'],
            lambda_=lam,kappa=beta*lam,As=AS_TARGET,lambda_sign=int(np.sign(lam)),
            lambda_perturbative=bool(abs(lam)<1),loop_fraction_max=loop_max,
            near_stationary_precision=bool(w['eps']<1e-18),
            loop_control=bool(loop_max<1),Vstar_Mp4=lam*w['u'],
            paper_plot_window=bool((.955<1-6*w['eps']+2*w['eta']<.99) and 16*w['eps']<.055),
            exploratory_ACT_screen=bool((.967<=1-6*w['eps']+2*w['eta']<=.981) and 16*w['eps']<.036),
            strict_phi_window=bool(min(xe,xs)>=1e16/MP_GEV and max(xe,xs)<=1.0)))
    return rows


def check_integral(record):
    """Independent e-fold integral, sign-correct for either rolling direction."""
    if not record['success']: return np.nan
    xi,beta,p=record['xi'],record['beta'],record['prescription']
    xe,star=record['x_end'],record['x_star']
    f=lambda x:(lambda w:w['a']**2/w['E'])(scalar(x,xi,beta,p))
    return quad(f,xe,star,epsabs=1e-9,epsrel=1e-8,limit=250)[0]



# =====================================================================
# SELF-CONTAINED SINGLE-PRESCRIPTION SCANNER AND PLOTTER
# =====================================================================
import argparse
import json
import os
import warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Reliable for terminal / Windows; images saved as PNG and PDF.
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

PRESCRIPTION = 2
LABEL = 'II'
SEED = 20261107
ACT_CENTER, ACT_HALF_WIDTH = 0.9743, 0.0068
SPT_CENTER, SPT_HALF_WIDTH = 0.9684, 0.006
R_LIMIT = 0.036

# Numeric output fields. "lambda" is obtained by scalar amplitude normalization,
# NOT sampled independently. "kappa" = beta * lambda. All units are documented.
FIELDS = [
    'sample_id', 'prescription', 'xi', 'beta', 'N',
    'xi_stratum', 'beta_stratum', 'N_stratum',
    'end_id', 'n_end_candidates', 'roll_direction', 'end_epsilon',
    'x_end', 'x_star', 'phi_end_GeV', 'phi_star_GeV',
    'eps_end', 'eta_end', 'eps_star', 'eta_star', 'zeta2_star',
    'ns', 'r', 'alpha_s', 'As', 'lambda', 'kappa', 'Vstar_Mp4',
    'loop_fraction_max', 'lambda_perturbative', 'loop_control',
    'near_stationary_precision', 'strict_phi_window', 'paper_plot_window',
    'theory_screen', 'ACT95', 'SPT95', 'both95', 'negative_lambda_flag'
]


def make_design(n, seed):
    """Matching input design for I/II when same --samples and --seed are used.

    Stratified log xi range 1e-6 .. 1e5; signed beta from near-zero to +/-0.05;
    N = 50 / 55 / 60 / uniform 50..60. Sampling is by beta=kappa/lambda;
    physical kappa is saved AFTER normalization, not directly sampled.
    """
    rng = np.random.default_rng(seed)
    z = rng.random((n, 3))
    xm = np.arange(n) % 5
    rng.shuffle(xm)
    lo = np.array([-6., -4., -2., 0., 2.])[xm]
    hi = np.array([-4., -2., 0., 2., 5.])[xm]
    xi = 10.**(lo + (hi-lo)*z[:, 0])

    bm = np.arange(n) % 6
    rng.shuffle(bm)
    beta = np.empty(n)
    for k in range(6):
        sel = (bm == k)
        zz = z[sel, 1]
        if k == 0: beta[sel] = 10.**(-10 + 8.7*zz)
        if k == 1: beta[sel] = -10.**(-10 + 8.7*zz)
        if k == 2: beta[sel] = 0.05*zz
        if k == 3: beta[sel] = -0.05*zz
        if k == 4: beta[sel] = 10.**(-3.5 + 2*zz)
        if k == 5: beta[sel] = -10.**(-3.5 + 2*zz)

    nm = np.arange(n) % 4
    rng.shuffle(nm)
    N = np.where(nm==0, 50., np.where(nm==1, 55.,
                 np.where(nm==2, 60., 50. + 10.*z[:, 2])))
    return pd.DataFrame({'sample_id': np.arange(n, dtype=np.int64),
                         'xi': xi, 'beta': beta, 'N': N,
                         'xi_stratum': xm, 'beta_stratum': bm,
                         'N_stratum': nm})


def compute_one(payload):
    """Compute all numerically resolved end/trajectory branches at one input."""
    record, n_grid, path_check = payload
    sid, xi, beta, N, xs, bs, ns = record
    try:
        results = solve_all_branches(float(xi), float(beta), float(N),
                                    PRESCRIPTION, n_grid=n_grid,
                                    check_path=path_check)
    except Exception as exc:
        results = [{'success': False, 'status': 'uncaught_worker_exception',
                    'details': f'{type(exc).__name__}: {str(exc)[:160]}'}]
    for r in results:
        r.update(sample_id=int(sid), prescription=PRESCRIPTION,
                 xi=float(xi), beta=float(beta), N=float(N),
                 xi_stratum=int(xs), beta_stratum=int(bs), N_stratum=int(ns))
    return results


def bool_series(series):
    """Safe even if old files contain mixed objects; no pandas downcast warning."""
    return series.astype('string').str.strip().str.lower().isin(('true','1','yes'))


def classify(df):
    d = df.copy()
    if 'lambda_' in d.columns: d = d.rename(columns={'lambda_': 'lambda'})
    for key in ['ns', 'r', 'N', 'lambda', 'kappa', 'xi', 'beta']:
        if key not in d: d[key] = np.nan
        d[key] = pd.to_numeric(d[key], errors='coerce')
    for key in ['success','loop_control','near_stationary_precision',
                'lambda_perturbative','strict_phi_window','paper_plot_window']:
        if key not in d: d[key] = False
        d[key] = bool_series(d[key])
    d['end_epsilon'] = d['end_by'].astype('string').eq('epsilon') if 'end_by' in d else False
    d['negative_lambda_flag'] = d['lambda'].lt(0)
    d['theory_screen'] = (d['success'] & d['lambda'].gt(0) & d['lambda'].lt(1)
                          & d['loop_control'] & ~d['near_stationary_precision'])
    d['ACT95'] = (d['theory_screen'] & (d['ns']-ACT_CENTER).abs().le(ACT_HALF_WIDTH)
                  & d['r'].lt(R_LIMIT))
    d['SPT95'] = (d['theory_screen'] & (d['ns']-SPT_CENTER).abs().le(SPT_HALF_WIDTH)
                  & d['r'].lt(R_LIMIT))
    d['both95'] = d['ACT95'] & d['SPT95']
    return d


def run_scan(n, seed, root, n_grid, chunk_size, workers, path_check):
    root.mkdir(parents=True, exist_ok=True)
    chunks = root / 'chunks'
    chunks.mkdir(exist_ok=True)
    config = dict(prescription=PRESCRIPTION, n=n, seed=seed, n_grid=n_grid,
                  chunk_size=chunk_size, path_check=path_check,
                  x_min=X_MIN, x_max=X_MAX, As=AS_TARGET,
                  version='standalone-prescription-1.0')
    cpath = root / 'scan_config.json'
    if cpath.exists():
        old = json.loads(cpath.read_text(encoding='utf-8'))
        if old != config:
            raise RuntimeError('Existing folder uses different scan settings. '
                               'Use an alternative --output folder or restore the previous settings.')
    else:
        cpath.write_text(json.dumps(config, indent=2), encoding='utf-8')
    design_file = root / 'random_inputs.csv'
    if design_file.exists():
        design = pd.read_csv(design_file)
        if len(design) != n: raise RuntimeError('Input design length differs.')
    else:
        design = make_design(n, seed)
        design.to_csv(design_file, index=False, float_format='%.17g')

    print(f'Prescription {LABEL}: starting/resuming {n:,} input samples', flush=True)
    print(f'xi: 1e-6 .. 1e5; signed beta: up to +/-0.05; N: 50..60', flush=True)
    print('End criterion: first epsilon=1 or |eta|=1; both roll directions', flush=True)
    print(f'Chunk count: {(n+chunk_size-1)//chunk_size}; workers: {workers}', flush=True)
    pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        for left in range(0, n, chunk_size):
            right = min(n, left+chunk_size)
            outfile = chunks / f'chunk_{left:07d}_{right:07d}.csv'
            if outfile.exists():
                continue  # Resume without recomputing completed chunks.
            inputs = list(design.iloc[left:right].itertuples(index=False, name=None))
            jobs = [(row, n_grid, path_check) for row in inputs]
            if pool is None:
                results = list(map(compute_one, jobs))
            else:
                results = list(pool.map(compute_one, jobs, chunksize=8))
            flat = [r for sub in results for r in sub]
            temp = outfile.with_suffix('.tmp')
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='invalid value encountered in cast', category=RuntimeWarning)
                pd.DataFrame(flat).to_csv(temp, index=False)
            os.replace(temp, outfile)
            if right == n or left == 0 or (left // chunk_size) % 5 == 0:
                print(f'  finished input samples {right:,}/{n:,}; '
                      f'last chunk has {len(flat)} branch/failure records', flush=True)
    finally:
        if pool is not None: pool.shutdown(wait=True)
    print(f'Prescription {LABEL} scan finished. Exporting .dat...', flush=True)


def export_dat(root, include_partial=True):
    """Save all successful branches, filtered subsets, and complete variable header.

    Unsuccessful evaluations remain in chunk CSVs with failure status.
    """
    paths = sorted((root/'chunks').glob('chunk_*.csv'))
    if not paths: raise FileNotFoundError('No scan chunks found. Run without --plot-only first.')
    config = json.loads((root/'scan_config.json').read_text(encoding='utf-8'))
    expected = (config['n']+config['chunk_size']-1)//config['chunk_size']
    complete = len(paths) == expected
    if not complete:
        print(f'Partial results: {len(paths)}/{expected} chunks (not full scan)')
        if not include_partial: raise RuntimeError('Not all input samples scanned')
    out = root/'dat'
    out.mkdir(exist_ok=True)
    cat_names = ['ALL','THEORY','ACT95','SPT95','BOTH95']
    handles = {k: (out/f'Prescription_{LABEL}_{k}.dat').open('w', encoding='utf-8') for k in cat_names}
    for h in handles.values(): h.write('# '+' '.join(FIELDS)+'\n')
    count = {k: 0 for k in cat_names}
    max_ns = float('-inf')
    try:
        for i, file in enumerate(paths):
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='invalid value encountered in cast', category=RuntimeWarning)
                d = pd.read_csv(file, low_memory=False)
            if 'success' not in d: continue
            d = d.loc[bool_series(d['success'])].copy()
            if len(d)==0: continue
            d = classify(d)
            for k in FIELDS:
                if k not in d: d[k] = np.nan
            for key in ['end_epsilon','lambda_perturbative','loop_control',
                        'near_stationary_precision','strict_phi_window','paper_plot_window',
                        'theory_screen','ACT95','SPT95','both95','negative_lambda_flag']:
                d[key] = bool_series(d[key]).astype('int8')
            subsets = {'ALL': d, 'THEORY': d[d.theory_screen.eq(1)],
                       'ACT95': d[d.ACT95.eq(1)],
                       'SPT95': d[d.SPT95.eq(1)],
                       'BOTH95': d[d.both95.eq(1)]}
            for name, part in subsets.items():
                if part.empty: continue
                m = part[FIELDS].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)
                np.savetxt(handles[name], m, fmt='%.12e')
                count[name] += len(part)
            max_ns = max(max_ns, float(d['ns'].max()))
            if (i+1) % 50 == 0:
                print(f'  exported {i+1}/{len(paths)} chunks', flush=True)
    finally:
        for h in handles.values(): h.close()
    summary = {'prescription': LABEL, 'complete': complete,
               'completed_chunks': len(paths), 'expected_chunks': expected,
               'branch_counts': count,
               'max_ns_all': max_ns if np.isfinite(max_ns) else None,
               'columns': FIELDS,
               'warning': 'Accepted fractions are sampling-dependent, not Bayesian model evidence.'}
    (root/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('Saved variables (in .dat order):')
    print(', '.join(FIELDS))
    print('Branch counts:', count)
    print('MAX ns across successful branches:', summary['max_ns_all'])
    print('DATA FOLDER:', out.resolve())
    return summary


def load_dat_columns(filename, columns):
    with filename.open('r',encoding='utf-8') as f:
        hdr = f.readline().strip()
    names = hdr[2:].split() if hdr.startswith('# ') else FIELDS
    locs = [names.index(name) for name in columns]
    if filename.stat().st_size <= len(hdr)+2:
        return pd.DataFrame(columns=columns)
    # Loading only needed columns avoids exhausting RAM with large production scans.
    return pd.read_csv(filename, sep=r'\s+', comment='#', header=None,
                       usecols=locs, names=names, engine='c')[columns]


def plot_ns_r(root):
    source = root/'dat'/f'Prescription_{LABEL}_ALL.dat'
    if not source.exists(): raise FileNotFoundError(source)
    d = load_dat_columns(source, ['ns','r','N','theory_screen','ACT95','SPT95'])
    if d.empty:
        print('No successful solutions yet. Plot not generated.')
        return
    x, y = pd.to_numeric(d.ns, errors='coerce'), pd.to_numeric(d.r, errors='coerce')
    ok = np.isfinite(x)&np.isfinite(y)
    theory = d.theory_screen.eq(1)&ok
    selection = (d.ACT95.eq(1)|d.SPT95.eq(1))&ok

    fig, ax = plt.subplots(figsize=(9.8,7.1))
    ax.axvspan(ACT_CENTER-ACT_HALF_WIDTH, ACT_CENTER+ACT_HALF_WIDTH,
               color='#70ce9c', alpha=0.23, zorder=0)
    ax.axvspan(SPT_CENTER-SPT_HALF_WIDTH, SPT_CENTER+SPT_HALF_WIDTH,
               color='#80b7e7', alpha=0.20, zorder=0)
    ax.axvline(ACT_CENTER, color='#2a8961', linestyle=':', lw=1.6)
    ax.axvline(SPT_CENTER, color='#3974b6', linestyle=':', lw=1.6)
    ax.axhline(R_LIMIT, linestyle='--', color='black', lw=1.7)
    ax.scatter(x[ok],y[ok],s=4.5,alpha=0.16,color='#737a82',
               linewidths=0,rasterized=True,zorder=1)
    scale=Normalize(vmin=50,vmax=60)
    ax.scatter(x[theory],y[theory],c=d.loc[theory,'N'],cmap='viridis',norm=scale,
               s=13,alpha=0.75,linewidths=0,rasterized=True,zorder=2)
    ax.scatter(x[selection],y[selection],c=d.loc[selection,'N'],cmap='viridis',norm=scale,
               s=27,alpha=0.96,edgecolors='#151a22',linewidths=0.18,
               rasterized=True,zorder=3)
    ax.set_title(f'Prescription {LABEL}', fontsize=22, fontweight='bold', pad=14)
    ax.set_xlabel(r'$n_s$',fontsize=26,fontweight='bold')
    ax.set_ylabel(r'$r$',fontsize=26,fontweight='bold')
    ax.set_xlim(0.945,0.99)
    ax.set_ylim(0,0.065)
    ax.tick_params(which='both',direction='in',top=True,right=True,labelsize=15,width=1.25)
    ax.minorticks_on()
    for spine in ax.spines.values(): spine.set_linewidth(1.4)
    handles=[Line2D([],[],ls='',marker='o',markersize=5,color='#85898d',label='All solutions (gray)'),
             Line2D([],[],ls='',marker='o',markersize=7,color='#299989',label='Theory-screened'),
             Line2D([],[],ls='',marker='o',markersize=7,color='#299989',markeredgecolor='black',label='ACT/SPT-selected'),
             Patch(facecolor='#70ce9c',alpha=.3,label='P–ACT–LB–BK18 (95%)'),
             Patch(facecolor='#80b7e7',alpha=.3,label='P–ACT–SPT (95%)'),
             Line2D([],[],linestyle='--',color='black',label=r'$r<0.036$')]
    ax.legend(handles=handles,loc='upper left',frameon=True,framealpha=.95,
              facecolor='white',edgecolor='black',prop={'size':10,'weight':'bold'})
    cb=fig.colorbar(plt.cm.ScalarMappable(norm=scale,cmap='viridis'),ax=ax,pad=.02)
    cb.set_label(r'$N_*$ (e-folds)',fontsize=17,fontweight='bold')
    cb.ax.tick_params(labelsize=13)
    fig.tight_layout()
    folder=root/'figures'; folder.mkdir(exist_ok=True)
    for suffix in ['png','pdf']:
        dest=folder/f'Prescription_{LABEL}_ns_vs_r.{suffix}'
        fig.savefig(dest,dpi=220 if suffix=='png' else None,bbox_inches='tight')
        print('Saved figure:',dest.resolve())
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=f'Independent scan for Higgs-inflation Prescription {LABEL}')
    p.add_argument('--samples',type=int,default=300000,help='Number of random inputs (default: 300000)')
    p.add_argument('--seed',type=int,default=SEED,help='Same seed gives matched inputs for PI/PII')
    p.add_argument('--grid',type=int,default=900,help='End-root search grid; increase for convergence')
    p.add_argument('--chunk-size',type=int,default=400)
    p.add_argument('--workers',type=int,default=1,help='Use 1 for simplest Windows setup; increase for multicore')
    p.add_argument('--output',type=str,default=f'Higgs_Prescription_{LABEL}_Scan')
    p.add_argument('--plot-only',action='store_true',help='Export finished chunks and plot, without any new scan')
    p.add_argument('--skip-path-check',action='store_true',help='Faster but less reliable, not recommended')
    args=p.parse_args()
    if args.samples<1 or args.grid<100 or args.chunk_size<1 or args.workers<1:
        p.error('samples > 0; grid >= 100; chunk-size >=1; workers >= 1 required')
    root=Path(args.output)
    if not args.plot_only:
        run_scan(args.samples,args.seed,root,args.grid,args.chunk_size,args.workers,
                 not args.skip_path_check)
    export_dat(root)
    plot_ns_r(root)
    print('FINISHED. Each successful branch is stored in the ALL.dat file.')
    print('Failure records (including numerical solver errors) remain in chunks/*.csv')


if __name__ == '__main__':
    main()  # Required by Windows multiprocessing.
