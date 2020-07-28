# -*- coding: utf-8 -*-
"""
Created on Thu May 14 12:35:08 2020

@author: Erich
"""

import numpy as np
import scipy.stats
import scipy.special
import scipy.optimize as so
import matplotlib.pyplot as plt
import statsmodels.api as sm
from os import chdir
from statsmodels.tsa.stattools import acf, pacf
## C:\Users\Erich\Documents\Python\SDE\sde_library
chdir('C:/Users/Computer/Documents/SDE')

###############################################################################
def ou_path(mu,alpha,sigma,dt,n,x0):
    """
        Generate univariate Ornstein-Uhlenbeck path given 
        (mu,alpha,sigma,dt,n,x0)
            mu : long term mean of process
            alpha : speed of reversion to mean
            sigma : instantaneous volitility
            dt : time between in-sequence observations
            n : number of observations to geneerate (INCLUDING initial value)
            x0 : initial value of sequence
            
        dX[t] = alpha*(mu-X[t])*dt + sigma*dB[t]
        
        Returns:
            x - n-observation realization of a OU(mu,alpha,sigma,dt) process
            that starts at x0
                x[0] == x0.
                
    """
    norm = scipy.stats.norm
    x = np.zeros(int(n));
    x[0] = x0;
    cond_scale = np.sqrt(sigma**2/(2*alpha)*(1-np.exp(-2*alpha*dt)))
    for ii in np.arange(1,n):
        cond_mean = x[ii-1]*np.exp(-alpha*dt) + mu*(1-np.exp(-alpha*dt))
        x[ii] = norm(loc=cond_mean,scale=cond_scale).rvs();
    return(x);
###############################################################################


###############################################################################
def ou_ll(theta, x, dt, sign=1.0):
    """
        Log-likelihood of OU(theta) generating observation sequence [x].
        theta = (mu, alpha, sigma)
            mu : long term mean of process
            alpha : speed of reversion to mean
            sigma : instantaneous volitility
        dt : time-step between observations of sequence [x].
        sign : controls sign of output, for use with numerical optimization
               routines.
    """
    norm = scipy.stats.norm
    mu,alpha,sigma = theta    
    x0 = x[0:-1];
    x1 = x[1:];    
    cond_scale = np.sqrt(sigma**2/(2*alpha)*(1-np.exp(-2*alpha*dt)));
    cond_mean =  x0*np.exp(-alpha*dt) + mu*(1-np.exp(-alpha*dt));
    ## if x.size == (n+1)
    ## --> x0.size == x1.size == cond_mean.size == n
    ## 
    loglike = np.sum(norm(loc=cond_mean,scale=cond_scale).logpdf(x1));
    return( np.sign(sign)*loglike );
###############################################################################

def ou_ll_v2(theta,x,dt,sign=1.0):
    """
    """
    mu,alpha,sigma = theta
    x0 = x[0:-1];
    x1 = x[1:];
    n = x.size-1;
    cond_mean = x0*np.exp(-alpha*dt) + mu*(1-np.exp(-alpha*dt))
    cond_var = sigma**2/(2*alpha) * (1-np.exp(-2*alpha*dt));
    ll = -0.5*n*np.log(2*np.pi*cond_var) -0.5*np.sum((x1-cond_mean)**2/(cond_var))
    return(ll);
    

###############################################################################
def ou_ols(x,dt):
    """
    Ornstein-Uhlenbeck Process Ordinary Least Squares estimates given
    observations [xt] separated by time-difference [dt].
    """    
    x0 = x[0:-1];
    x1 = x[1:];
    model = sm.OLS(x1, sm.add_constant(x0))
    results = model.fit()
    results.params
    a_, b_ = results.params
    se_ = np.sqrt(np.sum(results.resid**2)/(results.resid.size-2))
    ## invert to get parameters of sde
    alpha_ = -np.log(b_)/dt;
    tau_ = 1/alpha_;
    mu_ = a_/(1-b_)
    sigma_ = se_*np.sqrt((-2*np.log(b_))/((1-b_**2)*dt))
    theta_ols = (mu_, tau_, sigma_);
    return(theta_ols);
###############################################################################
    
###############################################################################
def ou_fit(x,dt):
    """
    """
    ## get OLS parameter values
    ##theta_ols = ou_ols(x,dt);
    cons = ({'type': 'ineq', 'fun': lambda theta: theta[1]},
            {'type': 'ineq', 'fun': lambda theta: theta[2]});
    ##ou_ll(theta_ols,x,dt)
    
    
    n = x.size-1
    M1 = np.mean(x)
    M2 = (1/n)*np.sum((x-M1)**2)
    M3 = (1/(n-1))*np.sum((x[1:]-M1)*(x[0:-1]-M1));
    M4 = (1/(n-2))*np.sum( (x[2:]-M1)*(x[0:-2]-M1) );
    if (M2 < 0 or M3 < 0 or M4 < 0): 
        theta_mom = slick.ouMLEfit(x,dt)
    else:
        theta_mom = np.array([M1, 1/dt*np.log(M3/M4), \
                2*(1/dt)*M3**2/M4*np.log(M3/M4) ]);
    print(theta_mom)
    ## minimize negative log-likelihood --> maximize log-likelihood.
    
    bnds = ((None,None),(0,None),(0,None))  # alpha, sigma >0
    
    solution = so.minimize(fun=ou_ll, method = 'SLSQP', bounds=bnds,
                           x0 = theta_mom,
                           args = (x,dt,-1));
    theta_mle = solution.x
    return(theta_mle);

###############################################################################



###############################################################################
def cir_path(mu,alpha,sigma,dt,n,x0):
    """
    
        dX[t] = alpha*(mu-X[t])dt + sigma*sqrt(x[t])dB[t]
        
        let c = 4*alpha*sigma**(-2)*(1-exp(-alpha*dt))**(-1)
        
        The transitional distribution of c*X[t] given X[t-1] is
        NonCentralChiSquare( df = 4*alpha*mu*sigma**(-2)
                             nc = c*X[t-1]*exp(-alpha*dt))
        
        nc == x[t-1] * c*exp(-alpha*dt)
           == x[t-1] * nc_coeff
           where
               nc_coeff = c*exp(-alpha*dt)
    """
    ## 
    if mu <= 0 or alpha <= 0 or sigma <= 0 :
        print('ERROR cir_path, parameter < = 0')
        return -1
    if 2*mu*alpha < sigma**2:
        print("ERROR cir_path, 2(mu x alpha) < sigma**2")
        return -2
    ncx2 = scipy.stats.ncx2
    x = np.zeros(n);
    x[0] = x0;
    c = (2*alpha)/((1-np.exp(-alpha*dt))*sigma**2);
    nc_coeff = 2*c*np.exp(-alpha*dt);
    df_ = 4*alpha*mu/sigma**2;
    for ii in np.arange(1,n):
        nc_ = nc_coeff*x[ii-1]
        Y_ = ncx2(df=df_,nc=nc_).rvs()
        x[ii] = Y_/(2*c);
        
    return(x);
###############################################################################



###############################################################################
def ll_cir(theta, yt, dt, sign=1.0):
    """
    Signed log-likelihood of the CIR(mu,alpha,sigma) process generating the 
    observations in the chain(s) in yt.
        
    Parameters
    ----------
    theta : array-like
        mu, alpha, sigma = theta
        mu    : long term mean
        alpha : rate of reversion toward the mean
        sigma : instantaneous volitility
    yt : array-like or list
        chain(s) of observations. 
    dt : time between observations of a chain in yt
    sign : tells function to return positive or negative log-likelihood
        
    Returns
    -------
    ll_sum : Signed log-likelihood of the parameters in theta generating
              the observations in the chain(s) in yt. 
        
    """
    ncx2 = scipy.stats.ncx2;
    mu,alpha,sigma = theta
    
    if type(yt) == list:
        ## log-likelihood will be the sum of the loglikelihoods for each
        ## realization in yt:
        ll_sum = 0.0;
        for yt_ in yt:
            ll_sum += ll_cir(theta,yt_,dt);
    else:
        ## assumed that [yt] is a single realization of a CIR process
        ## --> type(yt) is numpy.ndarray == True
        y0 = yt[0:-1];
        y1 = yt[1:];
        N = yt.size;
        c = (2*alpha)/((1-np.exp(-alpha*dt))*sigma**2);
        q = (2*alpha*mu)/(sigma**2) - 1;
        u = c*y0*np.exp(-alpha*dt);
        v = c*y1;
        s=2*c*y1
        nc = 2*u;
        df = 2*q+2;
        gpdf = ncx2(df=df,nc=nc).pdf(s);
        ppdf = 2*c*gpdf
        ll_sum = np.sum(np.log(ppdf))
    return( np.sign(sign)*ll_sum );
###############################################################################
       
    

###############################################################################
def cir_fit(yt,dt,quantile=None):
    """
    MLE fit of parameters (theta = (mu,alpha,sigma)) for a CIR process
    Given observation sequence yt and constant time-between-observations of dt.
    
    Parameters
    ----------
    yt : array-like or list of array-like
         chain(s) of observations from a CIR process. 
    dt : numeric
         intra-time-difference between observations of chain(s) in yt.
    quantile : (Optional)
               quantile = {'q':q_val, 'p':p_val}
               q_val : numeric; 0 < q_val
               p_val : numeric; 0 < p_val < 1.0
               constrains MLE to parameterizations of the CIR process that
               have q_val as the p_val*100%-tile of the asymptotic process.
               e.g. {'q':100., 'p':0.90} CIR process will have 100 as its 90th 
               percentile.
    Returns
    -------
     theta_hat = mu_hat,alpha_hat,sigma_hat
     MLE of the parameters of the CIR process.
    
        
    
    as t-->Inf a CIR process converges to a 
    Gamma(shape=2*mu*alpha/sigma**2, scale=sigma**2/(2*alpha)) random variable.
    """
    ##
    gamma = scipy.stats.gamma;
    
    p = q = None;
    cons = None;
    if quantile is not None:
        ## expecting quantile to be dict that contains keys {'q','p'}
        ## quantile = {p:p_val, q:q_val}
        p = quantile['p'];
        q = quantile['q'];
        cons = ({'type': 'eq', 'fun': lambda theta: \
            gamma(a=2*theta[1]*theta[0]/theta[2]**2, \
            scale=theta[2]**2/(2*theta[1])).cdf(q) - p })
    
    ## Determining initial parameterization guess based on whether one or 
    ## multiple chains were passed.
    if type(yt) == list:
        ## 
        nchains = len(yt); ## number of chains
        chain_lengths = [temp.size for temp in yt] ## nobs of each chain.
        ## append all chains together into one long "temp_chain"
        temp_chain = np.array([]);
        for temp in yt:
            temp_chain = np.concatenate((temp_chain,temp));
        
        ## indecies where individual chains were appended:
        idx_ = [temp for temp in np.cumsum(chain_lengths[0:-1])]
        
        y0 = temp_chain[0:-1]
        dy = np.matrix(np.diff(temp_chain)/np.sqrt(y0)).T        
        ## remove these indecies from y0 and dy 
        ## (they correspond to the connection between 2 chains):
        idx_r = np.array(idx_)-1
        
        y0 = y0[[temp not in idx_r for temp in np.arange(y0.size)]]
        dy = dy[[temp not in idx_r for temp in np.arange(dy.size)]]
        
        ## Now find the OLS solution
        reg = np.matrix([dt/np.sqrt(y0), dt*np.sqrt(y0)]).T      
        theta,_,_,_ = np.linalg.lstsq(reg,dy,rcond=None)
        a0 = float(-theta[1])
        mu0 = float(theta[0]/a0)
        sigma0 = np.std(dy-np.matmul(reg,theta))/np.sqrt(dt)
        theta0 = np.array((mu0,a0,sigma0));
    else:
        ## assumed that yt is a single realization
        ## yt is a numpy.ndarray of sequential observations of a CIR process.
        y0 = yt[0:-1];        
        ## OLS for initial parameter estimates
        dy = np.matrix(np.diff(yt)/np.sqrt(y0)).T
        reg = np.matrix([dt/np.sqrt(y0), dt*np.sqrt(y0)]).T      
        theta,_,_,_ = np.linalg.lstsq(reg,dy,rcond=None)
        a0 = float(-theta[1])
        mu0 = float(theta[0]/a0)
        sigma0 = np.std(dy-np.matmul(reg,theta))/np.sqrt(dt)
        theta0 = np.array((mu0,a0,sigma0));
    
    ## Numreically find the MLE of theta
    if cons is None:
        ## unconstrained maximume likelihood estimates
        sol = so.minimize(fun=ll_cir, x0=theta0, args=(yt,dt,-1), \
                          bounds=((0.0,None),(0.0,None),(0.0,None)));
    else:
        ## constrained MLEs
        sol = so.minimize(fun=ll_cir, x0=theta0, args=(yt,dt,-1), \
                 bounds=((0.0,None),(0.0,None),(0.0,None)), constraints=cons);
    return(sol.x);
###############################################################################
    
###############################################################################
def cir_ppf(theta,p,sign=1.0):
    """
    CIR process (mu,alpha,sigma):
        X_{inf} ~ Gamma(a=mu*2*alpha/sigma**2,  scale=sigma**2/(2*alpha))
    """
    mu_,alpha_,sigma_ = theta;
    a_ = mu_*2*alpha_/sigma_**2;
    scale_ = sigma_**2/(2*alpha_);
    return( np.sign(sign)*scipy.stats.gamma(a=a_,scale=scale_).ppf(p) );
###############################################################################    


###############################################################################
def ci_cir_ppf( yt, dt, p, alpha=0.05):
    """
    Profile-likelihood based confidence interval of the asymptotic p-th
    percentile of the CIR process based on the chain(s) in yt
    """
    ## 
    chi2 = scipy.stats.chi2
    ## unconstrained MLE
    theta_hat = cir_fit(yt,dt);
    ## MLE of the p-th percentile
    q_hat = cir_ppf(theta_hat,p);
    ## log-likelihood of the chain(s) evaluated at the unconstrained MLEs
    ll_mle = ll_cir(theta_hat,yt,dt);
    
    ## as we shift the p-th percentile off of q_hat, the constrained 
    ## log-liklihood will drop off from ll_mle. What log-likelihood corresponds
    ## to the (1-alpha) confident lower/upper bounds? --> ll_target.
    ## 2*(ll_mle-ll_target) - chi2(df=1).ppf(1-alpha) == 0
    ## --> ll_target = ll_mle-chi2(df-1).ppf(1-alpha)/2
    ll_target = ll_mle - chi2(df=1).ppf(1-alpha)/2;        
    ## cir_ppf(theta_target,p) minimized or maximized subject to the constraint:
    ##     ll_cir(theta_target,yt,dt) == ll_target
    cons = ({'type':'eq',  'fun': lambda theta: ll_cir(theta,yt,dt) - ll_target},)
    
    ## Lower bound
    sol = so.minimize(fun=cir_ppf,
                      x0 = theta_hat,
                      args=(p,1.,), ## minimize cir_ppf function
                      bounds=((0.0,None),(0.0,None),(0.0,None)),
                      constraints = cons)
    ## CIR parameterization associated with the lower bound of the p-th 
    ## percentile
    theta_L = sol.x;
    ## lower bound of the p-th percentile ("q")
    q_L = cir_ppf(theta_L,p)
    ## Upper bound
    sol = so.minimize(fun=cir_ppf,
            x0 = theta_hat,
            args=(p,-1.,), ## minimize -1*cir_ppf function --> maximize cir_ppf
            bounds=((0.0,None),(0.0,None),(0.0,None)),
            constraints = cons);
    
    ## CIR parameterization  associated with the upper bound of the p-th 
    ## percentile
    theta_U = sol.x
    ## upper bound of the p-th percentile ("q")
    q_U = cir_ppf(theta_U,p)    
    return( np.array([q_L, q_U]));
###############################################################################
    



###############################################################################
class cir_asymptotic:
    def __init__(self,mu,alpha,sigma):
        self.mu = mu;
        self.alpha = alpha;
        self.sigma = sigma;
        self._a = 2*alpha*mu/sigma**2;
        self._scale = sigma**2/(2*alpha);
        return;
    def pdf(self,x):
        """
        """
        return(scipy.stats.gamma(a=self._a,scale=self._scale).pdf(x));        
    def cdf(self,x):
        return(scipy.stats.gamma(a=self._a,scale=self._scale).cdf(x));
    def ppf(self,q):
        return(scipy.stats.gamma(a=self._a,scale=self._scale).ppf(q));
###############################################################################


def ll_ensemble_cir(theta, yt, dt, sign=1.0):
    """
        "ensemble" meanin yt can be one or more time series realizations.
        
    """
      
    return(-1);


# mu,alpha,sigma = (25.0, 1.05, 1.0)
# n_realizations = 250;
# X = [sde.ou_path(mu,alpha,sigma,dt, 100, norm(loc=mu,scale=4*sigma/np.sqrt(2*alpha)).rvs()) for ii in np.arange(n_realizations)]
# theta_mle_ii = [ou_fit(X[ii],dt) for ii in np.arange(n_realizations)]
# ll_mle_ii = [ou_ll(theta_mle_ii[ii],X[ii],dt) for ii in np.arange(n_realizations)]
# chi_rv = [2*(ll_mle_ii[ii] - ou_ll(theta,X[ii],dt)) for ii in np.arange(n_realizations)]



# W = np.matrix(X[0:2])

# [plt.plot(temp,'.-') for temp in X]

def ouMLEfit(x, dt=1, verbose=False):
    '''
    Maximum likelihood OU fit, based on
    1. MLE estimates of AR(1) parameters of x
    2. OU parameters from the AR(1) fit

    Parameters
    ----------
    x : TYPE input list or array, float
        DESCRIPTION. OU process is fit to these data:
            dX = k(x-mu)dt + sigma*dW
    dt : TYPE, float, optional time/distance between observations
        DESCRIPTION. The default is 1.

    Returns
    -------
    k : TYPE float
        DESCRIPTION reversion rate
    mu : TYPE float 
        DESCRIPTION mean of the process, x converges to mu.
    sigma : TYPE float
        DESCRIPTION uncertainity in the process.

    '''
    
    from statsmodels.tsa.arima_model import ARMA

#    import numpy as np
#    from scipy.stats import norm
    
    ## AR(1) fit
    mod = ARMA(x, order=(1,0))
    result = mod.fit()  

    # fit model
    
    if (verbose):
        print(result.summary())

    mu,b = result.params

    k =  np.sqrt(-np.log(b)/dt)  
    a = (1-np.exp(-k*dt))*mu
    se = np.std(result.resid)
    sigma = se*np.sqrt((1-2*np.log(b))/((1-b**2)*dt))
   
    return (mu, k, sigma)

def ouDistn(X0, T, k, mu, sigma):
    '''
    

    Parameters
    ----------
    X0 : TYPE float
        DESCRIPTION starting point 
    T : TYPE float
        DESCRIPTION distance from X0, time/distance units
    k : TYPE float
        DESCRIPTION rate of reversion
    mu : TYPE float
        DESCRIPTION long-term mean of the process
    sigma : TYPE float
        DESCRIPTION process uncertainity.

    Returns
    -------
    expectedMU : TYPE float
        DESCRIPTION. expected value of X, T units from X0
    stdev : TYPE float
        DESCRIPTION standard deviation of X, T units from X0

    '''
    
    expectedMu = X0*np.exp(-k*T) + mu*(1-np.exp(-k*T))
    stdev = sigma**2 *(1-np.exp(-2*k*T))/(2*k)
    stdev = np.sqrt(stdev)
    
    return (expectedMu, stdev)


def autoC(X_CIR):
    from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
    
    if len(X_CIR) < 100:
        print('insufficient points')
        return -1

    plt.figure()
    plot_acf(X_CIR, lags=20, title='CIR autocorrleation')
    plt.grid()

    plt.figure(5)
    plot_pacf(X_CIR, lags=20, title='CIR partial autocorrelation')
    plt.grid()
    plt.show()
    
    return 1


## mle estimates of mu, alpha, sigma
#theta_CIR = cir_fit(X_CIR, dt)

# MLE values


def cirProfileLike(X_CIR, dt):
    '''
    provile likelihood Confidence intervals for 
    parameters of a CIR process

    Parameters
    ----------
    X_CIR : TYPE np.array
        DESCRIPTION. collection of CIR realizations
   
    dt : TYPE float
        DESCRIPTION time between samples.

    Returns
    -------
    int
        DESCRIPTION successfull completion

    '''

    theta_CIR = cir_fit(X_CIR, dt)
    print('MLE: {}'.format(theta_CIR))
    
    logLikeSoln = ll_cir(theta_CIR, X_CIR, dt)
    print('log likelihood at MLE: {}'.format(logLikeSoln))
    chi2Val = scipy.stats.chi2.ppf(0.95, df=1)
    
    ## **** profile likelihood of mu *****
    
    theta = np.copy(theta_CIR)
#    mu = float(np.copy(theta_CIR[0]))
    
    # find range
    LCL=0
    rangeStep = theta[0]/150
    for xx in np.arange(0,theta[0],rangeStep):
        theta[0] = xx
        logLike = ll_cir(theta,X_CIR, dt)
        pl = -2*(logLikeSoln-logLike)
#        print(pl)
        if pl > -chi2Val:
            LCL = xx-rangeStep
            break
        
        
    UCL = 2*theta_CIR[0]
    theta = np.copy(theta_CIR) 
    pl = 0.0
    while pl > -chi2Val:
        theta[0] += rangeStep
        logLike = ll_cir(theta, X_CIR, dt)
        pl = -2*(logLikeSoln-logLike)
        if pl < -chi2Val:
            UCL = theta[0]+rangeStep
            break
    
    
    saveMu = [0*100]
    muRange = [x for x in np.arange(LCL, UCL, step=rangeStep)]
    for muDelta in muRange: # CIR: can't be negative
        theta[0] = muDelta
        logLike= ll_cir(theta,X_CIR,dt)
        pl = -2*(logLikeSoln - logLike)
        saveMu.append(pl)

    plt.figure()    
    plt.plot(muRange,saveMu[1:])
    plt.title('Profile Likelihood: ' + r'$\mu$')
    plt.xlabel(r'$\mu$')
    plt.ylabel('Likelihood Ratio')
    plt.axhline(y=-chi2Val, color='red', ls='--')
    plt.grid()
    plt.show()

    ## *****  profile likelihood of alpha ***************   

    theta = np.copy(theta_CIR)
#    alpha = float(np.copy(theta_CIR[1]))
    
     # find range
   
    LCL=0
    rangeStep = theta[1]/150
    for xx in np.arange(0,theta[0],rangeStep):
        theta[1] = xx
        logLike = ll_cir(theta,X_CIR, dt)
        pl = -2*(logLikeSoln-logLike)
#        print(pl)
        if pl > -chi2Val:
            LCL = xx-rangeStep
            break     
        
        
    UCL = 2*theta_CIR[1]   
    theta = np.copy(theta_CIR) 
    pl = 0.0
    while pl > -chi2Val:
        theta[1] += rangeStep
        logLike = ll_cir(theta, X_CIR, dt)
        pl = -2*(logLikeSoln-logLike)
        if pl < -chi2Val:
            UCL = theta[1]+rangeStep
            break
    
    
    saveAlpha = [0*100]
    alphaRange = [x for x in np.arange(LCL,UCL, step=rangeStep)] 
    for alphaDelta in alphaRange:
        theta[1] = alphaDelta
        logLike=ll_cir(theta, X_CIR, dt)
        pl = -2*(logLikeSoln - logLike)
        saveAlpha.append(pl)

    plt.figure()
    plt.plot(alphaRange, saveAlpha[1:])  
    plt.title('Profile likelihood: ' + r'$\alpha$')
    plt.xlabel(r'$\alpha$')
    plt.ylabel('Likelihood Ratio') 
    plt.axhline(y=-chi2Val, color='red', ls=':') 
    plt.grid()
    plt.show()

    ## ***** profile likelihood of sigma  ***********

    theta = np.copy(theta_CIR)
#   sigma = float(np.copy(theta_CIR[2]))
    
      # find range
    LCL=0
    rangeStep = theta[2]/150
    for xx in np.arange(0,theta[2],rangeStep):
        theta[2] = xx
        logLike = ll_cir(theta,X_CIR, dt)
        pl = -2*(logLikeSoln-logLike)
#        print(pl)
        if pl > -chi2Val:
            LCL = xx-rangeStep
            break
        
        
    UCL = 2*theta_CIR[2]
    theta = np.copy(theta_CIR) 
    pl = 0.0
    while pl > -chi2Val:
        theta[2] += rangeStep
        logLike = ll_cir(theta, X_CIR, dt)
        pl = -2*(logLikeSoln-logLike)
        if pl < -chi2Val:
            UCL = theta[2]+rangeStep
            break
    
    saveSigma = [0*100]
    sigmaRange = [x for x in np.arange (LCL, UCL, rangeStep)] 
    for sigmaDelta in sigmaRange:
        theta[2] = sigmaDelta
        logLike=ll_cir(theta, X_CIR, dt)
        pl = -2*(logLikeSoln - logLike)
        saveSigma.append(pl)

    plt.figure()
    plt.plot(sigmaRange, saveSigma[1:])  
    plt.title('Profile likelihood: ' + r'$\sigma$')
    plt.xlabel(r'$\sigma$')
    plt.ylabel('Likelihood Ratio')  
    plt.axhline(y=-chi2Val, color='red', ls=':') 
    plt.grid()
    plt.show()    
    
    return 1

def ouProfileLike(X_OU, dt):
    '''
    provile likelihood Confidence intervals for 
    parameters of a CIR process

    Parameters
    ----------
    X_OU : TYPE np.array
        DESCRIPTION. collection of OU realizations
   
    dt : TYPE float
        DESCRIPTION time between samples.

    Returns
    -------
    int
        DESCRIPTION 1 = successful completion

    '''

    from scipy.stats import chi2
    import matplotlib.pyplot as plt
    
#    theta_OU = ou_fit(X_OU, dt) # X0 different initial value
    theta_OU = ouMLEfit(X_OU, dt)
    print('MLE: {}'.format(theta_OU))
    if theta_OU[2] < 0:
        print('Error in profile likelihood: sigma {} '.format(theta_OU[2]))
        return -1
    
    logLikeSoln = ou_ll(theta_OU, X_OU, dt)
    print('OU log likelihood at MLE: {}'.format(logLikeSoln))
    chi2Val = chi2.ppf(0.95, df=1)
    
    ## **** profile likelihood of mu *****
    
    theta = np.copy(theta_OU)
    
    # find range
    LCL=0
    rangeStep = theta[0]/150
    for xx in np.arange(0,theta[0],rangeStep):
        theta[0] = xx
        logLike = ou_ll(theta,X_OU, dt)
        pl = -2*(logLikeSoln-logLike)
#        print(pl)
        if pl > -chi2Val:
            LCL = xx-rangeStep
            break
        
        
         
    UCL = 2*theta_OU[0]
    theta = np.copy(theta_OU) 
    pl = 0.0
    while pl > -chi2Val:
        theta[0] += rangeStep
        logLike = ou_ll(theta, X_OU, dt)
        pl = -2*(logLikeSoln-logLike)
        if pl < -chi2Val:
            UCL = theta[0]+rangeStep
            break
    
    saveMu = [0*150]
    muRange = [x for x in np.arange(LCL,UCL, step=rangeStep)]
    for muDelta in muRange:
        theta[0] = muDelta
        logLike= ou_ll(theta,X_OU,dt)
        pl = -2*(logLikeSoln - logLike)
        saveMu.append(pl)

    plt.figure()    
    plt.plot(muRange,saveMu[1:])
    plt.title('Profile Likelihood: ' + r'$\mu$')
    plt.xlabel(r'$\mu$')
    plt.ylabel('Likelihood Ratio')
    plt.axhline(y=-chi2Val, color='red', ls='--')
    plt.grid()
    plt.show()

    ## *****  profile likelihood of alpha ***************   

    theta = np.copy(theta_OU)
    
     # find range
    LCL=0
    rangeStep = theta[1]/150
    for xx in np.arange(0,theta[1],rangeStep):
        theta[1] = xx
        logLike = ou_ll(theta,X_OU, dt)
        pl = -2*(logLikeSoln-logLike)
#        print(pl)
        if pl > -chi2Val:
            LCL = xx-rangeStep
            break
        
        
    UCL = 2*theta_OU[1]
    theta = np.copy(theta_OU) 
    pl = 0.0
    while pl > -chi2Val:
        theta[1] += rangeStep
        logLike = ou_ll(theta, X_OU,dt)
        pl = -2*(logLikeSoln-logLike)
        if pl < -chi2Val:
            UCL = theta[1]+rangeStep
            break
         
    
    saveAlpha = [0*100]
    alphaRange = [x for x in np.arange(LCL,UCL, step=rangeStep)] 
    for alphaDelta in alphaRange:
        theta[1] = alphaDelta
        logLike=ou_ll(theta, X_OU, dt)
        pl = -2*(logLikeSoln - logLike)
        saveAlpha.append(pl)

    plt.figure()
    plt.plot(alphaRange, saveAlpha[1:])  
    plt.title('Profile likelihood: ' + r'$\alpha$')
    plt.xlabel(r'$\alpha$')
    plt.ylabel('Likelihood Ratio') 
    plt.axhline(y=-chi2Val, color='red', ls=':') 
    plt.grid()
    plt.show()

    ## ***** profile likelihood of sigma  ***********

    theta = np.copy(theta_OU)
    
      # find range
    LCL=0
    rangeStep = theta[2]/150
    for xx in np.arange(0,theta_OU[2],rangeStep):
        theta[2] = xx
        logLike = ou_ll(theta,X_OU, dt)
        pl = -2*(logLikeSoln-logLike)
#        print(pl)
        if pl > -chi2Val:
            LCL = xx-rangeStep
            break
        
    UCL = 2*theta_OU[2]
    theta = np.copy(theta_OU) 
    pl = 0.0
    while pl > -chi2Val:
        theta[2] += rangeStep
        logLike = ou_ll(theta, X_OU, dt)
        pl = -2*(logLikeSoln-logLike)
        if pl < -chi2Val:
            UCL = theta[2]+rangeStep
            break
    
    print(LCL, UCL)
    
    saveSigma = [0*100]
    sigmaRange = [x for x in np.arange (LCL, UCL, rangeStep)] 
    for sigmaDelta in sigmaRange:
        theta[2] = sigmaDelta
        logLike=ou_ll(theta, X_OU, dt)
        pl = -2*(logLikeSoln - logLike)
        saveSigma.append(pl)

    plt.figure()
    plt.plot(sigmaRange, saveSigma[1:])  
    plt.title('Profile likelihood: ' + r'$\sigma$')
    plt.xlabel(r'$\sigma$')
    plt.ylabel('Likelihood Ratio')  
    plt.axhline(y=-chi2Val, color='red', ls=':') 
    plt.grid()
    plt.show()    
    
    return 1

