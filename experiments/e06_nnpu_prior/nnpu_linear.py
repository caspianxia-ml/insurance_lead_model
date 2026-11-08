"""Linear logistic-loss nnPU empirical risk, solved as a convex max problem.

R_nnPU = max(A,B), A=E_U softplus(z)-pi E_P z,
B=pi E_P softplus(-z). Add lambda/2 * ||w||² (intercept unpenalized).
This minimizes the paper's nonnegative estimator; it is NOT a reproduction of
its stochastic Algorithm 1 or a neural network. Solve smooth uPU A first;
if the correction binds, solve the one-dimensional concave mixing dual.
"""
import numpy as np
from scipy.optimize import minimize,brentq
from scipy.special import expit,logit

def components(theta,xp,xu,prior):
    zp=np.asarray(xp@theta[:-1]).ravel()+theta[-1]
    zu=np.asarray(xu@theta[:-1]).ravel()+theta[-1]
    positive=prior*np.logaddexp(0,-zp).mean()
    negative=np.logaddexp(0,zu).mean()-prior*np.logaddexp(0,zp).mean()
    return float(positive),float(negative)

def mixing_objective_gradient(theta,xp,xu,prior,penalty,alpha):
    zp=np.asarray(xp@theta[:-1]).ravel()+theta[-1]
    zu=np.asarray(xu@theta[:-1]).ravel()+theta[-1]
    a=np.logaddexp(0,zu).mean()-prior*zp.mean()
    b=prior*np.logaddexp(0,-zp).mean()
    cp=prior*((1-alpha)*(expit(zp)-1)-alpha)/len(zp)
    cu=alpha*expit(zu)/len(zu)
    gradient=np.r_[np.asarray(xp.T@cp+xu.T@cu).ravel()+penalty*theta[:-1],cp.sum()+cu.sum()]
    value=alpha*a+(1-alpha)*b+.5*penalty*np.dot(theta[:-1],theta[:-1])
    return float(value),gradient

def fit_nnpu(xp,xu,prior,penalty):
    if not 0<prior<1 or penalty<=0:raise ValueError('Invalid prior or penalty')
    current=np.zeros(xp.shape[1]+1);current[-1]=logit(prior)
    totals={'iterations':0,'evaluations':0,'inner_solves':0,'roundoff_stationary_exits':0,'max_inner_gradient_inf':0.};solutions={}
    def solve(alpha):
        nonlocal current
        if alpha in solutions:return solutions[alpha]
        result=minimize(mixing_objective_gradient,current,args=(xp,xu,prior,penalty,alpha),jac=True,
          method='L-BFGS-B',options={'gtol':1e-9,'ftol':1e-15,'maxiter':2000,'maxls':50})
        totals['iterations']+=int(result.nit);totals['evaluations']+=int(result.nfev);totals['inner_solves']+=1
        grad=float(np.max(np.abs(result.jac)))
        # L-BFGS-B can exhaust its line search at machine precision even with
        # a gradient far below the final gate. Accept that specific status only
        # with a stricter stationary check; preserve and report every such exit.
        roundoff_stationary=(result.status==2 and grad<=1e-7)
        totals['roundoff_stationary_exits']+=int(roundoff_stationary)
        totals['max_inner_gradient_inf']=max(totals['max_inner_gradient_inf'],grad)
        if not (result.success or roundoff_stationary) or not np.isfinite(result.fun) or grad>2e-5:
            raise RuntimeError(f'nnPU inner optimization failed: alpha={alpha}, status={result.status}, gradient={grad}, {result.message}')
        current=result.x.copy();positive,negative=components(current,xp,xu,prior)
        solutions[alpha]=(current.copy(),positive,negative,grad)
        return solutions[alpha]
    initial=solve(1.0)
    if initial[2]>=-1e-8:alpha=1.0;solution=initial
    else:
        low=1e-6
        if solve(low)[2]<=0:raise RuntimeError('nnPU dual root not bracketed')
        alpha=float(brentq(lambda a:solve(float(a))[2],low,1.,xtol=1e-9,rtol=1e-9,maxiter=60))
        solution=solve(alpha)
    theta,positive,negative,grad=solution
    objective=positive+max(0,negative)+.5*penalty*np.dot(theta[:-1],theta[:-1])
    dual_value=positive+alpha*negative+.5*penalty*np.dot(theta[:-1],theta[:-1])
    gap=float(objective-dual_value)
    # The primal/dual objective gap is the certificate. An unweighted boundary
    # residual need not be tiny when its optimal multiplier is very small.
    if gap>5e-6 or gap < -1e-10:
        raise RuntimeError(f'nnPU optimum verification failed: alpha={alpha}, negative={negative}, gap={gap}')
    return theta,{'positive_risk':positive,'uncorrected_negative_risk':negative,'nnpu_risk':positive+max(0,negative),
      'objective':float(objective),'penalty_lambda':penalty,'dual_alpha':alpha,'primal_dual_gap':gap,'gradient_inf':grad,
      'correction_binding':alpha<1.,'uPU_optimum_negative_risk':initial[2],'converged':True,**totals}
