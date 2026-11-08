import unittest
import numpy as np
from scipy.sparse import csr_matrix
from scipy.optimize import approx_fprime,minimize
from scipy.special import expit
from nnpu_linear import fit_nnpu,components,mixing_objective_gradient

class NNPUTests(unittest.TestCase):
    def test_analytic_gradient_matches_finite_differences(self):
        rng=np.random.default_rng(4);xp=csr_matrix(rng.normal(size=(9,3)));xu=csr_matrix(rng.normal(size=(17,3)));theta=rng.normal(size=4)
        for alpha in [1.,.35]:
            fn=lambda t:mixing_objective_gradient(t,xp,xu,.2,.03,alpha)[0]
            actual=mixing_objective_gradient(theta,xp,xu,.2,.03,alpha)[1]
            np.testing.assert_allclose(actual,approx_fprime(theta,fn,1e-7),atol=2e-7,rtol=1e-5)

    def test_empirical_risk_correction_and_dual_against_independent_epigraph(self):
        # Deliberately incompatible P/U source distributions exercise the boundary.
        xp=csr_matrix([[2.],[3.],[4.]]);xu=csr_matrix([[-2.],[-1.],[0.],[.3]])
        pi=.4;penalty=.01;theta,info=fit_nnpu(xp,xu,pi,penalty)
        self.assertTrue(info['correction_binding']);self.assertLess(info['primal_dual_gap'],5e-6)
        def risks(t):
            zp=np.asarray(xp@t[:1]).ravel()+t[1];zu=np.asarray(xu@t[:1]).ravel()+t[1]
            b=pi*np.logaddexp(0,-zp).mean();a=np.logaddexp(0,zu).mean()-pi*zp.mean()
            return a,b
        cons=[{'type':'ineq','fun':lambda t:t[-1]-risks(t)[0]},
              {'type':'ineq','fun':lambda t:t[-1]-risks(t)[1]}]
        reference=minimize(lambda t:t[-1]+penalty*t[0]**2/2,[0.,0.,1.],method='SLSQP',constraints=cons,
          options={'ftol':1e-11,'maxiter':1000})
        self.assertTrue(reference.success);self.assertAlmostEqual(info['objective'],reference.fun,places=6)
        p,n=components(theta,xp,xu,pi);self.assertGreaterEqual(p+max(0,n),0)

    def test_nonbinding_solution_and_reproducibility(self):
        xp=csr_matrix([[0.],[1.],[2.]]);xu=csr_matrix([[-2.],[-1.],[0.],[1.],[2.]])
        a,ia=fit_nnpu(xp,xu,.05,.1);b,ib=fit_nnpu(xp,xu,.05,.1)
        np.testing.assert_array_equal(a,b);self.assertTrue(ia['converged']);self.assertFalse(ia['correction_binding'])
        self.assertTrue(np.isfinite(expit(xu@a[:-1]+a[-1])).all())

    def test_prior_is_not_inferred_from_observed_sample_ratio(self):
        for pi in [0,1,-.1,1.2]:
            with self.assertRaises(ValueError):fit_nnpu(csr_matrix([[1.]]),csr_matrix([[2.]]),pi,.1)
        for count in [2,20]:
            _,result=fit_nnpu(csr_matrix([[0.]]*count),csr_matrix([[0.]]*5),.1,.1)
            # With no informative X, prior fixes sigmoid(intercept), not nP/(nP+nU).
            model,_=fit_nnpu(csr_matrix([[0.]]*count),csr_matrix([[0.]]*5),.1,.1)
            self.assertAlmostEqual(float(expit(model[-1])),.1,places=6)

if __name__=='__main__':unittest.main()
