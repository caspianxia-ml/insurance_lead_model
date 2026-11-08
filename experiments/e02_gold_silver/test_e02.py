import unittest
import numpy as np
import pandas as pd
from e02 import training_weights,common_features,rank_metrics,fit_arm,FeatureSpec,FEATURES
from legacy.src.config import NUMERIC_FEATURES,CATEGORICAL_FEATURES

class E02Tests(unittest.TestCase):
    def test_equal_class_mass_and_regularization_scale(self):
        a=training_weights(100,100,100);b=training_weights(100,31000,100)
        self.assertAlmostEqual(a.sum(),b.sum())
        self.assertAlmostEqual(b[:31000].sum(),b[31000:].sum())
        self.assertTrue(np.allclose(b[:31000],100/31000))
        with self.assertRaises(ValueError):training_weights(0,1,1)

    def test_common_features_excludes_arm_specific_fields(self):
        def frame():
            return pd.DataFrame({**{c:np.arange(10,dtype=float) for c in NUMERIC_FEATURES},
                                **{c:['a']*10 for c in CATEGORICAL_FEATURES}})
        a,b,u=frame(),frame(),frame();a['paid_capital_log1p']=np.nan
        spec,audit=common_features(a,b,u)
        self.assertNotIn('paid_capital_log1p',spec.all)
        self.assertEqual(set(spec.all),set(FEATURES)-{'paid_capital_log1p'})
        self.assertNotIn('quality_effective',spec.all)

    def test_ranking_ties_and_fixed_denominator(self):
        df=pd.DataFrame({'credit_code_normalized':['b','a','c'],'kind':['U_evaluation','Gold_validation','Gold_validation'],'score':[.5,.5,.2]})
        metrics,ranked=rank_metrics(df,[.1,.5])
        self.assertEqual(ranked.credit_code_normalized.tolist(),['a','b','c'])
        self.assertEqual(metrics[0]['top_k'],1)
        self.assertEqual(metrics[0]['capture_rate'],.5)
        self.assertEqual(metrics[1]['evaluation_rows'],3)

    def test_model_ignores_source_and_identifiers(self):
        p=pd.DataFrame({'company_age':np.linspace(5,15,20),'company_type_clean':['P']*20,'source':['sourceP']*20})
        u=pd.DataFrame({'company_age':np.linspace(1,10,20),'company_type_clean':['U']*20,'source':['sourceU']*20})
        target=pd.DataFrame({'company_age':[4.,8.],'company_type_clean':['P','U'],'source':['leaky','text'],'quality_effective':['Gold','Silver']})
        spec=FeatureSpec(['company_age'],['company_type_clean'])
        scores,info,model=fit_arm(p,u,target,spec,20,42)
        changed=target.copy();changed['source']='opposite';changed['quality_effective']='U'
        np.testing.assert_allclose(scores,model.predict_proba(changed[spec.all])[:,1])
        self.assertTrue(info['converged'])

if __name__=='__main__':unittest.main()
