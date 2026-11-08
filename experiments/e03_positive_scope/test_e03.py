import unittest
import numpy as np
import pandas as pd
from e03 import make_arms,common_features,ARMS
from e02_support import training_weights
from legacy.src.config import NUMERIC_FEATURES,CATEGORICAL_FEATURES

class E03Tests(unittest.TestCase):
    def test_scope_and_all_national_validation_exclusion(self):
        gold=pd.DataFrame({'credit_code_normalized':['local_val','other_val','local_fit','other_fit'],
                           'region_group':['Shanghai','Other','Suzhou','Other'],'quality_effective':['Gold']*4})
        silver=pd.DataFrame({'credit_code_normalized':['silver_local','silver_other'],
                            'region_group':['Shanghai','Other'],'quality_effective':['Silver']*2})
        folds=pd.Series({'local_val':1,'other_val':1,'local_fit':2,'other_fit':2})
        arms=make_arms(gold,silver,folds,1)
        self.assertEqual(list(arms),ARMS)
        expected=[{'local_fit'},{'local_fit','silver_local'},{'local_fit','other_fit'},{'local_fit','other_fit','silver_local','silver_other'}]
        for frame,codes in zip(arms.values(),expected):self.assertEqual(set(frame.credit_code_normalized),codes)
        self.assertTrue(all('other_val' not in set(frame.credit_code_normalized) for frame in arms.values()))

    def test_common_features_are_four_way_training_intersection(self):
        def frame():return pd.DataFrame({**{c:np.arange(10,dtype=float) for c in NUMERIC_FEATURES},
                                       **{c:['known']*10 for c in CATEGORICAL_FEATURES}})
        arms={arm:frame() for arm in ARMS};arms['C_national_Gold']['paid_capital_log1p']=np.nan
        spec,audit=common_features(arms,frame())
        self.assertNotIn('paid_capital_log1p',spec.all)
        self.assertNotIn('region_group',spec.all)
        self.assertTrue(audit.loc[audit.feature.eq('paid_capital_log1p'),'eligible_common'].eq(False).all())

    def test_all_four_arms_keep_same_class_and_total_mass(self):
        m=100
        for positive_n in [100,1000,250,3500]:
            weights=training_weights(m,positive_n,m)
            self.assertAlmostEqual(weights[:positive_n].sum(),100)
            self.assertAlmostEqual(weights[positive_n:].sum(),100)
            self.assertAlmostEqual(weights.sum(),200)

if __name__=='__main__':unittest.main()
