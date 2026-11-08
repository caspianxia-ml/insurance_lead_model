from pathlib import Path
import tempfile,unittest
import numpy as np
import pandas as pd
from e04 import panel_frame,common_features,ARMS
from legacy.src.config import NUMERIC_FEATURES,CATEGORICAL_FEATURES
from e02_support import training_weights

class E04Tests(unittest.TestCase):
    def test_common_and_expanded_panels_keep_same_validation_Gold(self):
        idx=pd.DataFrame({'credit_code_normalized':['u1','u2','e1','e2','g1'],
                          'kind':['U_common','U_common','U_increment','U_increment','Gold_validation']})
        score=np.array([.1,.2,.3,.4,.5])
        common=panel_frame(idx,score,'common_U2');expanded=panel_frame(idx,score,'expanded_U3')
        self.assertEqual(common.credit_code_normalized.tolist(),['u1','u2','g1'])
        self.assertEqual(common.score.tolist(),[.1,.2,.5])
        self.assertEqual(expanded[expanded.kind.eq('Gold_validation')].credit_code_normalized.tolist(),['g1'])
        with self.assertRaises(ValueError):panel_frame(idx,score,'wrong')

    def test_four_way_gate_covers_U3_specific_missingness(self):
        def frame():return pd.DataFrame({**{c:np.arange(10,dtype=float) for c in NUMERIC_FEATURES},
                                       **{c:['a']*10 for c in CATEGORICAL_FEATURES}})
        p,m,u2,u3=frame(),frame(),frame(),frame();u3['paid_capital_log1p']=np.nan
        spec,audit=common_features(p,m,u2,u3)
        self.assertNotIn('paid_capital_log1p',spec.all)
        self.assertEqual(set(audit.arm),set(ARMS))
        self.assertNotIn('source',spec.all)

    def test_float64_prediction_archive_is_lossless(self):
        values=np.array([[np.nextafter(.5,1),.5,1e-15],[.1,.2,.3]],dtype=np.float64)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'scores.npz';np.savez_compressed(p,mean=values,arms=np.array(['a','b']))
            with np.load(p,allow_pickle=False) as data:np.testing.assert_array_equal(data['mean'],values)

    def test_positive_mass_fixed_for_both_U_scopes_and_quality_routes(self):
        for n_p in [1987,1987+30851]:
            for n_u in [1987,1987]:
                w=training_weights(1987,n_p,n_u)
                self.assertAlmostEqual(w[:n_p].sum(),1987)
                self.assertAlmostEqual(w[n_p:].sum(),1987)

if __name__=='__main__':unittest.main()
