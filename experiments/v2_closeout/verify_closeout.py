"""Independent checks of fixed budgets, archived LR rankings and industry outputs."""
import argparse,json,itertools
from pathlib import Path
import numpy as np
import pandas as pd
from analyze_closeout import read,sha,save,dump,FIELDS

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--e06',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    a.out.mkdir(parents=True);checks=[]
    def check(name,passed):
        checks.append(dict(check=name,passed=bool(passed)))
        if not passed:raise AssertionError(name)
    audit=a.root/'analysis_002';standard=a.root/'industry_standardized_003'
    for d in [audit,standard]:
        manifest=json.loads((d/'output_hashes.json').read_text())
        for rel,expected in manifest.items():check(f'hash:{d.name}:{rel}',sha(d/rel)==expected)
    raw=read(audit/'fixedK_Gold_capture_by_fold_seed.csv');r=a.e06/'runs/main_002'
    for scope,fold,seed in itertools.product(['U2','U3'],range(1,6),[42,123,2026]):
        idx=read(r/'prediction_index'/f'fold_{fold}_{scope}.csv')
        with np.load(r/'predictions'/f'fold_{fold}_{scope}_seed_{seed}.npz') as z:
            for scores,route,method in zip(z['mean'],z['routes'],z['methods']):
                independently=idx[['credit_code_normalized','kind']].copy();independently['score']=scores
                independently=independently.sort_values(['score','credit_code_normalized'],ascending=[False,True])
                for k in [100,500,1000]:
                    row=raw.loc[raw.scope.eq(scope)&raw.fold.eq(fold)&raw.seed.eq(seed)&raw.route.eq(route)&raw.method.eq(method)&raw.top_k.eq(k)]
                    check(f'fixedK:{scope}:{fold}:{seed}:{route}:{method}:{k}',len(row)==1 and int(row.gold_hits.iloc[0])==int(independently.head(k).kind.eq('Gold_validation').sum()))
    for scope in ['U2','U3']:
        panel='common_U2' if scope=='U2' else 'expanded_U3'
        for route in ['Gold_only','Gold_Silver']:
            candidate=read(audit/f'{scope}_{route}_LR_U_candidate_top1000.csv')
            old=read(a.e06/'reports/main_001'/f'{scope}_{panel}_{route}_U_development_scores_and_ranks.csv')
            selected=old.set_index('credit_code_normalized').loc[candidate.credit_code_normalized]
            check(f'{scope}:{route}:old_LR_scores_unchanged',np.allclose(pd.to_numeric(candidate.score_mean),pd.to_numeric(selected.LR_score),rtol=0,atol=1e-15))
            check(f'{scope}:{route}:old_Top1000_set_unchanged',set(candidate.credit_code_normalized)==set(old.sort_values('LR_rank').head(1000).credit_code_normalized))
    summaries=json.loads((standard/'summary.json').read_text())
    for s in summaries:
        role=s['role'];df=read(standard/f'{role}_standard_industry.csv')
        check(f'{role}:no_row_removed',len(df)==s['n_before']==s['n_after'])
        if role!='U1_subset':
            before=read(audit/f'industry_entities_{role}.csv')
            check(f'{role}:raw_industry_columns_preserved',df[['credit_code_normalized',*FIELDS]].equals(before[['credit_code_normalized',*FIELDS]]))
        check(f'{role}:unknown_count_preserved',int(df.industry_standard_status.eq('unknown_retained').sum())==s['unknown_retained'])
    save(a.out/'independent_checks.csv',checks);dump(a.out/'manifest.json',dict(status='complete',checks=len(checks),all_passed=True,new_fits=0,final_Gold_scored=False,code_sha256=sha(Path(__file__))))
    dump(a.out/'output_hashes.json',{str(p.relative_to(a.out)):sha(p) for p in a.out.iterdir() if p.is_file()});print(len(checks),'independent checks passed')

if __name__=='__main__':main()
