"""Paired E02 Gold-only vs Gold+Silver development experiment.

Prepare once, smoke-check once, then train a fresh full run. All directories are
append-only by run ID. Final Gold holdout features are never read by train().
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import platform
import time
import warnings
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits
from legacy.src.config import NUMERIC_FEATURES, CATEGORICAL_FEATURES
from legacy.src.data_cleaning import _valid_uscc, _normalise_credit_code
from legacy.src.feature_engineering import build_features
from legacy.src.model_logistic_baseline import stratified_subset
from legacy.src.preprocessing import FeatureSpec, feature_eligibility, make_preprocessor

HERE=Path(__file__).resolve().parent
FEATURES=NUMERIC_FEATURES+CATEGORICAL_FEATURES
SEEDS=[42,123,2026]
REFERENCE_DATE='2026-10-20'
TOP_FRACTIONS=[.01,.05,.10,.20]
META=['credit_code_normalized','company_name','record_id','source_record_number','quality_effective',
      'source','region_province','region_city','region_group','p_quality','industry_category_clean']

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def code_hash(values):return hashlib.sha256('\n'.join(sorted(map(str,values))).encode()).hexdigest()
def dump(value,p):Path(p).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def save(df,p):
    assert not df.columns.duplicated().any()
    df.to_csv(p,index=False,encoding='utf-8-sig')
def versions():return {'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__}
def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
def read_features(p):
    header=pd.read_csv(p,nrows=0).columns
    types={c:str for c in header if c not in NUMERIC_FEATURES}
    return pd.read_csv(p,dtype=types).reset_index(drop=True)
def validate_entities(df,label):
    codes=df.credit_code_normalized
    if codes.isna().any() or codes.duplicated().any() or not codes.map(_valid_uscc).all():
        raise ValueError(f'{label}: nonunique, missing or invalid codes')
def output_hashes(out):
    return {str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='output_hashes.json'}
def verify_hashes(root):
    for p,digest in json.loads((root/'output_hashes.json').read_text()).items():
        if sha(root/p)!=digest:raise ValueError(f'Hash mismatch: {root/p}')

def prepare(project,upstream,out):
    if out.exists():raise FileExistsError(out)
    verify_hashes(upstream)
    frozen=json.loads((project/'docs/data_versions/v2_raw_001.json').read_text())
    input_records=[];cohorts={};exclusions=[]
    for label,original in [('P_national','P_National'),('U2','U2')]:
        f=next(x for x in frozen['files'] if x['dataset']==original)
        rawpath=project/f['registered_path']
        if sha(rawpath)!=f['sha256']:raise ValueError(f'Frozen input mismatch: {rawpath}')
        raw=pd.read_csv(rawpath,dtype=str,keep_default_na=False)
        if len(raw)!=f['data_rows'] or len(raw.columns)!=f['columns']:raise ValueError('Frozen shape mismatch')
        eligiblepath=upstream/'eligible'/f'{label}.csv'
        eligible=pd.read_csv(eligiblepath,dtype=str,keep_default_na=False)
        ids=eligible.source_record_number.astype(int).to_numpy()-1
        selected=raw.iloc[ids].copy().reset_index(drop=True)
        actual=_normalise_credit_code(selected.credit_code).astype(str).tolist()
        assert actual==eligible.credit_code_normalized.tolist()
        assert selected.company_name.tolist()==eligible.company_name.tolist()
        selected['record_id']=eligible.record_id
        selected['source_record_number']=eligible.source_record_number
        selected['quality_effective']=eligible.p_quality if label=='P_national' else 'U'
        if label=='P_national':
            expected=np.where(selected.p_quality.eq('Gold')|selected.source.str.contains('量化Q3榜单',regex=False),'Gold','Silver')
            assert selected.quality_effective.tolist()==list(expected)
        selected['region_group']=np.select([selected.is_shanghai.eq('1'),selected.is_suzhou.eq('1')],['Shanghai','Suzhou'],default='Other')
        frame=build_features(selected,reference_date=pd.Timestamp(REFERENCE_DATE))
        mask=frame.business_status_clean.eq('异常/终止')
        excluded=frame.loc[mask,['record_id','credit_code_normalized','company_name','business_status','quality_effective']].copy()
        excluded.insert(0,'dataset',label);excluded['exclusion_reason']='legacy_abnormal_business_status'
        exclusions.append(excluded)
        frame=frame.loc[~mask].copy().sort_values('credit_code_normalized').reset_index(drop=True)
        validate_entities(frame,label)
        cohorts[label]=frame.reindex(columns=list(dict.fromkeys(META+FEATURES)))
        input_records.append({'dataset':label,'raw_path':str(rawpath),'raw_sha256':f['sha256'],'upstream_path':str(eligiblepath),
          'upstream_sha256':sha(eligiblepath),'upstream_rows':len(eligible),'excluded_business_status':int(mask.sum()),'final_rows':len(frame)})
    p,u=cohorts['P_national'],cohorts['U2']
    all_raw_p=pd.read_csv(project/next(x['registered_path'] for x in frozen['files'] if x['dataset']=='P_National'),dtype=str,keep_default_na=False)
    all_known=set(_normalise_credit_code(all_raw_p.credit_code).dropna())
    assert not set(u.credit_code_normalized)&all_known
    gold=p[p.quality_effective.eq('Gold')].reset_index(drop=True)
    silver=p[p.quality_effective.eq('Silver')].reset_index(drop=True)
    assert len(gold)==3105 and len(silver)==30851 and len(u)==18790
    strata=gold.region_group.fillna('Other')+'|'+gold.company_size_clean.fillna('Unknown')
    counts=strata.value_counts();strata=strata.where(strata.map(counts)>=10,gold.region_group+'|pooled_size')
    assert strata.value_counts().min()>=5
    development,held=train_test_split(np.arange(len(gold)),test_size=.2,random_state=42,stratify=strata)
    dev=gold.iloc[development].reset_index(drop=True);hold=gold.iloc[held].reset_index(drop=True)
    dev_strata=strata.iloc[development].reset_index(drop=True)
    fold_ids=np.zeros(len(dev),dtype=int)
    for fold,(_,val) in enumerate(StratifiedKFold(5,shuffle=True,random_state=42).split(dev,dev_strata),1):fold_ids[val]=fold
    ufit=stratified_subset(u,max(1,int(.4*len(u))),42).reset_index(drop=True)
    ueval=u.loc[~u.credit_code_normalized.isin(set(ufit.credit_code_normalized))].reset_index(drop=True)
    assert not set(ufit.credit_code_normalized)&set(ueval.credit_code_normalized)
    assert not set(dev.credit_code_normalized)&set(hold.credit_code_normalized)
    out.mkdir(parents=True)
    for name,df in [('gold_development',dev),('gold_final_holdout',hold),('silver',silver),('U_fit_pool',ufit),('U_evaluation',ueval)]:save(df,out/f'{name}.csv')
    goldsplit=gold[['credit_code_normalized','company_name','quality_effective','region_group']].copy()
    goldsplit['role']=np.where(gold.credit_code_normalized.isin(set(hold.credit_code_normalized)),'final_holdout','development')
    goldsplit['development_fold']=gold.credit_code_normalized.map(dict(zip(dev.credit_code_normalized,fold_ids))).fillna(0).astype(int)
    save(goldsplit,out/'gold_entity_split.csv')
    usplit=u[['credit_code_normalized','company_name','region_group']].copy()
    usplit['role']=np.where(u.credit_code_normalized.isin(set(ufit.credit_code_normalized)),'fit_pool','evaluation_only')
    save(usplit,out/'U_entity_split.csv');save(pd.concat(exclusions,ignore_index=True),out/'status_exclusions.csv')
    checks=[{'check':'original_raw_input_hashes','passed':True},{'check':'entity_uniqueness_and_checksum','passed':True},
      {'check':'U_disjoint_from_all_raw_P','passed':True},{'check':'Gold_holdout_disjoint_from_development_and_silver','passed':not bool(set(hold.credit_code_normalized)&(set(dev.credit_code_normalized)|set(silver.credit_code_normalized)))},
      {'check':'U_fit_and_evaluation_disjoint','passed':True},{'check':'Gold_holdout_size','passed':len(hold)==621},
      {'check':'every_development_Gold_has_one_fold','passed':bool((fold_ids>0).all())}]
    assert all(x['passed'] for x in checks);save(pd.DataFrame(checks),out/'preparation_checks.csv')
    dump({'created_at':now(),'reference_date':REFERENCE_DATE,'source_version':'v2_raw_001','input_files':input_records,
      'upstream_hash_manifest':sha(upstream/'output_hashes.json'),'code_sha256':sha(Path(__file__)),'legacy_code':json.loads((HERE/'legacy/source_manifest.json').read_text()),
      'versions':versions(),'counts':{'Gold':len(gold),'Gold_development':len(dev),'Gold_final_holdout':len(hold),'Silver':len(silver),'U2':len(u),'U_fit_pool':len(ufit),'U_evaluation':len(ueval)},
      'split_seed':42,'development_folds':5,'holdout_policy':'reserved; no training/prediction/model selection in E02',
      'split_strata':'SH/SZ/Other crossed with legacy size; rare size cells pooled within region','legacy_features':FEATURES},out/'manifest.json')
    dump(output_hashes(out),out/'output_hashes.json')
    print(json.dumps({'prepared':str(out),'counts':json.loads((out/'manifest.json').read_text())['counts']},ensure_ascii=False),flush=True)

def common_features(gold,mixed,ufit):
    a=feature_eligibility(gold,ufit).rename(columns={'eligible':'eligible_A','reason':'reason_A'})
    b=feature_eligibility(mixed,ufit).rename(columns={'eligible':'eligible_B','reason':'reason_B'})
    joined=a.merge(b,on=['feature','type'],suffixes=('_A','_B'),validate='one_to_one')
    joined['eligible_common']=joined.eligible_A&joined.eligible_B
    spec=FeatureSpec(joined.loc[joined.eligible_common&joined.type.eq('numeric'),'feature'].tolist(),
                     joined.loc[joined.eligible_common&joined.type.eq('categorical'),'feature'].tolist())
    assert spec.all and set(spec.all)<=set(FEATURES)
    return spec,joined

def training_weights(n_gold,n_positive,n_u):
    # For both arms total P weight = total U weight = n_gold.
    if min(n_gold,n_positive,n_u)<=0:raise ValueError('empty training class')
    return np.r_[np.full(n_positive,n_gold/n_positive),np.full(n_u,n_gold/n_u)]

def fit_arm(positive,sampled_u,targets,spec,n_gold,seed):
    frame=pd.concat([positive,sampled_u],ignore_index=True)
    y=np.r_[np.ones(len(positive)),np.zeros(len(sampled_u))]
    weights=training_weights(n_gold,len(positive),len(sampled_u))
    pipeline=Pipeline([('preprocessor',make_preprocessor(spec)),('model',LogisticRegression(
      solver='lbfgs',C=1.0,max_iter=1000,tol=1e-4,class_weight=None,random_state=seed))])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always',ConvergenceWarning)
        pipeline.fit(frame[spec.all],y,model__sample_weight=weights)
    converged=not any(issubclass(w.category,ConvergenceWarning) for w in caught)
    if not converged:raise RuntimeError(f'LR failed to converge at seed {seed}; stop instead of silently comparing unfinished fits')
    prediction=pipeline.predict_proba(targets[spec.all])[:,1]
    assert np.isfinite(prediction).all()
    transformed_columns=len(pipeline.named_steps['preprocessor'].get_feature_names_out())
    return prediction,{'positive_rows':len(positive),'U_rows':len(sampled_u),'positive_weight_sum':float(weights[:len(positive)].sum()),
      'U_weight_sum':float(weights[len(positive):].sum()),'total_weight':float(weights.sum()),
      'max_n_iter':int(pipeline.named_steps['model'].n_iter_.max()),'converged':converged,'encoded_columns':transformed_columns},pipeline

def rank_metrics(predictions,fractions=TOP_FRACTIONS):
    ranked=predictions.sort_values(['score','credit_code_normalized'],ascending=[False,True],kind='stable').copy()
    ranked['rank']=np.arange(1,len(ranked)+1)
    gold_n=int(ranked.kind.eq('Gold_validation').sum())
    metrics=[]
    for fraction in fractions:
        k=max(1,int(math.ceil(len(ranked)*fraction)))
        hit=int(ranked.iloc[:k].kind.eq('Gold_validation').sum())
        metrics.append({'top_fraction':fraction,'top_k':k,'evaluation_rows':len(ranked),'validation_Gold':gold_n,
          'captured_Gold':hit,'capture_rate':hit/gold_n if gold_n else 0,'random_reference':k/len(ranked)})
    return metrics,ranked

def train(prepared,out,smoke=False):
    if out.exists():raise FileExistsError(out)
    verify_hashes(prepared)
    pm=json.loads((prepared/'manifest.json').read_text())
    # Do not read gold_final_holdout.csv; entity IDs suffice for leakage checks.
    gold=read_features(prepared/'gold_development.csv');silver=read_features(prepared/'silver.csv')
    ufit=read_features(prepared/'U_fit_pool.csv');ueval=read_features(prepared/'U_evaluation.csv')
    splits=pd.read_csv(prepared/'gold_entity_split.csv',dtype={'credit_code_normalized':str})
    foldmap=splits.set_index('credit_code_normalized').development_fold
    reserved=set(splits.loc[splits.role.eq('final_holdout'),'credit_code_normalized'])
    assert not reserved&(set(gold.credit_code_normalized)|set(silver.credit_code_normalized)|set(ufit.credit_code_normalized)|set(ueval.credit_code_normalized))
    folds=gold.credit_code_normalized.map(foldmap).astype(int)
    out.mkdir(parents=True);(out/'predictions').mkdir();(out/'feature_audits').mkdir()
    seeds=SEEDS[:1] if smoke else SEEDS;repeats=1 if smoke else 10;fold_range=[1] if smoke else list(range(1,6))
    protocol={'status':'running','started_at':now(),'stage':'smoke' if smoke else 'full_development',
      'prepared_manifest_sha256':sha(prepared/'manifest.json'),'prepared_output_hashes_sha256':sha(prepared/'output_hashes.json'),
      'code_sha256':sha(Path(__file__)),'versions':versions(),'seeds':seeds,'repeats':repeats,'folds':fold_range,
      'model':{'family':'LogisticRegression','solver':'lbfgs','C':1.0,'max_iter':1000,'tol':1e-4,'class_weight':None},
      'comparison':'same Gold, fixed U sample per repeat, same feature fields per fold; B adds Silver',
      'weights':'each P in arm gets n_Gold/n_P; each sampled U gets n_Gold/n_U; class totals and regularization scale identical',
      'feature_rule':'intersection of training-only legacy eligibility for both arms against fixed U fit pool',
      'holdout_scored':False,'ranking':'descending score, ascending credit code ties; ceil(total*fraction)',
      'U_scope':'independent U evaluation pool only; no production full-U scores or Gold final-holdout scores',
      'selection':'development CV only; do not claim final validation or business precision'}
    dump(protocol,out/'manifest.json')
    fits=[];metrics=[];predictions=[];pairchecks=[];sample_records=[]
    try:
      with threadpool_limits(limits=1):
       for fold in fold_range:
        ptrain=gold.loc[folds.ne(fold)].reset_index(drop=True);gval=gold.loc[folds.eq(fold)].reset_index(drop=True)
        mixed=pd.concat([ptrain,silver],ignore_index=True)
        assert not set(gval.credit_code_normalized)&(set(mixed.credit_code_normalized)|reserved)
        spec,audit=common_features(ptrain,mixed,ufit)
        save(audit,out/'feature_audits'/f'fold_{fold}.csv')
        target=pd.concat([ueval,gval],ignore_index=True)
        kind=np.r_[np.repeat('U_evaluation',len(ueval)),np.repeat('Gold_validation',len(gval))]
        for seed in seeds:
         started=time.monotonic();byarm={'A_Gold_only':[],'B_Gold_Silver':[]}
         for repeat in range(repeats):
          sample_seed=seed+fold*10000+repeat
          sampled=stratified_subset(ufit,len(ptrain),sample_seed).reset_index(drop=True)
          sampled_codes=set(sampled.credit_code_normalized)
          assert not sampled_codes&(set(target.credit_code_normalized)|reserved)
          uhash=code_hash(sampled.credit_code_normalized)
          sample_records.extend({'fold':fold,'seed':seed,'repeat':repeat+1,'credit_code_normalized':c} for c in sampled.credit_code_normalized)
          pair=[]
          for arm,positive in [('A_Gold_only',ptrain),('B_Gold_Silver',mixed)]:
           fit_started=time.monotonic()
           scores,info,model=fit_arm(positive,sampled,target,spec,len(ptrain),sample_seed)
           byarm[arm].append(scores)
           fits.append({'arm':arm,'fold':fold,'seed':seed,'repeat':repeat+1,'sample_seed':sample_seed,
             'U_sample_hash':uhash,'feature_fields':'|'.join(spec.all),'positive_entity_hash':code_hash(positive.credit_code_normalized),
             'target_entity_hash':code_hash(target.credit_code_normalized),'fit_seconds':time.monotonic()-fit_started,**info})
           pair.append(info)
          pairchecks.append({'fold':fold,'seed':seed,'repeat':repeat+1,'same_U_entities':True,'same_feature_fields':True,
             'same_targets':True,'equal_total_weight':bool(np.isclose(pair[0]['total_weight'],pair[1]['total_weight'])),
             'class_weight_balanced_both':all(np.isclose(p['positive_weight_sum'],p['U_weight_sum']) for p in pair),
             'training_target_and_final_holdout_disjoint':True})
         for arm,values in byarm.items():
          matrix=np.vstack(values)
          result=target[['credit_code_normalized','company_name','region_group','industry_major_category_clean','company_size_clean','source']].copy()
          result['kind']=kind;result['arm']=arm;result['fold']=fold;result['seed']=seed
          result['score']=matrix.mean(axis=0);result['repeat_score_std']=matrix.std(axis=0)
          metric,ranked=rank_metrics(result)
          metrics.extend({'arm':arm,'fold':fold,'seed':seed,**m} for m in metric)
          save(ranked,out/'predictions'/f'fold_{fold}_seed_{seed}_{arm}.csv')
          predictions.append(ranked)
         save(pd.DataFrame(fits),out/'fit_log.csv');save(pd.DataFrame(metrics),out/'cv_metrics.csv')
         print(json.dumps({'fold':fold,'seed':seed,'repeats':repeats,'features':spec.all,'elapsed_seconds':round(time.monotonic()-started,2),
           'development_Gold_validation':len(gval),'progress_pairs':len(pairchecks),'expected_pairs':len(fold_range)*len(seeds)*repeats}),flush=True)
      allpred=pd.concat(predictions,ignore_index=True)
      save(pd.DataFrame(pairchecks),out/'paired_integrity_checks.csv');save(pd.DataFrame(sample_records),out/'paired_U_samples.csv')
      checks=pd.DataFrame(pairchecks)
      checkcolumns=['same_U_entities','same_feature_fields','same_targets','equal_total_weight','class_weight_balanced_both','training_target_and_final_holdout_disjoint']
      assert checks[checkcolumns].all().all()
      assert len(fits)==2*len(fold_range)*len(seeds)*repeats
      assert not set(allpred.credit_code_normalized)&reserved
      protocol.update(status='complete',completed_at=now(),fits=len(fits),paired_checks=len(pairchecks),
          validation_Gold_scored=int(allpred.loc[allpred.kind.eq('Gold_validation'),'credit_code_normalized'].nunique()),
          unique_U_evaluation_scored=len(ueval),final_Gold_holdout_scored=False)
      dump(protocol,out/'manifest.json');dump(output_hashes(out),out/'output_hashes.json')
      print(json.dumps({'run':str(out),'status':'complete','fits':len(fits),'final_holdout_scored':False}),flush=True)
    except BaseException as exc:
      protocol.update(status='failed',error=repr(exc),failed_at=now())
      dump(protocol,out/'manifest.json')
      if fits:save(pd.DataFrame(fits),out/'fit_log.csv')
      raise

if __name__=='__main__':
    ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--project',type=Path,required=True);p.add_argument('--upstream',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    t=sub.add_parser('train');t.add_argument('--prepared',type=Path,required=True);t.add_argument('--out',type=Path,required=True);t.add_argument('--smoke',action='store_true')
    args=ap.parse_args()
    if args.command=='prepare':prepare(args.project.resolve(),args.upstream.resolve(),args.out.resolve())
    else:train(args.prepared.resolve(),args.out.resolve(),args.smoke)
