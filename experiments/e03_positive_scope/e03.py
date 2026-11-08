"""E03: local/national positive scope crossed with Gold-only/Gold+Silver.
Reuses E02 development folds, its reserved Gold IDs, and U2 pool assignment.
Only SH/SZ development Gold and U1 are scored. No final Gold is scored.
"""
from pathlib import Path
import argparse,json,time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from e02_support import (FEATURES,SEEDS,META,REFERENCE_DATE,sha,code_hash,dump,save,versions,now,
                         output_hashes,verify_hashes,read_features,validate_entities,fit_arm,rank_metrics)
from legacy.src.config import NUMERIC_FEATURES,CATEGORICAL_FEATURES
from legacy.src.feature_engineering import build_features
from legacy.src.data_cleaning import _normalise_credit_code
from legacy.src.model_logistic_baseline import stratified_subset
from legacy.src.preprocessing import feature_eligibility,FeatureSpec

HERE=Path(__file__).resolve().parent
ARMS=['A_local_Gold','B_local_mixed','C_national_Gold','D_national_mixed']
PAIRS=[('C_national_Gold','A_local_Gold','C_minus_A'),('D_national_mixed','B_local_mixed','D_minus_B')]

def is_local(frame):return frame.region_group.isin(['Shanghai','Suzhou'])

def make_arms(gold,silver,foldmap,fold):
    # Exclude all national validation Gold for the current fold, even though
    # E03 only scores the SH/SZ subset. This preserves the E02 fold boundary.
    gtrain=gold.loc[gold.credit_code_normalized.map(foldmap).ne(fold)].reset_index(drop=True)
    local=gtrain.loc[is_local(gtrain)].reset_index(drop=True)
    s_local=silver.loc[is_local(silver)].reset_index(drop=True)
    return {'A_local_Gold':local,'B_local_mixed':pd.concat([local,s_local],ignore_index=True),
            'C_national_Gold':gtrain,'D_national_mixed':pd.concat([gtrain,silver],ignore_index=True)}

def common_features(arms,ufit):
    audits=[]
    for arm,frame in arms.items():
        audit=feature_eligibility(frame,ufit);audit.insert(0,'arm',arm);audits.append(audit)
    audit=pd.concat(audits,ignore_index=True)
    common=audit.groupby('feature').eligible.all()
    audit['eligible_common']=audit.feature.map(common)
    spec=FeatureSpec([c for c in NUMERIC_FEATURES if common[c]],[c for c in CATEGORICAL_FEATURES if common[c]])
    if not spec.all:raise ValueError('No common usable fields')
    return spec,audit

def prepare(project,e02prep,upstream,out):
    if out.exists():raise FileExistsError(out)
    verify_hashes(e02prep);verify_hashes(upstream)
    pm=json.loads((e02prep/'manifest.json').read_text())
    for entry in pm['input_files']:
        if sha(entry['raw_path'])!=entry['raw_sha256']:raise ValueError('Frozen P/U2 hash changed')
    frozen=json.loads((project/'docs/data_versions/v2_raw_001.json').read_text())
    uentry=next(x for x in frozen['files'] if x['dataset']=='U1')
    rawpath=project/uentry['registered_path'];assert sha(rawpath)==uentry['sha256']
    raw=pd.read_csv(rawpath,dtype=str,keep_default_na=False)
    eligiblepath=upstream/'eligible/U1.csv';eligible=pd.read_csv(eligiblepath,dtype=str,keep_default_na=False)
    selected=raw.iloc[eligible.source_record_number.astype(int).to_numpy()-1].copy().reset_index(drop=True)
    assert _normalise_credit_code(selected.credit_code).astype(str).tolist()==eligible.credit_code_normalized.tolist()
    assert selected.company_name.tolist()==eligible.company_name.tolist()
    selected['record_id']=eligible.record_id;selected['source_record_number']=eligible.source_record_number
    selected['quality_effective']='U'
    selected['region_group']=np.select([selected.is_shanghai.eq('1'),selected.is_suzhou.eq('1')],['Shanghai','Suzhou'],default='Other')
    assert selected.region_group.ne('Other').all()
    u=build_features(selected,reference_date=pd.Timestamp(REFERENCE_DATE))
    mask=u.business_status_clean.eq('异常/终止')
    excluded=u.loc[mask,['credit_code_normalized','company_name','business_status','record_id']].copy()
    excluded['reason']='legacy_abnormal_business_status'
    u=u.loc[~mask].reindex(columns=list(dict.fromkeys(META+FEATURES))).sort_values('credit_code_normalized').reset_index(drop=True)
    gold=read_features(e02prep/'gold_development.csv');silver=read_features(e02prep/'silver.csv')
    split=pd.read_csv(e02prep/'gold_entity_split.csv',dtype={'credit_code_normalized':str})
    usplit=pd.read_csv(e02prep/'U_entity_split.csv',dtype={'credit_code_normalized':str})
    role=usplit.set_index('credit_code_normalized').role
    assert set(u.credit_code_normalized)<=set(role.index)
    ufit=u.loc[u.credit_code_normalized.map(role).eq('fit_pool')].reset_index(drop=True)
    ueval=u.loc[u.credit_code_normalized.map(role).eq('evaluation_only')].reset_index(drop=True)
    reserved=set(split.loc[split.role.eq('final_holdout'),'credit_code_normalized'])
    for label,df in [('gold',gold),('silver',silver),('U1',u)]:validate_entities(df,label)
    assert not reserved&(set(gold.credit_code_normalized)|set(silver.credit_code_normalized)|set(u.credit_code_normalized))
    assert not set(u.credit_code_normalized)&(set(gold.credit_code_normalized)|set(silver.credit_code_normalized))
    assert len(u)==11013 and len(ufit)==4353 and len(ueval)==6660
    assert is_local(gold).sum()==1163 and is_local(silver).sum()==9126
    # Raw U1 attributes are authoritative for E03. Diagnose equality to E02 U2
    # copies rather than assuming a code-subset implies identical attributes.
    old_u=pd.concat([read_features(e02prep/'U_fit_pool.csv'),read_features(e02prep/'U_evaluation.csv')],ignore_index=True)
    old_u=old_u.set_index('credit_code_normalized').loc[u.credit_code_normalized]
    equality=[]
    for field in FEATURES:
        if field in NUMERIC_FEATURES:
            equal=np.isclose(pd.to_numeric(old_u[field],errors='coerce').to_numpy(),pd.to_numeric(u[field],errors='coerce').to_numpy(),equal_nan=True)
        else:equal=old_u[field].fillna('<missing>').to_numpy()==u[field].fillna('<missing>').to_numpy()
        equality.append({'field':field,'different_U1_U2_rows':int((~equal).sum())})
    out.mkdir(parents=True)
    for name,df in [('gold_development',gold),('silver',silver),('U1_fit_pool',ufit),('U1_evaluation',ueval),('gold_entity_split',split)]:save(df,out/f'{name}.csv')
    save(excluded,out/'status_exclusions.csv');save(pd.DataFrame(equality),out/'U1_U2_feature_equality.csv')
    localfold=gold.loc[is_local(gold),['credit_code_normalized','company_name','region_group']].copy()
    localfold['fold']=localfold.credit_code_normalized.map(split.set_index('credit_code_normalized').development_fold)
    save(localfold,out/'local_Gold_development_folds.csv')
    us=u[['credit_code_normalized','company_name','region_group']].copy();us['role']=us.credit_code_normalized.map(role)
    save(us,out/'U1_entity_split.csv')
    foldmap=split.set_index('credit_code_normalized').development_fold
    populations=[]
    for fold in range(1,6):
        for arm,frame in make_arms(gold,silver,foldmap,fold).items():
            goldn=int(frame.quality_effective.eq('Gold').sum());silvern=int(frame.quality_effective.eq('Silver').sum())
            populations.append({'fold':fold,'arm':arm,'positive_rows':len(frame),'Gold_rows':goldn,'Silver_rows':silvern,
              'Gold_share_within_positive_weight':goldn/len(frame),'local_rows':int(is_local(frame).sum()),'nonlocal_rows':int((~is_local(frame)).sum()),
              'positive_entity_hash':code_hash(frame.credit_code_normalized)})
    save(pd.DataFrame(populations),out/'arm_training_populations.csv')
    checks=[{'check':'parent_E02_hashes_and_frozen_P_U2','passed':True},{'check':'frozen_U1_and_eligibility_identity','passed':True},
      {'check':'U1_inherits_existing_U2_roles','passed':True},{'check':'all_621_final_Gold_excluded','passed':True},
      {'check':'U1_P_disjoint_and_codes_unique_valid','passed':True},{'check':'U1_fit_evaluation_disjoint','passed':not bool(set(ufit.credit_code_normalized)&set(ueval.credit_code_normalized))}]
    assert all(x['passed'] for x in checks);save(pd.DataFrame(checks),out/'preparation_checks.csv')
    dump({'created_at':now(),'source_version':'v2_raw_001','reference_date':REFERENCE_DATE,'parent_E02_prepared':str(e02prep),
      'parent_E02_manifest_sha256':sha(e02prep/'manifest.json'),'parent_E02_output_hashes_sha256':sha(e02prep/'output_hashes.json'),
      'frozen_U1_path':str(rawpath),'frozen_U1_sha256':uentry['sha256'],'upstream_U1_eligibility_sha256':sha(eligiblepath),
      'code_sha256':sha(Path(__file__)),'versions':versions(),'inherited_fold_and_U_roles':True,
      'counts':{'national_development_Gold':len(gold),'local_development_Gold':int(is_local(gold).sum()),'national_Silver':len(silver),'local_Silver':int(is_local(silver).sum()),
        'U1':len(u),'U1_fit_pool':len(ufit),'U1_evaluation':len(ueval),'reserved_final_Gold':len(reserved)},
      'holdout_policy':'no final Gold feature file copied or loaded; IDs retained only for exclusion',
      'scope':'SH/SZ defined by original freeze flags via E02 region_group; never a predictive feature'},out/'manifest.json')
    dump(output_hashes(out),out/'output_hashes.json')
    print(json.dumps({'prepared':str(out),'counts':json.loads((out/'manifest.json').read_text())['counts']},ensure_ascii=False),flush=True)

def train(prepared,out,smoke=False):
    if out.exists():raise FileExistsError(out)
    verify_hashes(prepared)
    gold=read_features(prepared/'gold_development.csv');silver=read_features(prepared/'silver.csv')
    ufit=read_features(prepared/'U1_fit_pool.csv');ueval=read_features(prepared/'U1_evaluation.csv')
    splits=pd.read_csv(prepared/'gold_entity_split.csv',dtype={'credit_code_normalized':str})
    foldmap=splits.set_index('credit_code_normalized').development_fold
    reserved=set(splits.loc[splits.role.eq('final_holdout'),'credit_code_normalized'])
    out.mkdir(parents=True);(out/'predictions').mkdir();(out/'feature_audits').mkdir()
    seeds=SEEDS[:1] if smoke else SEEDS;repeats=1 if smoke else 10;folds=[1] if smoke else list(range(1,6))
    protocol={'started_at':now(),'status':'running','stage':'smoke' if smoke else 'full_development',
      'prepared_manifest_sha256':sha(prepared/'manifest.json'),'code_sha256':sha(Path(__file__)),'support_code_sha256':sha(HERE/'e02_support.py'),
      'versions':versions(),'seeds':seeds,'repeats':repeats,'folds':folds,'arms':ARMS,'primary_contrasts':['C_minus_A','D_minus_B'],
      'U_sampling':'same U1 sample for all four arms; size = number of local training Gold',
      'weights':'P total = U total = local training Gold count; all four arms same total; within each P each entity equal weight',
      'feature_rule':'four-way intersection of training-only legacy eligibility against U1 fit pool',
      'parameters':{'solver':'lbfgs','C':1.0,'tol':1e-4,'max_iter':1000,'class_weight':None},
      'scope':'same-source SH/SZ development Gold ranking in same U1 panel; not population business accuracy',
      'confounding_note':'national expansion adds sample size, source/industry composition; mixed arms also change Gold/Silver weight shares; strategy effect, not pure causal geography effect',
      'final_Gold_holdout_scored':False,'new_geography_features':False}
    dump(protocol,out/'manifest.json');fits=[];metrics=[];samples=[];checks=[]
    try:
      with threadpool_limits(limits=1):
       for fold in folds:
        arms=make_arms(gold,silver,foldmap,fold)
        validation=gold.loc[is_local(gold)&gold.credit_code_normalized.map(foldmap).eq(fold)].reset_index(drop=True)
        excluded_fold=set(splits.loc[splits.development_fold.eq(fold),'credit_code_normalized'])
        assert all(not (set(frame.credit_code_normalized)&(excluded_fold|reserved|set(ueval.credit_code_normalized))) for frame in arms.values())
        spec,audit=common_features(arms,ufit);save(audit,out/'feature_audits'/f'fold_{fold}.csv')
        target=pd.concat([ueval,validation],ignore_index=True)
        kinds=np.r_[np.repeat('U_evaluation',len(ueval)),np.repeat('Gold_validation',len(validation))]
        m=len(arms['A_local_Gold'])
        for seed in seeds:
         started=time.monotonic();byarm={arm:[] for arm in ARMS}
         for repeat in range(repeats):
          sample_seed=seed+fold*10000+repeat
          u_sample=stratified_subset(ufit,m,sample_seed).reset_index(drop=True)
          ucodes=set(u_sample.credit_code_normalized)
          assert not ucodes&(set(target.credit_code_normalized)|reserved)
          samples.extend({'fold':fold,'seed':seed,'repeat':repeat+1,'credit_code_normalized':c} for c in u_sample.credit_code_normalized)
          info_rows=[]
          for arm,positive in arms.items():
           t=time.monotonic();scores,info,_=fit_arm(positive,u_sample,target,spec,m,sample_seed)
           byarm[arm].append(scores);info_rows.append(info)
           fits.append({'arm':arm,'fold':fold,'seed':seed,'repeat':repeat+1,'sample_seed':sample_seed,
             'U_sample_hash':code_hash(ucodes),'positive_entity_hash':code_hash(positive.credit_code_normalized),'target_entity_hash':code_hash(target.credit_code_normalized),
             'feature_fields':'|'.join(spec.all),'Gold_rows':int(positive.quality_effective.eq('Gold').sum()),'Silver_rows':int(positive.quality_effective.eq('Silver').sum()),
             'fit_seconds':time.monotonic()-t,**info})
          checks.append({'fold':fold,'seed':seed,'repeat':repeat+1,'same_U_entities_and_size':True,'same_feature_fields':True,'same_targets':True,
             'equal_total_and_class_weight':all(np.isclose(row['total_weight'],2*m) and np.isclose(row['positive_weight_sum'],row['U_weight_sum']) for row in info_rows),
             'current_fold_and_final_Gold_excluded':True,'local_Gold_is_subset_national_Gold':set(arms['A_local_Gold'].credit_code_normalized)<=set(arms['C_national_Gold'].credit_code_normalized)})
         for arm,values in byarm.items():
          matrix=np.vstack(values)
          result=target[['credit_code_normalized','company_name','region_group','industry_major_category_clean','company_size_clean','source']].copy()
          result['kind']=kinds;result['arm']=arm;result['fold']=fold;result['seed']=seed;result['score']=matrix.mean(axis=0);result['repeat_score_std']=matrix.std(axis=0)
          measures,ranked=rank_metrics(result)
          metrics.extend({'arm':arm,'fold':fold,'seed':seed,**r} for r in measures)
          save(ranked,out/'predictions'/f'fold_{fold}_seed_{seed}_{arm}.csv')
         save(pd.DataFrame(fits),out/'fit_log.csv');save(pd.DataFrame(metrics),out/'cv_metrics.csv')
         print(json.dumps({'fold':fold,'seed':seed,'repeats':repeats,'local_Gold_validation':len(validation),'common_fields':spec.all,
            'completed_quadruples':len(checks),'expected_quadruples':len(folds)*len(seeds)*repeats,'elapsed_seconds':round(time.monotonic()-started,2)}),flush=True)
      assert len(fits)==4*len(folds)*len(seeds)*repeats
      assert pd.DataFrame(checks).drop(columns=['fold','seed','repeat']).all().all()
      save(pd.DataFrame(samples),out/'common_U_samples.csv');save(pd.DataFrame(checks),out/'four_arm_integrity_checks.csv')
      protocol.update(status='complete',completed_at=now(),fits=len(fits),matched_quadruples=len(checks))
      dump(protocol,out/'manifest.json');dump(output_hashes(out),out/'output_hashes.json')
      print(json.dumps({'status':'complete','fits':len(fits),'run':str(out),'final_Gold_scored':False}),flush=True)
    except BaseException as exc:
      protocol.update(status='failed',error=repr(exc),failed_at=now());dump(protocol,out/'manifest.json')
      if fits:save(pd.DataFrame(fits),out/'fit_log.csv')
      raise

if __name__=='__main__':
    ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--project',type=Path,required=True);p.add_argument('--e02-prepared',type=Path,required=True);p.add_argument('--upstream',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    t=sub.add_parser('train');t.add_argument('--prepared',type=Path,required=True);t.add_argument('--out',type=Path,required=True);t.add_argument('--smoke',action='store_true')
    args=ap.parse_args()
    if args.command=='prepare':prepare(args.project.resolve(),args.e02_prepared.resolve(),args.upstream.resolve(),args.out.resolve())
    else:train(args.prepared.resolve(),args.out.resolve(),args.smoke)
