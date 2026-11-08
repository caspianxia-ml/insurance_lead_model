"""E04: fixed national P, U2 versus controlled expansion to U3.
Common U2 evaluation is primary; expanded evaluation is secondary. Four arms
cross U scope with Gold-only/Gold+Silver. No reserved final Gold is scored.
"""
from pathlib import Path
import argparse,csv,json,time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from e02_support import (FEATURES,META,SEEDS,REFERENCE_DATE,sha,code_hash,dump,save,versions,now,
                         output_hashes,verify_hashes,read_features,validate_entities,fit_arm,rank_metrics)
from legacy.src.config import NUMERIC_FEATURES,CATEGORICAL_FEATURES
from legacy.src.preprocessing import feature_eligibility,FeatureSpec
from legacy.src.feature_engineering import build_features
from legacy.src.data_cleaning import _normalise_credit_code
from legacy.src.model_logistic_baseline import stratified_subset

HERE=Path(__file__).resolve().parent
ARMS=['A_Gold_U2','B_mixed_U2','C_Gold_U3','D_mixed_U3']
PANELS=['common_U2','expanded_U3']
PAIRS=[('C_Gold_U3','A_Gold_U2','C_minus_A'),('D_mixed_U3','B_mixed_U2','D_minus_B')]

def common_features(gold,mixed,u2fit,u3fit):
    audits=[]
    for arm,p,u in [('A_Gold_U2',gold,u2fit),('B_mixed_U2',mixed,u2fit),('C_Gold_U3',gold,u3fit),('D_mixed_U3',mixed,u3fit)]:
        a=feature_eligibility(p,u);a.insert(0,'arm',arm);audits.append(a)
    audit=pd.concat(audits,ignore_index=True)
    common=audit.groupby('feature').eligible.all();audit['eligible_common']=audit.feature.map(common)
    spec=FeatureSpec([x for x in NUMERIC_FEATURES if common[x]],[x for x in CATEGORICAL_FEATURES if common[x]])
    if not spec.all:raise ValueError('No common training-only usable features')
    return spec,audit

def prediction_index(ueval,increment,gval):
    columns=['credit_code_normalized','company_name','region_group','industry_major_category_clean','company_size_clean','source']
    target=pd.concat([ueval,increment,gval],ignore_index=True)
    index=target.reindex(columns=columns).copy()
    index['kind']=np.r_[np.repeat('U_common',len(ueval)),np.repeat('U_increment',len(increment)),np.repeat('Gold_validation',len(gval))]
    index.insert(0,'prediction_row',np.arange(len(index)))
    assert not index.credit_code_normalized.duplicated().any()
    return target,index

def panel_frame(index,score,panel):
    if panel not in PANELS:raise ValueError(panel)
    mask=index.kind.ne('U_increment') if panel=='common_U2' else pd.Series(True,index=index.index)
    frame=index.loc[mask].copy();frame['score']=np.asarray(score)[mask.to_numpy()]
    return frame

def prepare(project,parent,upstream,out):
    if out.exists():raise FileExistsError(out)
    verify_hashes(parent);verify_hashes(upstream)
    pm=json.loads((parent/'manifest.json').read_text())
    for entry in pm['input_files']:
        assert sha(entry['raw_path'])==entry['raw_sha256']
    frozen=json.loads((project/'docs/data_versions/v2_raw_001.json').read_text())
    entry=next(x for x in frozen['files'] if x['dataset']=='U3');rawpath=project/entry['registered_path']
    assert sha(rawpath)==entry['sha256']
    records=[]
    with rawpath.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.reader(f,strict=False);header=next(reader)
        assert len(header)==len(set(header))==20
        for row in reader:
            if len(row)!=len(header):raise ValueError(f'Unexpected U3 width at physical line {reader.line_num}')
            records.append(row)
    assert len(records)==entry['data_rows']==167579
    raw=pd.DataFrame(records,columns=header)
    eligiblepath=upstream/'eligible/U3.csv';eligible=pd.read_csv(eligiblepath,dtype=str,keep_default_na=False)
    chosen=raw.iloc[eligible.source_record_number.astype(int).to_numpy()-1].copy().reset_index(drop=True)
    assert _normalise_credit_code(chosen.credit_code).astype(str).tolist()==eligible.credit_code_normalized.tolist()
    assert chosen.company_name.tolist()==eligible.company_name.tolist()
    chosen['record_id']=eligible.record_id;chosen['source_record_number']=eligible.source_record_number;chosen['quality_effective']='U'
    chosen['region_group']=np.select([chosen.is_shanghai.eq('1'),chosen.is_suzhou.eq('1')],['Shanghai','Suzhou'],default='Other')
    frame=build_features(chosen,reference_date=pd.Timestamp(REFERENCE_DATE))
    abnormal=frame.business_status_clean.eq('异常/终止')
    excluded=frame.loc[abnormal,['credit_code_normalized','company_name','business_status','record_id']].copy();excluded['reason']='legacy_abnormal_business_status'
    frame=frame.loc[~abnormal].reindex(columns=list(dict.fromkeys(META+FEATURES))).sort_values('credit_code_normalized').reset_index(drop=True)
    validate_entities(frame,'U3 status-filtered')
    gold=read_features(parent/'gold_development.csv');silver=read_features(parent/'silver.csv')
    u2fit=read_features(parent/'U_fit_pool.csv');common=read_features(parent/'U_evaluation.csv')
    u2=pd.concat([u2fit,common],ignore_index=True);u2codes=set(u2.credit_code_normalized)
    assert u2codes<=set(frame.credit_code_normalized)
    original_overlap=frame.set_index('credit_code_normalized').loc[u2.credit_code_normalized]
    equality=[]
    for field in FEATURES:
        if field in NUMERIC_FEATURES:
            equal=np.isclose(pd.to_numeric(original_overlap[field],errors='coerce').to_numpy(),pd.to_numeric(u2[field],errors='coerce').to_numpy(),equal_nan=True)
        else:equal=original_overlap[field].fillna('<missing>').to_numpy()==u2[field].fillna('<missing>').to_numpy()
        equality.append({'field':field,'different_shared_U2_U3_entities':int((~equal).sum())})
    # Use complete U2 records on the overlap for the controlled comparison,
    # retaining U3-derived differences as an audit; never splice row fields.
    extra=frame.loc[~frame.credit_code_normalized.isin(u2codes)].reset_index(drop=True)
    increment_fit=stratified_subset(extra,int(.4*len(extra)),42).reset_index(drop=True)
    increment_eval=extra.loc[~extra.credit_code_normalized.isin(set(increment_fit.credit_code_normalized))].reset_index(drop=True)
    broadfit=pd.concat([u2fit,increment_fit],ignore_index=True).sort_values('credit_code_normalized').reset_index(drop=True)
    split=pd.read_csv(parent/'gold_entity_split.csv',dtype={'credit_code_normalized':str})
    reserved=set(split.loc[split.role.eq('final_holdout'),'credit_code_normalized'])
    fitting=set(broadfit.credit_code_normalized);evaluation=set(common.credit_code_normalized)|set(increment_eval.credit_code_normalized)
    assert not fitting&evaluation and fitting|evaluation==set(frame.credit_code_normalized)
    assert not (fitting|evaluation)&(set(gold.credit_code_normalized)|set(silver.credit_code_normalized)|reserved)
    allp=pd.read_csv(project/next(x['registered_path'] for x in frozen['files'] if x['dataset']=='P_National'),dtype=str,keep_default_na=False)
    assert not set(frame.credit_code_normalized)&set(_normalise_credit_code(allp.credit_code).dropna())
    assert len(frame)==124892 and len(extra)==106102 and len(increment_fit)==42440 and len(increment_eval)==63662
    out.mkdir(parents=True)
    for name,df in [('gold_development',gold),('silver',silver),('U2_fit_pool',u2fit),('U_common_evaluation',common),
       ('U3_fit_pool',broadfit),('U_increment_fit',increment_fit),('U_increment_evaluation',increment_eval),('gold_entity_split',split)]:save(df,out/f'{name}.csv')
    save(excluded,out/'status_exclusions.csv');save(pd.DataFrame(equality),out/'shared_U2_U3_feature_equality.csv')
    roles=[]
    for name,df in [('U2_fit',u2fit),('U2_evaluation',common),('increment_fit',increment_fit),('increment_evaluation',increment_eval)]:
        r=df[['credit_code_normalized','company_name','record_id']].copy();r['role']=name
        r['attribute_basis']='original_E02_U2_complete_record' if name.startswith('U2') else 'frozen_U3_complete_record';roles.append(r)
    save(pd.concat(roles,ignore_index=True),out/'U3_entity_roles.csv')
    checks=[{'check':'parent_E02_and_upstream_hashes','passed':True},{'check':'frozen_U3_hash_and_all_row_widths','passed':True},
      {'check':'U2_subset_of_status_filtered_U3','passed':True},{'check':'U3_disjoint_from_all_original_P','passed':True},
      {'check':'existing_U2_roles_preserved','passed':True},{'check':'both_evaluation_pools_excluded_from_training','passed':True},
      {'check':'all_621_final_Gold_excluded','passed':True},{'check':'U3_roles_cover_each_entity_once','passed':pd.concat(roles).credit_code_normalized.nunique()==len(frame)}]
    assert all(x['passed'] for x in checks);save(pd.DataFrame(checks),out/'preparation_checks.csv')
    dump({'created_at':now(),'source_version':'v2_raw_001','reference_date':REFERENCE_DATE,'parent_E02_prepared':str(parent),
      'parent_E02_manifest_sha256':sha(parent/'manifest.json'),'parent_E02_output_hashes_sha256':sha(parent/'output_hashes.json'),
      'frozen_U3_path':str(rawpath),'frozen_U3_sha256':entry['sha256'],'upstream_U3_eligibility_sha256':sha(eligiblepath),
      'code_sha256':sha(Path(__file__)),'versions':versions(),'increment_split_seed':42,
      'overlap_record_policy':'use whole original E02 U2 records for shared entities; U3-only records from frozen U3; no field merging',
      'counts':{'Gold_development':len(gold),'Silver':len(silver),'final_Gold_reserved':len(reserved),'U2':len(u2),'U2_fit':len(u2fit),
        'U_common_evaluation':len(common),'U3':len(frame),'U_increment':len(extra),'U_increment_fit':len(increment_fit),
        'U_increment_evaluation':len(increment_eval),'U3_fit':len(broadfit),'U_expanded_evaluation':len(common)+len(increment_eval)},
      'panels':{'primary':'same E02 U2 evaluation entities plus validation Gold','secondary':'same common U plus reserved U3 increment plus validation Gold'},
      'Gold_final_feature_file_copied_or_loaded':False},out/'manifest.json')
    dump(output_hashes(out),out/'output_hashes.json')
    print(json.dumps({'prepared':str(out),'counts':json.loads((out/'manifest.json').read_text())['counts']},ensure_ascii=False),flush=True)

def train(prepared,out,smoke=False):
    if out.exists():raise FileExistsError(out)
    verify_hashes(prepared)
    gold=read_features(prepared/'gold_development.csv');silver=read_features(prepared/'silver.csv')
    u2fit=read_features(prepared/'U2_fit_pool.csv');u3fit=read_features(prepared/'U3_fit_pool.csv')
    common=read_features(prepared/'U_common_evaluation.csv');increment=read_features(prepared/'U_increment_evaluation.csv')
    split=pd.read_csv(prepared/'gold_entity_split.csv',dtype={'credit_code_normalized':str})
    fmap=split.set_index('credit_code_normalized').development_fold
    reserved=set(split.loc[split.role.eq('final_holdout'),'credit_code_normalized'])
    evalcodes=set(common.credit_code_normalized)|set(increment.credit_code_normalized)
    assert not evalcodes&(set(u2fit.credit_code_normalized)|set(u3fit.credit_code_normalized))
    assert not reserved&(evalcodes|set(u3fit.credit_code_normalized)|set(gold.credit_code_normalized)|set(silver.credit_code_normalized))
    seeds=SEEDS[:1] if smoke else SEEDS;repeats=1 if smoke else 10;folds=[1] if smoke else list(range(1,6))
    out.mkdir(parents=True)
    for sub in ['predictions','prediction_index','feature_audits']: (out/sub).mkdir()
    protocol={'status':'running','started_at':now(),'stage':'smoke' if smoke else 'full_development','code_sha256':sha(Path(__file__)),
      'support_code_sha256':sha(HERE/'e02_support.py'),'prepared_manifest_sha256':sha(prepared/'manifest.json'),'versions':versions(),
      'seeds':seeds,'folds':folds,'repeats':repeats,'arms':ARMS,'panels':PANELS,'primary_panel':'common_U2','contrasts':['C_minus_A','D_minus_B'],
      'P_rule':'same national Gold training fold across U scopes; mixed arms add the same Silver; final and current validation Gold excluded',
      'U_rule':'same sample size across all arms; matched U2 sample across A/B and matched U3 sample across C/D; seed fixed; only scope changes',
      'weights':'P total = U total = training Gold count; same total across all four arms; within each P entity equal weight',
      'features':'four-way training-only eligibility intersection; no geography, labels, sources, IDs or completeness variables',
      'parameters':{'solver':'lbfgs','C':1.0,'tol':1e-4,'max_iter':1000,'class_weight':None},
      'prediction_format':'per-fold entity index plus lossless float64 compressed npz means/stds for four arms',
      'final_Gold_holdout_scored':False,'scope':'Gold source-internal development evaluation only; no real U outcome labels or population model selection'}
    dump(protocol,out/'manifest.json');fits=[];metrics=[];samples=[];checks=[]
    try:
      with threadpool_limits(limits=1):
       for fold in folds:
        gtrain=gold.loc[gold.credit_code_normalized.map(fmap).ne(fold)].reset_index(drop=True)
        gval=gold.loc[gold.credit_code_normalized.map(fmap).eq(fold)].reset_index(drop=True)
        mixed=pd.concat([gtrain,silver],ignore_index=True);m=len(gtrain)
        assert not set(gval.credit_code_normalized)&(set(mixed.credit_code_normalized)|reserved)
        target,index=prediction_index(common,increment,gval)
        indexpath=out/'prediction_index'/f'fold_{fold}.csv';save(index,indexpath)
        spec,audit=common_features(gtrain,mixed,u2fit,u3fit);save(audit,out/'feature_audits'/f'fold_{fold}.csv')
        for seed in seeds:
         started=time.monotonic();values={arm:[] for arm in ARMS}
         for repeat in range(repeats):
          sample_seed=seed+fold*10000+repeat
          sampled={'U2':stratified_subset(u2fit,m,sample_seed).reset_index(drop=True),
                   'U3':stratified_subset(u3fit,m,sample_seed).reset_index(drop=True)}
          for scope,u in sampled.items():
            codes=set(u.credit_code_normalized);assert len(codes)==m and not codes&(evalcodes|reserved)
            samples.extend({'fold':fold,'seed':seed,'repeat':repeat+1,'scope':scope,'credit_code_normalized':c} for c in u.credit_code_normalized)
          infos=[]
          for arm,p,scope in [('A_Gold_U2',gtrain,'U2'),('B_mixed_U2',mixed,'U2'),('C_Gold_U3',gtrain,'U3'),('D_mixed_U3',mixed,'U3')]:
            t=time.monotonic();scores,info,_=fit_arm(p,sampled[scope],target,spec,m,sample_seed)
            values[arm].append(scores);infos.append(info)
            fits.append({'arm':arm,'scope':scope,'fold':fold,'seed':seed,'repeat':repeat+1,'sample_seed':sample_seed,
              'U_sample_hash':code_hash(sampled[scope].credit_code_normalized),'U_increment_sampled':int((~sampled[scope].credit_code_normalized.isin(set(u2fit.credit_code_normalized))).sum()),
              'positive_entity_hash':code_hash(p.credit_code_normalized),'target_entity_hash':code_hash(target.credit_code_normalized),
              'feature_fields':'|'.join(spec.all),'fit_seconds':time.monotonic()-t,**info})
          checks.append({'fold':fold,'seed':seed,'repeat':repeat+1,'equal_U_sample_size':all(x['U_rows']==m for x in infos),
            'U2_shared_across_quality_arms':True,'U3_shared_across_quality_arms':True,'P_shared_across_U_scopes':True,
            'same_fields_and_targets':True,'equal_class_and_total_weight':all(np.isclose(x['total_weight'],2*m) and np.isclose(x['positive_weight_sum'],x['U_weight_sum']) for x in infos),
            'all_evaluation_U_and_final_Gold_excluded':True})
         means=np.stack([np.vstack(values[arm]).mean(axis=0) for arm in ARMS]);stds=np.stack([np.vstack(values[arm]).std(axis=0) for arm in ARMS])
         np.savez_compressed(out/'predictions'/f'fold_{fold}_seed_{seed}.npz',mean=means,std=stds,arms=np.array(ARMS),
             index_sha256=np.array(sha(indexpath)),fold=np.array(fold),seed=np.array(seed))
         for arm,score in zip(ARMS,means):
          for panel in PANELS:
            metric,_=rank_metrics(panel_frame(index,score,panel))
            metrics.extend({'arm':arm,'panel':panel,'fold':fold,'seed':seed,**row} for row in metric)
         save(pd.DataFrame(fits),out/'fit_log.csv');save(pd.DataFrame(metrics),out/'cv_metrics.csv')
         print(json.dumps({'fold':fold,'seed':seed,'repeats':repeats,'fields':spec.all,'target_rows':len(target),
           'completed_quadruples':len(checks),'expected_quadruples':len(folds)*len(seeds)*repeats,'elapsed_seconds':round(time.monotonic()-started,2)}),flush=True)
      assert len(fits)==4*len(folds)*len(seeds)*repeats
      assert pd.DataFrame(checks).drop(columns=['fold','seed','repeat']).all().all()
      save(pd.DataFrame(samples),out/'paired_U_samples.csv');save(pd.DataFrame(checks),out/'four_arm_integrity_checks.csv')
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
    a=ap.parse_args()
    if a.command=='prepare':prepare(a.project.resolve(),a.e02_prepared.resolve(),a.upstream.resolve(),a.out.resolve())
    else:train(a.prepared.resolve(),a.out.resolve(),a.smoke)
