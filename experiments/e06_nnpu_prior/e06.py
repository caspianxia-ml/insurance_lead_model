"""E06: F0 linear LR vs logistic-loss nnPU, prior sensitivity on U2/U3."""
from pathlib import Path
import argparse,json,time,warnings
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from legacy.src.preprocessing import FeatureSpec,make_preprocessor
from e02_support import sha,code_hash,dump,save,verify_hashes,output_hashes,versions,now,training_weights,rank_metrics,SEEDS
from nnpu_linear import fit_nnpu

HERE=Path(__file__).resolve().parent
SPEC=FeatureSpec(['company_age','registered_capital_log1p','insured_count_log1p'],['industry_major_category_clean','company_type_clean'])
PRIORS=[.01,.05,.10,.20,.40];METHODS=['LR']+[f'nnPU_pi_{int(p*100):02d}' for p in PRIORS]
ROUTES=['Gold_only','Gold_Silver'];SCOPES=['U2','U3']
META=['credit_code_normalized','company_name','source_record_number','record_id','source','region_group','company_size_clean']
FIELDS=list(dict.fromkeys(META+SPEC.all))

def read_frame(path):
    cols=pd.read_csv(path,nrows=0).columns
    return pd.read_csv(path,dtype={c:str for c in cols if c not in SPEC.numeric},float_precision='round_trip').reset_index(drop=True)

def prepare(project,e05,e04,out):
    if out.exists():raise FileExistsError(out)
    verify_hashes(e05);verify_hashes(e04)
    frozen=json.loads((project/'docs/data_versions/v2_raw_001.json').read_text())
    for item in frozen['files']:assert sha(project/item['registered_path'])==item['sha256']
    frames={name:read_frame(e05/f'{name}.csv').reindex(columns=FIELDS) for name in ['gold_development','silver','U_fit_pool','U_evaluation']}
    frames['U2_fit_pool']=frames.pop('U_fit_pool');frames['U_common_evaluation']=frames.pop('U_evaluation')
    old3=read_frame(e04/'U3_fit_pool.csv').reindex(columns=FIELDS)
    two=set(frames['U2_fit_pool'].credit_code_normalized)
    frames['U3_fit_pool']=pd.concat([frames['U2_fit_pool'],old3.loc[~old3.credit_code_normalized.isin(two)]],ignore_index=True).sort_values('credit_code_normalized').reset_index(drop=True)
    frames['U_increment_evaluation']=read_frame(e04/'U_increment_evaluation.csv').reindex(columns=FIELDS)
    split=pd.read_csv(e05/'gold_entity_split.csv',dtype={'credit_code_normalized':str})
    reserved=set(split.loc[split.role.eq('final_holdout'),'credit_code_normalized'])
    assert len(reserved)==621
    pcodes=set(frames['gold_development'].credit_code_normalized)|set(frames['silver'].credit_code_normalized)
    ucodes=set(frames['U3_fit_pool'].credit_code_normalized)|set(frames['U_common_evaluation'].credit_code_normalized)|set(frames['U_increment_evaluation'].credit_code_normalized)
    assert not pcodes&ucodes and not reserved&(pcodes|ucodes)
    assert two<=set(frames['U3_fit_pool'].credit_code_normalized)
    assert not set(frames['U3_fit_pool'].credit_code_normalized)&(set(frames['U_common_evaluation'].credit_code_normalized)|set(frames['U_increment_evaluation'].credit_code_normalized))
    assert len(frames['gold_development'])==2484 and len(frames['silver'])==30851 and len(frames['U3_fit_pool'])==49956
    out.mkdir(parents=True)
    for name,frame in frames.items():
        assert not frame.credit_code_normalized.duplicated().any();save(frame,out/f'{name}.csv')
    save(split,out/'gold_entity_split.csv')
    dump({'status':'prepared','created_at':now(),'source_version':'v2_raw_001','parent_E05_prepared':str(e05),'parent_E04_prepared':str(e04),
      'parent_E05_hashes_sha256':sha(e05/'output_hashes.json'),'parent_E04_hashes_sha256':sha(e04/'output_hashes.json'),
      'code_sha256':sha(Path(__file__)),'counts':{k:len(v) for k,v in frames.items()},'final_Gold_reserved':621,
      'features':SPEC.all,'priors':PRIORS,'prior_interpretation':'prespecified sensitivity scenarios, not estimates, plausible bounds or inferred P fractions',
      'geography_and_collection_features':False,'holdout_feature_file_used':False},out/'manifest.json')
    dump(output_hashes(out),out/'output_hashes.json');print(json.dumps({'status':'prepared','counts':{k:len(v) for k,v in frames.items()}},ensure_ascii=False),flush=True)

def preprocess_state(pre):
    numeric=pre.named_transformers_['numeric'];categorical=pre.named_transformers_['categorical']
    return {'feature_names':pre.get_feature_names_out().tolist(),
      'numeric_imputer':numeric.named_steps['imputer'].statistics_.tolist(),'numeric_mean':numeric.named_steps['scaler'].mean_.tolist(),
      'numeric_scale':numeric.named_steps['scaler'].scale_.tolist(),'categorical_imputer':categorical.named_steps['imputer'].statistics_.tolist(),
      'categories':[a.tolist() for a in categorical.named_steps['onehot'].categories_]}

def train(prepared,parent_e04,parent_e05,out,smoke=False):
    if out.exists():raise FileExistsError(out)
    for root in [prepared,parent_e04,parent_e05]:verify_hashes(root)
    assert json.loads((prepared/'manifest.json').read_text())['code_sha256']==sha(Path(__file__))
    gold=read_frame(prepared/'gold_development.csv');silver=read_frame(prepared/'silver.csv')
    common=read_frame(prepared/'U_common_evaluation.csv');extra=read_frame(prepared/'U_increment_evaluation.csv')
    pools={s:read_frame(prepared/f'{s}_fit_pool.csv') for s in SCOPES}
    split=pd.read_csv(prepared/'gold_entity_split.csv',dtype={'credit_code_normalized':str});fmap=split.set_index('credit_code_normalized').development_fold
    reserved=set(split.loc[split.role.eq('final_holdout'),'credit_code_normalized'])
    samples=pd.read_csv(parent_e04/'paired_U_samples.csv',dtype={'credit_code_normalized':str})
    folds=[1] if smoke else list(range(1,6));seeds=SEEDS[:1] if smoke else SEEDS;repeats=1 if smoke else 10
    out.mkdir(parents=True)
    for name in ['prediction_index','predictions','models']:(out/name).mkdir()
    protocol={'status':'running','stage':'smoke' if smoke else 'full_development','started_at':now(),'versions':versions(),
      'code_sha256':sha(Path(__file__)),'model_code_sha256':sha(HERE/'nnpu_linear.py'),'support_code_sha256':sha(HERE/'e02_support.py'),
      'prepared_manifest_sha256':sha(prepared/'manifest.json'),'parent_E04_run':str(parent_e04),'parent_E05_run':str(parent_e05),
      'parent_E04_hashes_sha256':sha(parent_e04/'output_hashes.json'),'parent_E05_hashes_sha256':sha(parent_e05/'output_hashes.json'),
      'features':SPEC.all,'priors':PRIORS,'methods':METHODS,'routes':ROUTES,'scopes':SCOPES,'folds':folds,'seeds':seeds,'repeats':repeats,
      'model':'linear logistic-loss nnPU estimator, convex full-batch optimization; not neural networks or original stochastic Algorithm 1',
      'loss':'pi*mean_P softplus(-z) + max(0, mean_U softplus(z) - pi*mean_P softplus(z))',
      'regularization':'lambda=1/(2*n_training_Gold), lambda/2*||w||², intercept unpenalized; matches normalized LR C=1 penalty',
      'optimizer':'L-BFGS-B for smooth convex mixture; if correction binds, brentq mixing dual; analytic gradients and numerical stationarity/duality checks',
      'sampling':'exact E04 U2/U3 draw replay; same draw across prior/method/positive routes; no added U budget',
      'U2_LR':'reuse E05 F0 exact ensemble mean/std; never refit. Smoke uses old 10-repeat reference, not an effect estimate.',
      'evaluation':'U2 primary on fixed common U2+Gold; U3 secondary on fixed expanded U+Gold and common subset. Compare methods within scope/panel only.',
      'prior_policy':'1%,5%,10%,20%,40% separate stress scenarios per scope, not prevalence estimates or confidence bounds; never select true prior by Gold capture',
      'limitations':'P source representativeness unproven; Silver may be noisy; both violate ideal nnPU assumptions; scores not probabilities',
      'final_Gold_scored':False,'expected_new_fits':len(folds)*len(seeds)*repeats*22}
    dump(protocol,out/'manifest.json');logs=[];metrics=[];reuses=[];completed=0
    try:
      with threadpool_limits(limits=1):
       for fold in folds:
        gt=gold.loc[gold.credit_code_normalized.map(fmap).ne(fold)].reset_index(drop=True);gv=gold.loc[gold.credit_code_normalized.map(fmap).eq(fold)].reset_index(drop=True);m=len(gt)
        positives={'Gold_only':gt,'Gold_Silver':pd.concat([gt,silver],ignore_index=True)}
        for scope in SCOPES:
         ueval=common if scope=='U2' else pd.concat([common,extra],ignore_index=True)
         target=pd.concat([ueval,gv],ignore_index=True)
         index=target[['credit_code_normalized','company_name','source','region_group','company_size_clean','industry_major_category_clean']].copy()
         index['kind']=np.r_[np.repeat('U_common',len(common)),np.repeat('U_increment',len(ueval)-len(common)),np.repeat('Gold_validation',len(gv))]
         index.insert(0,'prediction_row',np.arange(len(index)));idxpath=out/'prediction_index'/f'fold_{fold}_{scope}.csv';save(index,idxpath)
         assert not set(target.credit_code_normalized)&(set(pools[scope].credit_code_normalized)|set(positives['Gold_Silver'].credit_code_normalized)|reserved)
         for seed in seeds:
          start=time.monotonic();values={(r,method):[] for r in ROUTES for method in METHODS if method!='LR' or scope=='U3'}
          for repeat in range(1,repeats+1):
           group=samples.loc[samples.fold.eq(fold)&samples.seed.eq(seed)&samples['repeat'].eq(repeat)&samples.scope.eq(scope)]
           codes=group.credit_code_normalized.tolist();u=pools[scope].set_index('credit_code_normalized').loc[codes].reset_index();assert len(u)==m
           for route,p in positives.items():
            began=time.monotonic();pre=make_preprocessor(SPEC);pre.fit(pd.concat([p,u],ignore_index=True)[SPEC.all])
            xp=csr_matrix(pre.transform(p[SPEC.all]),dtype=np.float64);xu=csr_matrix(pre.transform(u[SPEC.all]),dtype=np.float64);xt=csr_matrix(pre.transform(target[SPEC.all]),dtype=np.float64)
            stem=f'fold_{fold}_{scope}_seed_{seed}_repeat_{repeat}_{route}';state=preprocess_state(pre);dump(state,out/'models'/f'{stem}.json')
            commonlog={'scope':scope,'route':route,'fold':fold,'seed':seed,'repeat':repeat,'n_training_Gold':m,'positive_rows':len(p),'U_rows':len(u),
              'positive_entity_hash':code_hash(p.credit_code_normalized),'U_sample_hash':code_hash(codes),'target_entity_hash':code_hash(target.credit_code_normalized),
              'feature_fields':'|'.join(SPEC.all),'encoded_columns':xp.shape[1],'preprocess_sha256':sha(out/'models'/f'{stem}.json'),'model_stem':stem}
            coefficients=[]
            for prior,method in zip(PRIORS,METHODS[1:]):
                t=time.monotonic();theta,info=fit_nnpu(xp,xu,prior,1/(2*m));score=expit(np.asarray(xt@theta[:-1]).ravel()+theta[-1]);assert np.isfinite(score).all()
                coefficients.append(theta);values[route,method].append(score)
                logs.append({**commonlog,'method':method,'prior':prior,'fit_seconds':time.monotonic()-t,**info});completed+=1
            lrcoef=np.empty((0,),dtype=float)
            if scope=='U3':
                model=LogisticRegression(C=1.,solver='lbfgs',tol=1e-4,max_iter=1000,class_weight=None,random_state=seed+fold*10000+repeat-1)
                from scipy.sparse import vstack
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always',ConvergenceWarning);model.fit(vstack([xp,xu]),np.r_[np.ones(len(p)),np.zeros(len(u))],sample_weight=training_weights(m,len(p),len(u)))
                if any(issubclass(w.category,ConvergenceWarning) for w in caught):raise RuntimeError('U3 LR failed to converge')
                score=model.predict_proba(xt)[:,1];values[route,'LR'].append(score);lrcoef=np.r_[model.coef_[0],model.intercept_[0]]
                logs.append({**commonlog,'method':'LR','prior':np.nan,'converged':True,'iterations':int(model.n_iter_.max()),'fit_seconds':time.monotonic()-began,'penalty_lambda':1/(2*m)});completed+=1
            np.savez_compressed(out/'models'/f'{stem}.npz',nnpu_coefficients=np.stack(coefficients),priors=np.array(PRIORS),LR_coefficients=lrcoef)
          keys=[(r,method) for r in ROUTES for method in METHODS];means=[];stds=[]
          for route,method in keys:
            if scope=='U2' and method=='LR':
                path=parent_e05/'predictions'/f'fold_{fold}_seed_{seed}.npz'
                oldidx=pd.read_csv(parent_e05/'prediction_index'/f'fold_{fold}.csv',dtype={'credit_code_normalized':str})
                assert oldidx.credit_code_normalized.tolist()==index.credit_code_normalized.tolist()
                with np.load(path,allow_pickle=False) as old:
                    key=np.flatnonzero((old['variants']=='F0')&(old['routes']==route));assert len(key)==1
                    means.append(old['mean'][key[0]]);stds.append(old['std'][key[0]])
                reuses.append({'scope':scope,'route':route,'fold':fold,'seed':seed,'source_file':str(path),'source_sha256':sha(path),'variant':'F0','reference_only':smoke})
            else:
                a=np.vstack(values[route,method]);means.append(a.mean(axis=0));stds.append(a.std(axis=0))
          means=np.stack(means);stds=np.stack(stds)
          np.savez_compressed(out/'predictions'/f'fold_{fold}_{scope}_seed_{seed}.npz',mean=means,std=stds,routes=np.array([r for r,method in keys]),methods=np.array([method for r,method in keys]),
             index_sha256=np.array(sha(idxpath)),fold=np.array(fold),seed=np.array(seed),scope=np.array(scope))
          for (route,method),score in zip(keys,means):
            for panel in (['common_U2'] if scope=='U2' else ['common_U2','expanded_U3']):
                mask=index.kind.ne('U_increment') if panel=='common_U2' else pd.Series(True,index=index.index)
                pred=index.loc[mask].copy();pred['score']=score[mask.to_numpy()]
                for row in rank_metrics(pred)[0]:metrics.append({'scope':scope,'route':route,'method':method,'panel':panel,'fold':fold,'seed':seed,**row})
          print(json.dumps({'fold':fold,'scope':scope,'seed':seed,'new_fits_completed':completed,'expected':protocol['expected_new_fits'],'seconds':round(time.monotonic()-start,2)}),flush=True)
      save(pd.DataFrame(logs),out/'fit_log.csv');save(pd.DataFrame(metrics),out/'cv_metrics.csv');save(pd.DataFrame(reuses),out/'U2_LR_reuse_log.csv');save(samples,out/'replayed_U_samples.csv')
      protocol.update(status='complete',completed_at=now(),new_fits=len(logs),nnpu_fits=int(sum(r['method']!='LR' for r in logs)),new_LR_fits=int(sum(r['method']=='LR' for r in logs)))
      dump(protocol,out/'manifest.json');dump(output_hashes(out),out/'output_hashes.json');print(json.dumps({'status':'complete','new_fits':len(logs),'final_Gold_scored':False}),flush=True)
    except Exception as e:
      protocol.update(status='failed',error=str(e));dump(protocol,out/'manifest.json');raise

if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='mode',required=True)
    s=sub.add_parser('prepare');s.add_argument('--project',type=Path,required=True);s.add_argument('--e05-prepared',type=Path,required=True);s.add_argument('--e04-prepared',type=Path,required=True);s.add_argument('--out',type=Path,required=True)
    s=sub.add_parser('train');s.add_argument('--prepared',type=Path,required=True);s.add_argument('--parent-e04',type=Path,required=True);s.add_argument('--parent-e05',type=Path,required=True);s.add_argument('--out',type=Path,required=True);s.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    if a.mode=='prepare':prepare(a.project.resolve(),a.e05_prepared.resolve(),a.e04_prepared.resolve(),a.out.resolve())
    else:train(a.prepared.resolve(),a.parent_e04.resolve(),a.parent_e05.resolve(),a.out.resolve(),a.smoke)
