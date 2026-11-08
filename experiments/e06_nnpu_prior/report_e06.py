"""Independent coefficient replay, risk checks, and within-panel E06 reporting."""
from pathlib import Path
from itertools import combinations
import argparse,json,math,os,tempfile
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.special import expit
from threadpoolctl import threadpool_limits
from legacy.src.preprocessing import make_preprocessor
from e06 import SPEC,PRIORS,METHODS,ROUTES,SCOPES,read_frame,preprocess_state
from e02_support import sha,code_hash,save,dump,verify_hashes,output_hashes

def report(prepared,run,parent04,parent05,out):
    if out.exists():raise FileExistsError(out)
    for root in [prepared,run,parent04,parent05]:verify_hashes(root)
    protocol=json.loads((run/'manifest.json').read_text());pm=json.loads((prepared/'manifest.json').read_text())
    assert protocol['status']=='complete' and protocol['stage']=='full_development' and not protocol['final_Gold_scored']
    assert protocol['code_sha256']==sha(Path(__file__).with_name('e06.py'))
    assert protocol['model_code_sha256']==sha(Path(__file__).with_name('nnpu_linear.py'))
    assert protocol['support_code_sha256']==sha(Path(__file__).with_name('e02_support.py'))
    assert protocol['prepared_manifest_sha256']==sha(prepared/'manifest.json')
    assert protocol['parent_E04_hashes_sha256']==sha(parent04/'output_hashes.json')
    assert protocol['parent_E05_hashes_sha256']==sha(parent05/'output_hashes.json')
    gold=read_frame(prepared/'gold_development.csv');silver=read_frame(prepared/'silver.csv')
    common=read_frame(prepared/'U_common_evaluation.csv');extra=read_frame(prepared/'U_increment_evaluation.csv')
    pools={scope:read_frame(prepared/f'{scope}_fit_pool.csv') for scope in SCOPES}
    split=pd.read_csv(prepared/'gold_entity_split.csv',dtype={'credit_code_normalized':str});fmap=split.set_index('credit_code_normalized').development_fold
    reserved=set(split.loc[split.role.eq('final_holdout'),'credit_code_normalized'])
    logs=pd.read_csv(run/'fit_log.csv');metrics=pd.read_csv(run/'cv_metrics.csv');samples=pd.read_csv(run/'replayed_U_samples.csv',dtype={'credit_code_normalized':str});reuse=pd.read_csv(run/'U2_LR_reuse_log.csv')
    checks=[]
    def check(name,passed,detail=''):
        checks.append({'check':name,'passed':bool(passed),'detail':str(detail)})
        if not passed:raise AssertionError(name+': '+str(detail))
    check('all_input_and_code_hashes',True)
    check('3300_completed_fits',len(logs)==3300 and logs.method.ne('LR').sum()==3000 and logs.method.eq('LR').sum()==300 and logs.converged.all())
    check('five_fixed_scenarios',sorted(logs.loc[logs.method.ne('LR'),'prior'].unique())==PRIORS and protocol['priors']==PRIORS)
    check('621_final_Gold_excluded_from_preparation',len(reserved)==621 and not reserved&(set(gold.credit_code_normalized)|set(silver.credit_code_normalized)|set(pools['U3'].credit_code_normalized)|set(common.credit_code_normalized)|set(extra.credit_code_normalized)))
    check('no_final_Gold_feature_file',not any('holdout' in p.name.lower() for p in prepared.glob('*.csv')))
    # Source roles and unchanged F0 values; locate preparation next to supplied archives.
    parentprep05=parent05.parent.parent/'prepared_001';parentprep04=parent04.parent.parent/'prepared_001'
    for parentprep,label in [(parentprep05,'E05'),(parentprep04,'E04')]:
        verify_hashes(parentprep);check(label+'_prepared_lineage',sha(parentprep/'output_hashes.json')==pm[f'parent_{label}_hashes_sha256'])
    for name,frame,oldname in [('gold_development',gold,'gold_development'),('silver',silver,'silver'),('U2_fit_pool',pools['U2'],'U_fit_pool'),('U_common_evaluation',common,'U_evaluation')]:
        old=read_frame(parentprep05/f'{oldname}.csv')
        check(name+'_parent_entities',frame.credit_code_normalized.tolist()==old.credit_code_normalized.tolist())
        for field in SPEC.all:
            equal=np.array_equal(frame[field].to_numpy(float),old[field].to_numpy(float),equal_nan=True) if field in SPEC.numeric else frame[field].fillna('<NA>').tolist()==old[field].fillna('<NA>').tolist()
            check(name+'_'+field+'_unchanged',equal)
    oldsplit=pd.read_csv(parentprep05/'gold_entity_split.csv',dtype={'credit_code_normalized':str})
    check('unchanged_Gold_roles_and_folds',split.equals(oldsplit))
    old3=read_frame(parentprep04/'U3_fit_pool.csv');two=set(pools['U2'].credit_code_normalized)
    expected3=pd.concat([pools['U2'],old3.loc[~old3.credit_code_normalized.isin(two)]],ignore_index=True).sort_values('credit_code_normalized').reset_index(drop=True)
    oldextra=read_frame(parentprep04/'U_increment_evaluation.csv')
    for name,frame,old in [('U3_fit_pool',pools['U3'],expected3),('U_increment_evaluation',extra,oldextra)]:
        check(name+'_unchanged_role_and_order',frame.credit_code_normalized.tolist()==old.credit_code_normalized.tolist())
        for field in SPEC.all:
            equal=np.array_equal(frame[field].to_numpy(float),old[field].to_numpy(float),equal_nan=True) if field in SPEC.numeric else frame[field].fillna('<NA>').tolist()==old[field].fillna('<NA>').tolist()
            check(name+'_'+field+'_unchanged',equal)
    oldsample=pd.read_csv(parent04/'paired_U_samples.csv',dtype={'credit_code_normalized':str})
    check('exact_parent_U_draw_log_replayed',samples.equals(oldsample))
    logindex=logs.set_index(['scope','route','fold','seed','repeat','method'],verify_integrity=True)
    keys=[(route,method) for route in ROUTES for method in METHODS]
    gp=[];usums={'U2':np.zeros((12,len(common))),'U3':np.zeros((12,len(common)+len(extra)))};nmodels={s:0 for s in SCOPES}
    with threadpool_limits(limits=1):
     for fold in range(1,6):
      gt=gold.loc[gold.credit_code_normalized.map(fmap).ne(fold)].reset_index(drop=True);gv=gold.loc[gold.credit_code_normalized.map(fmap).eq(fold)].reset_index(drop=True);m=len(gt)
      positives={'Gold_only':gt,'Gold_Silver':pd.concat([gt,silver],ignore_index=True)}
      for scope in SCOPES:
       ueval=common if scope=='U2' else pd.concat([common,extra],ignore_index=True)
       target=pd.concat([ueval,gv],ignore_index=True);idxpath=run/'prediction_index'/f'fold_{fold}_{scope}.csv'
       index=pd.read_csv(idxpath,dtype=str,keep_default_na=False);gm=index.kind.eq('Gold_validation').to_numpy()
       check(f'{fold}_{scope}_fixed_index',index.credit_code_normalized.tolist()==target.credit_code_normalized.tolist() and not index.credit_code_normalized.duplicated().any())
       check(f'{fold}_{scope}_Gold_and_U_kinds',int(gm.sum())==len(gv) and index.loc[~gm,'credit_code_normalized'].tolist()==ueval.credit_code_normalized.tolist() and index.prediction_row.astype(int).tolist()==list(range(len(index))))
       for seed in [42,123,2026]:
        path=run/'predictions'/f'fold_{fold}_{scope}_seed_{seed}.npz'
        with np.load(path,allow_pickle=False) as data:
            mean=data['mean'];std=data['std']
            check(path.stem+'_index_hash',str(data['index_sha256'].item())==sha(idxpath))
            check(path.stem+'_key_and_format',list(zip(data['routes'],data['methods']))==keys and mean.dtype==std.dtype==np.float64 and mean.shape==std.shape==(12,len(index)))
            check(path.stem+'_finite_scores',np.isfinite(mean).all() and np.isfinite(std).all() and (std>=0).all())
        replay={(r,method):[] for r in ROUTES for method in METHODS if scope=='U3' or method!='LR'}
        for repeat in range(1,11):
          group=samples.loc[samples.fold.eq(fold)&samples.seed.eq(seed)&samples['repeat'].eq(repeat)&samples.scope.eq(scope)]
          codes=group.credit_code_normalized.tolist();u=pools[scope].set_index('credit_code_normalized').loc[codes].reset_index()
          check(f'{fold}_{scope}_{seed}_{repeat}_U_isolation',len(codes)==len(set(codes))==m and not set(codes)&(set(target.credit_code_normalized)|reserved))
          for route,p in positives.items():
            stem=f'fold_{fold}_{scope}_seed_{seed}_repeat_{repeat}_{route}'
            pre=make_preprocessor(SPEC);pre.fit(pd.concat([p,u],ignore_index=True)[SPEC.all])
            state=json.loads((run/'models'/f'{stem}.json').read_text())
            check(stem+'_train_only_preprocess',state==preprocess_state(pre))
            xp=csr_matrix(pre.transform(p[SPEC.all]),dtype=np.float64);xu=csr_matrix(pre.transform(u[SPEC.all]),dtype=np.float64);xt=csr_matrix(pre.transform(target[SPEC.all]),dtype=np.float64)
            with np.load(run/'models'/f'{stem}.npz',allow_pickle=False) as data:
                coefficients=data['nnpu_coefficients'];lrcoef=data['LR_coefficients']
                check(stem+'_coefficient_format',coefficients.shape==(5,xp.shape[1]+1) and np.array_equal(data['priors'],PRIORS) and np.isfinite(coefficients).all())
            zp=xp@coefficients[:,:-1].T+coefficients[:,-1];zu=xu@coefficients[:,:-1].T+coefficients[:,-1]
            target_scores=expit(xt@coefficients[:,:-1].T+coefficients[:,-1])
            for j,(prior,method) in enumerate(zip(PRIORS,METHODS[1:])):
                row=logindex.loc[(scope,route,fold,seed,repeat,method)]
                positive=prior*np.logaddexp(0,-zp[:,j]).mean();negative=np.logaddexp(0,zu[:,j]).mean()-prior*np.logaddexp(0,zp[:,j]).mean()
                alpha=float(row.dual_alpha);penalty=1/(2*m);risk=positive+max(0,negative);reg=.5*penalty*np.dot(coefficients[j,:-1],coefficients[j,:-1]);gap=max(0,negative)-alpha*negative
                cp=prior*((1-alpha)*(expit(zp[:,j])-1)-alpha)/len(p);cu=alpha*expit(zu[:,j])/len(u)
                gradient=np.r_[np.asarray(xp.T@cp+xu.T@cu).ravel()+penalty*coefficients[j,:-1],cp.sum()+cu.sum()]
                check(stem+'_'+method+'_risk_and_optimality',np.allclose([positive,negative,risk,risk+reg,gap],[row.positive_risk,row.uncorrected_negative_risk,row.nnpu_risk,row.objective,row.primal_dual_gap],rtol=1e-7,atol=1e-9)
                  and 0<alpha<=1 and np.max(np.abs(gradient))<=2e-5 and -1e-10<=gap<=5e-6 and np.isclose(row.penalty_lambda,penalty,rtol=1e-12,atol=1e-15) and bool(row.correction_binding)==(alpha<1)
                  and row.max_inner_gradient_inf<=2e-5 and 0<=row.roundoff_stationary_exits<=row.inner_solves)
                check(stem+'_'+method+'_roles_and_features',row.positive_entity_hash==code_hash(p.credit_code_normalized) and row.U_sample_hash==code_hash(codes) and row.target_entity_hash==code_hash(target.credit_code_normalized)
                  and row.feature_fields=='|'.join(SPEC.all) and row.preprocess_sha256==sha(run/'models'/f'{stem}.json') and row.positive_rows==len(p) and row.U_rows==m and row.n_training_Gold==m and row.prior==prior)
                replay[route,method].append(np.asarray(target_scores[:,j]).ravel())
            if scope=='U3':
                row=logindex.loc[(scope,route,fold,seed,repeat,'LR')]
                check(stem+'_LR_coefficient_and_roles',lrcoef.shape==(xp.shape[1]+1,) and np.isfinite(lrcoef).all() and row.positive_entity_hash==code_hash(p.credit_code_normalized) and row.U_sample_hash==code_hash(codes)
                  and row.target_entity_hash==code_hash(target.credit_code_normalized) and row.feature_fields=='|'.join(SPEC.all) and row.iterations<1000 and np.isclose(row.penalty_lambda,1/(2*m),rtol=1e-12,atol=1e-15))
                replay[route,'LR'].append(expit(np.asarray(xt@lrcoef[:-1]).ravel()+lrcoef[-1]))
            else:check(stem+'_no_redundant_LR_fit',lrcoef.size==0)
        for key,(route,method) in enumerate(keys):
            if scope=='U2' and method=='LR':
                oldpath=parent05/'predictions'/f'fold_{fold}_seed_{seed}.npz'
                with np.load(oldpath,allow_pickle=False) as old:
                    oldkey=np.flatnonzero((old['variants']=='F0')&(old['routes']==route));assert len(oldkey)==1
                    check(path.stem+'_'+route+'_LR_exact_reuse',np.array_equal(mean[key],old['mean'][oldkey[0]]) and np.array_equal(std[key],old['std'][oldkey[0]]))
                rec=reuse.loc[reuse.fold.eq(fold)&reuse.seed.eq(seed)&reuse.route.eq(route)]
                check(path.stem+'_'+route+'_LR_reuse_source',len(rec)==1 and not rec.reference_only.any() and rec.source_sha256.iloc[0]==sha(oldpath))
            else:
                values=np.vstack(replay[route,method])
                check(path.stem+'_'+route+'_'+method+'_independent_prediction_replay',np.allclose(values.mean(axis=0),mean[key],rtol=1e-10,atol=1e-12) and np.allclose(values.std(axis=0),std[key],rtol=1e-9,atol=1e-12))
            for panel in (['common_U2'] if scope=='U2' else ['common_U2','expanded_U3']):
                mask=index.kind.ne('U_increment').to_numpy() if panel=='common_U2' else np.ones(len(index),dtype=bool)
                ids=np.flatnonzero(mask);order=np.lexsort((index.loc[mask,'credit_code_normalized'].to_numpy(dtype=str),-mean[key,mask]));ranked=ids[order]
                ranks=np.zeros(len(index),dtype=int);ranks[ranked]=np.arange(1,len(ranked)+1)
                rows=metrics.loc[metrics.scope.eq(scope)&metrics.route.eq(route)&metrics.method.eq(method)&metrics.panel.eq(panel)&metrics.fold.eq(fold)&metrics.seed.eq(seed)]
                check(path.stem+'_'+route+'_'+method+'_'+panel+'_metric_count',len(rows)==4)
                for row in rows.to_dict('records'):
                    k=math.ceil(len(ids)*row['top_fraction']);hits=int(gm[ranked[:k]].sum())
                    check(path.stem+'_'+route+'_'+method+'_'+panel+'_'+str(row['top_fraction']),k==row['top_k'] and hits==row['captured_Gold'] and len(ids)==row['evaluation_rows'] and int(gm.sum())==row['validation_Gold'] and np.isclose(hits/gm.sum(),row['capture_rate']))
                frame=index.loc[gm,['credit_code_normalized','company_name','source','region_group','company_size_clean','industry_major_category_clean']].copy()
                frame['source_group']=np.where(frame.source.str.contains('量化Q3榜单',regex=False),'quantitative_list','historical_Gold')
                frame['scope']=scope;frame['route']=route;frame['method']=method;frame['panel']=panel;frame['fold']=fold;frame['seed']=seed;frame['score']=mean[key,gm];frame['rank']=ranks[gm];frame['evaluation_rows']=len(ids);gp.append(frame)
        usums[scope]+=mean[:,:len(ueval)];nmodels[scope]+=1
        print(json.dumps({'verified_fold':fold,'scope':scope,'seed':seed,'checks_so_far':len(checks)}),flush=True)
    goldpred=pd.concat(gp,ignore_index=True);coverage=goldpred.groupby(['scope','route','method','panel','seed']).credit_code_normalized.agg(['count','nunique'])
    check('all_Gold_OOF_coverage',len(coverage)==108 and coverage['count'].eq(2484).all() and coverage['nunique'].eq(2484).all())
    check('2160_capture_metrics',len(metrics)==2160)
    check('600_model_snapshots',len(list((run/'models').glob('*.npz')))==600 and len(list((run/'models').glob('*.json')))==600)
    out.mkdir(parents=True);save(pd.DataFrame(checks),out/'independent_verification.csv');save(goldpred,out/'Gold_development_OOF_predictions.csv')
    summarize(prepared,run,out,metrics,logs,goldpred,common,extra,{s:usums[s]/nmodels[s] for s in SCOPES},len(checks))

def summarize(prepared,run,out,metrics,logs,gp,common,extra,umean,nchecks):
    group=['scope','route','method','panel','seed','top_fraction']
    byseed=metrics.groupby(group,as_index=False).agg(captured_Gold=('captured_Gold','sum'),Gold_n=('validation_Gold','sum'))
    byseed['capture_rate']=byseed.captured_Gold/byseed.Gold_n;save(byseed,out/'capture_by_seed.csv')
    summary=byseed.groupby(['scope','route','method','panel','top_fraction'],as_index=False).agg(mean_capture=('capture_rate','mean'),min_capture=('capture_rate','min'),max_capture=('capture_rate','max'));save(summary,out/'capture_summary.csv')
    paired=[];pooled=[]
    for method,prior in zip(METHODS[1:],PRIORS):
        a=metrics[metrics.method.eq(method)].merge(metrics[metrics.method.eq('LR')],on=['scope','route','panel','fold','seed','top_fraction'],suffixes=('_nnPU','_LR'),validate='one_to_one');a['prior']=prior;a['method']=method;a['difference_pp']=100*(a.capture_rate_nnPU-a.capture_rate_LR);paired.append(a)
        b=byseed[byseed.method.eq(method)].merge(byseed[byseed.method.eq('LR')],on=['scope','route','panel','seed','top_fraction'],suffixes=('_nnPU','_LR'),validate='one_to_one');b['prior']=prior;b['method']=method;b['difference_pp']=100*(b.capture_rate_nnPU-b.capture_rate_LR);pooled.append(b)
    paired=pd.concat(paired,ignore_index=True);pooled=pd.concat(pooled,ignore_index=True)
    save(paired,out/'paired_nnPU_minus_LR.csv');save(pooled,out/'pooled_nnPU_minus_LR_by_seed.csv')
    effects=pooled.groupby(['scope','route','method','prior','panel','top_fraction'],as_index=False).agg(mean_difference_pp=('difference_pp','mean'),min_seed_difference_pp=('difference_pp','min'),max_seed_difference_pp=('difference_pp','max'));save(effects,out/'nnPU_minus_LR_summary.csv')
    directions=paired.groupby(['scope','route','method','prior','panel','top_fraction'],as_index=False).agg(improved=('difference_pp',lambda x:int((x>0).sum())),ties=('difference_pp',lambda x:int((x==0).sum())),comparisons=('difference_pp','size'));save(directions,out/'paired_direction_counts.csv')
    envelope=summary[summary.method.ne('LR')].groupby(['scope','route','panel','top_fraction'],as_index=False).agg(min_scenario_capture=('mean_capture','min'),max_scenario_capture=('mean_capture','max'))
    envelope['scenario_spread_pp']=100*(envelope.max_scenario_capture-envelope.min_scenario_capture)
    envelope=envelope.merge(summary.loc[summary.method.eq('LR'),['scope','route','panel','top_fraction','mean_capture']].rename(columns={'mean_capture':'LR_capture'}),on=['scope','route','panel','top_fraction'],validate='one_to_one');save(envelope,out/'prior_sensitivity_envelope_NOT_confidence_intervals.csv')
    nn=logs[logs.method.ne('LR')].copy()
    diagnostics=nn.groupby(['scope','route','prior'],as_index=False).agg(fits=('method','size'),correction_binding=('correction_binding','sum'),roundoff_stationary_exits=('roundoff_stationary_exits','sum'),
      max_gradient_inf=('gradient_inf','max'),max_inner_gradient_inf=('max_inner_gradient_inf','max'),max_primal_dual_residual=('primal_dual_gap','max'),min_dual_alpha=('dual_alpha','min'),max_dual_alpha=('dual_alpha','max'),median_inner_solves=('inner_solves','median'))
    diagnostics['correction_binding']=diagnostics.correction_binding.astype(int);diagnostics['roundoff_stationary_exits']=diagnostics.roundoff_stationary_exits.astype(int)
    save(diagnostics,out/'optimization_and_correction_diagnostics.csv')
    subgroup=[]
    for field in ['region_group','source_group','company_size_clean','industry_major_category_clean']:
        for fraction in [.1,.2]:
            frame=gp.copy();frame['group']=frame[field].replace('','<missing>');frame['captured']=frame['rank'].le(np.ceil(frame.evaluation_rows*fraction))
            rows=frame.groupby(['scope','route','method','panel','seed','group'],as_index=False).agg(Gold_n=('captured','size'),captured_Gold=('captured','sum'))
            rows['group_field']=field;rows['top_fraction']=fraction;rows['capture_rate']=rows.captured_Gold/rows.Gold_n;subgroup.append(rows)
    sub=pd.concat(subgroup,ignore_index=True);save(sub,out/'subgroup_capture_by_seed.csv')
    sub=sub.groupby(['scope','route','method','panel','group_field','group','top_fraction'],as_index=False).agg(Gold_n=('Gold_n','first'),mean_capture=('capture_rate','mean'));save(sub,out/'subgroup_capture_summary.csv')
    differences=[]
    for method,prior in zip(METHODS[1:],PRIORS):
        d=sub[sub.method.eq(method)].merge(sub[sub.method.eq('LR')],on=['scope','route','panel','group_field','group','top_fraction'],suffixes=('_nnPU','_LR'),validate='one_to_one')
        d['prior']=prior;d['method']=method;d['difference_pp']=100*(d.mean_capture_nnPU-d.mean_capture_LR);differences.append(d)
    save(pd.concat(differences,ignore_index=True),out/'subgroup_nnPU_minus_LR.csv')
    overlaps=[];correlations=[];topdetails=[];keymap={(r,method):j for j,(r,method) in enumerate((r,method) for r in ROUTES for method in METHODS)}
    for scope in SCOPES:
        ueval=common if scope=='U2' else pd.concat([common,extra],ignore_index=True)
        for panel in (['common_U2'] if scope=='U2' else ['common_U2','expanded_U3']):
            size=len(common) if panel=='common_U2' else len(ueval)
            for route in ROUTES:
                wide=ueval.iloc[:size][['credit_code_normalized','company_name','region_group']].copy();ranks={}
                for method in METHODS:
                    values=umean[scope][keymap[route,method],:size];order=np.lexsort((wide.credit_code_normalized.to_numpy(dtype=str),-values));rank=np.empty(size,dtype=int);rank[order]=np.arange(1,size+1)
                    wide[method+'_score']=values;wide[method+'_rank']=rank;ranks[method]=rank
                save(wide,out/f'{scope}_{panel}_{route}_U_development_scores_and_ranks.csv')
                comparisons=[(method,'LR','nnPU_vs_LR') for method in METHODS[1:]]
                comparisons += [(high,low,'prior_40_vs_01' if low==METHODS[1] and high==METHODS[-1] else 'prior_scenarios') for low,high in combinations(METHODS[1:],2)]
                for added,base,kind in comparisons:
                    correlations.append({'scope':scope,'panel':panel,'route':route,'added_method':added,'base_method':base,'comparison':kind,'entities':size,
                      'Spearman_rank_correlation':float(pd.Series(ranks[added]).corr(pd.Series(ranks[base]),method='pearson'))})
                    for k in [100,500,1000]:
                        a=ranks[added]<=k;b=ranks[base]<=k
                        overlaps.append({'scope':scope,'panel':panel,'route':route,'added_method':added,'base_method':base,'comparison':kind,'top_k':k,'intersection':int((a&b).sum()),'retained_fraction':float((a&b).sum()/k)})
                        if k==100:
                            d=wide.loc[a^b,['credit_code_normalized','company_name',added+'_rank',base+'_rank']].copy().rename(columns={added+'_rank':'added_rank',base+'_rank':'base_rank'})
                            d['scope']=scope;d['panel']=panel;d['route']=route;d['added_method']=added;d['base_method']=base;d['comparison']=kind;d['direction']=np.where(a[a^b],'added_only','base_only');topdetails.append(d)
    overlapframe=pd.DataFrame(overlaps);correlationframe=pd.DataFrame(correlations)
    save(overlapframe,out/'TopK_nnPU_and_prior_overlap.csv');save(correlationframe,out/'nnPU_and_prior_rank_correlations.csv');save(pd.concat(topdetails,ignore_index=True),out/'Top100_difference_details.csv')
    sensitivity=overlapframe[overlapframe.comparison.ne('nnPU_vs_LR')&overlapframe.top_k.eq(100)].groupby(['scope','panel','route'],as_index=False).agg(min_Top100_intersection=('intersection','min'),max_Top100_intersection=('intersection','max'),prior_pairs=('intersection','size'))
    corrsummary=correlationframe[correlationframe.comparison.ne('nnPU_vs_LR')].groupby(['scope','panel','route'],as_index=False).agg(min_Spearman=('Spearman_rank_correlation','min'),max_Spearman=('Spearman_rank_correlation','max'))
    save(sensitivity.merge(corrsummary,on=['scope','panel','route'],validate='one_to_one'),out/'all_prior_pair_ranking_sensitivity.csv')
    draw(out,summary)
    narrative(prepared,run,out,summary,effects,envelope,diagnostics,nchecks)

def draw(out,summary):
    os.environ.setdefault('MPLCONFIGDIR',tempfile.mkdtemp(prefix='e06-mpl-'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for i,(scope,panel,label) in enumerate([('U2','common_U2','U2 primary'),('U3','expanded_U3','U3 expanded secondary')]):
        for j,(route,title) in enumerate([('Gold_only','Gold only'),('Gold_Silver','Gold + Silver')]):
            ax=axes[i,j]
            for fraction,color in [(.1,'#3477ad'),(.2,'#d28a32')]:
                frame=summary[summary.scope.eq(scope)&summary.panel.eq(panel)&summary.route.eq(route)&summary.top_fraction.eq(fraction)].set_index('method')
                ax.plot(np.arange(5),[100*float(frame.loc[method,'mean_capture']) for method in METHODS[1:]],marker='o',color=color,label=f'nnPU Top {fraction:.0%}')
                ax.axhline(100*float(frame.loc['LR','mean_capture']),ls='--',color=color,alpha=.7,label=f'LR Top {fraction:.0%}')
            ax.set_title(label+' / '+title);ax.set_xticks(range(5),['1%','5%','10%','20%','40%']);ax.set_xlabel('Assumed prior (scenario, not estimated)');ax.set_ylabel('Development Gold capture (%)');ax.legend(frameon=False,fontsize=7);ax.spines[['top','right']].set_visible(False)
    fig.savefig(out/'E06_prior_sensitivity.png',dpi=180);plt.close(fig)

def narrative(prepared,run,out,summary,effects,envelope,diagnostics,nchecks):
    def val(scope,panel,route,method,fraction):return float(summary.loc[summary.scope.eq(scope)&summary.panel.eq(panel)&summary.route.eq(route)&summary.method.eq(method)&summary.top_fraction.eq(fraction),'mean_capture'].iloc[0])
    lines=['# E06：线性 nnPU / LR 与类别先验敏感性','',
      '开发主对照完成。3,000 次线性 nnPU＋300 次 U3 LR 新拟合完成数值最优性检查；U2 的 30 份 LR 集成评分及抽样标准差精确复用 E05 F0。621 家最终留出 Gold 未训练或评分。结果只反映现有 Gold 来源内找回能力，不是 U 企业真实福利发现率。','',
      '## 比较口径','',
      '预先固定 F0 五类画像：年龄、注册资本、参保人数、行业大类、企业类型。两方法均为线性模型，共用每次训练部分拟合的预处理；不加入行业门类/中类、地区、简介覆盖、年份、ID、来源或标签特征。',
      '开发 Gold 2,484、Silver 30,851，五折、三个既定种子、十次 U 抽样重复沿用旧划分。每次 U 约 1,987/1,988 家，完全重放 E04 清单。两条正例路线分别保留，混合路线仍存在 Silver 标签噪声与 Gold 权重份额较低的问题。',
      'U2 拟合候选 7,516、共同独立评价 11,274；U3 拟合候选 49,956、扩展独立评价 74,936。只在固定总体/面板内比较方法，不跨面板直接比较原始捕获率。','',
      '## 先验是什么、这次没有估计什么','',
      'π 是对应 U 中真实正例的比例。1%、5%、10%、20%、40% 是训练前固定的压力测试情景，不是已证实的比例、合理边界或置信区间。未使用观察 P/(P+U) 估计 π，也不凭开发 Gold 找回率最高选择真实 π。U2/U3 分别解释，同一网格不意味着两总体先验相同。',
      'nnPU 的理论风险解释要求 P 代表目标正类、U 代表目标混合分布，并使用适当先验。本项目 Gold 来源选择未解决，Silver 可能有误标，因此不能因采用 PU 方法就认为偏差已消除；分数也不解释为福利概率。','',
      '## 固定 U2 主比较','',
      '| 方法/先验情景 | Gold-only 前10% | Gold-only 前20% | 混合 前10% | 混合 前20% |','|---|---:|---:|---:|---:|']
    for method in METHODS:
        name='LR（复用 E05）' if method=='LR' else f'nnPU，π={PRIORS[METHODS.index(method)-1]:.0%}'
        lines.append('| '+name+' | '+' | '.join(f'{val("U2","common_U2",route,method,fraction):.1%}' for route in ROUTES for fraction in [.1,.2])+' |')
    lines+=['','## 固定扩展 U3 补充比较','','| 方法/先验情景 | Gold-only 前10% | Gold-only 前20% | 混合 前10% | 混合 前20% |','|---|---:|---:|---:|---:|']
    for method in METHODS:
        name='LR（同 F0 新建）' if method=='LR' else f'nnPU，π={PRIORS[METHODS.index(method)-1]:.0%}'
        lines.append('| '+name+' | '+' | '.join(f'{val("U3","expanded_U3",route,method,fraction):.1%}' for route in ROUTES for fraction in [.1,.2])+' |')
    lines+=['','U3 在共同 U2 子面板中的结果也保存在 capture_summary.csv；该表用于补充诊断，同样不能把原始捕获率与扩展面板直接横比。','',
      '## 情景包络与解释','',
      '以下是五个情景的最小/最大开发均值，不是统计置信区间，也不表示上下界覆盖真实 π。','',
      '| 总体/面板 | 路线 | 前10% nnPU 范围 | 前10% LR | 前20% nnPU 范围 | 前20% LR |','|---|---|---:|---:|---:|---:|']
    for scope,panel in [('U2','common_U2'),('U3','expanded_U3')]:
        for route in ROUTES:
            cells=[]
            for f in [.1,.2]:
                e=envelope.loc[envelope.scope.eq(scope)&envelope.panel.eq(panel)&envelope.route.eq(route)&envelope.top_fraction.eq(f)].iloc[0]
                cells.extend([f'{e.min_scenario_capture:.1%}–{e.max_scenario_capture:.1%}',f'{e.LR_capture:.1%}'])
            lines.append('| '+scope+' | '+route+' | '+' | '.join(cells)+' |')
    lines+=['','每个 π 相对同面板 LR 的差值、三种子范围和 15 个配对折/种子方向见 nnPU_minus_LR_summary.csv、paired_direction_counts.csv。十五次比较相关，多先验开发探索未构成最终验证。TopK_nnPU_and_prior_overlap.csv 另记录全部五个先验的十组两两名单变化，1%/40% 端点另标记；整体排名相似不保证头部名单相同。all_prior_pair_ranking_sensitivity.csv 汇总最小/最大头部重合与排名相关。','',
      '## 数值方法与核验','',
      '本轮优化 logistic-loss 非负经验风险：π×P正损失＋max(0，U负损失−π×P负损失)，再加 L2 惩罚。λ=1/(2×训练 Gold 数)，截距不惩罚，与归一化旧 LR C=1 的惩罚系数一致。',
      '比较的是完整 PU 风险方案与 LR：先验权重、P负损失抵消和非负修正一起改变；没有独立的 uPU 评分对照，不能把差值单独归因于 max 修正。',
      '利用线性 logistic 风险的凸结构求全批量数值解；这不是原随机 Algorithm 1，也不是神经网络实验。梯度有限差分、独立 SLSQP 上图问题对照、可复现性及先验测试四项通过。',
      f'独立核验 {nchecks:,} 项全部通过：重建每次训练预处理、从全部 600 份模型系数快照重新评分，核验 3,000 次风险/驻点/数值原始对偶残差及实体隔离，独立重算 2,160 条捕获指标，并精确核对 U2 LR 均值和标准差。',
      '首次 main_001 在极小梯度处因 L-BFGS-B 状态2终止，失败记录和旧代码保留。正式 main_002 只对状态2且内部梯度≤1e−7的情况以严格驻点标准接纳；最终梯度≤2e−5、数值原始对偶残差≤5e−6的门槛不变。未因验证结果变更情景或样本。',
      f'正式运行共记录 {int(diagnostics.roundoff_stationary_exits.sum()):,} 次内部浮点精度驻点退出；它们仍经过独立数值检查，不称为全部求解器正常退出。修正约束起作用的拟合共 {int(diagnostics.correction_binding.sum()):,}/3,000 次。各先验/总体/路线详见 optimization_and_correction_diagnostics.csv。','',
      '每次预处理状态、模型系数、无损预测、索引和输入/源码哈希均保留；源/地区/规模分组见 subgroup_nnPU_minus_LR.csv，少量来源样本不能代替充分的独立来源评价。','',
      '## 当前结论与下一步','',
      '本轮回答的是固定 F0 的线性 nnPU 在各先验情景下如何改变来源内排序。即使某个情景提高，也不能证明该 π 真实或整体方法在业务总体上胜出；即使没有提高，也不能否定未检验的非线性 nnPU。两条正例路线继续保留，最终 Gold 留出保持封存。',
      '下一项 E07 需要独立业务证据：预先固定分层抽样框、随机对照和证据标准，记录确认有/确认无/无法判断。当前没有自动外呼、发消息或捏造人工标签，也没有输出生产部署或真实概率校准结论。','',
      '方法来源：[Kiryo 等 NeurIPS 2017 原论文](https://proceedings.neurips.cc/paper/2017/file/7cce53cf90577442771720a370c3c723-Paper.pdf)；[作者原实现](https://github.com/kiryor/nnPUlearning/blob/master/pu_loss.py)。线性凸求解与本轮情景设计为本地实验选择，不声称复现原文神经网络结果。','']
    (out/'E06结果报告.md').write_text('\n'.join(lines),encoding='utf-8')
    dump({'status':'complete','source_run':str(run),'source_run_manifest_sha256':sha(run/'manifest.json'),'report_code_sha256':sha(Path(__file__)),
      'independent_checks':nchecks,'all_passed':True,'new_fits':3300,'nnpu_fits':3000,'final_Gold_scored':False,'true_prior_estimated':False,
      'model_scope':'linear logistic-loss nnPU only; source-internal development conditional on prior scenarios'},out/'manifest.json')
    dump(output_hashes(out),out/'output_hashes.json');print(json.dumps({'status':'complete','checks':nchecks,'report':str(out),'scenario_envelope':envelope.loc[envelope.top_fraction.isin([.1,.2])].to_dict('records')},ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepared',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--parent-e04',type=Path,required=True);p.add_argument('--parent-e05',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();report(a.prepared.resolve(),a.run.resolve(),a.parent_e04.resolve(),a.parent_e05.resolve(),a.out.resolve())
