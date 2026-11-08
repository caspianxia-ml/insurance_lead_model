"""E05 paired feature ablations on the unchanged E02 national P/U2 roles."""
from pathlib import Path
import argparse,json,re,time,unicodedata
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from e02_support import (sha,code_hash,dump,save,verify_hashes,output_hashes,versions,now,
    read_features,fit_arm,rank_metrics,SEEDS,FEATURES,REFERENCE_DATE,common_features)
from legacy.src.preprocessing import FeatureSpec
from legacy.src.data_cleaning import _normalise_text,_normalise_credit_code,_normalise_industry

HERE=Path(__file__).resolve().parent
BASE_N=['company_age','registered_capital_log1p','insured_count_log1p']
BASE_C=['industry_major_category_clean','company_type_clean']
SIZE=['company_size_clean'];IND=['industry_category_clean','industry_medium_category_clean']
PAID=['paid_capital_log1p'];YEAR=['insured_report_year_numeric'];GEO=['address_province','address_city'];INTRO=['introduction_flag']
SPECS={
 'F0':FeatureSpec(BASE_N,BASE_C),
 'F1_size':FeatureSpec(BASE_N,BASE_C+SIZE),
 'F1_industry':FeatureSpec(BASE_N,BASE_C+IND),
 'F1':FeatureSpec(BASE_N,BASE_C+SIZE+IND),
 'F2_safe':FeatureSpec(BASE_N+PAID,BASE_C+SIZE),
 'F2_year':FeatureSpec(BASE_N+PAID+YEAR,BASE_C+SIZE),
 'F2':FeatureSpec(BASE_N+PAID+YEAR,BASE_C+SIZE+IND),
 'F3':FeatureSpec(BASE_N+PAID,BASE_C+SIZE+GEO),
 'F4':FeatureSpec(BASE_N+PAID+INTRO,BASE_C+SIZE),
}
VARIANTS=list(SPECS);ROUTES=['Gold_only','Gold_Silver'];ALL_VARIANTS=['B0_reused']+VARIANTS
CONTRASTS=[('F1_size','F0','size'),('F1_industry','F0','industry_without_size'),
 ('F1','F1_size','industry_with_size'),('F1','F1_industry','size_with_industry'),
 ('F2_safe','F1_size','paid_capital'),('F2_year','F2_safe','insured_year'),
 ('F2','F1','paid_and_year_in_full_chain'),('F3','F2_safe','address_geography'),
 ('F4','F2_safe','introduction_coverage'),('F2_safe','F0','safe_profile_package'),
 ('B0_reused','F2_safe','legacy_status_and_field_order')]
NUMERIC=set(BASE_N+PAID+YEAR+INTRO)
MISSING={'','null','none','nan','nat','未知','-','--'}
MUNICIPAL=['上海','北京','天津','重庆']
PROVINCES=['河北','山西','辽宁','吉林','黑龙江','江苏','浙江','安徽','福建','江西','山东','河南','湖北','湖南','广东','海南','四川','贵州','云南','陕西','甘肃','青海','台湾']
SPECIAL={'内蒙古自治区':'内蒙古','广西壮族自治区':'广西','西藏自治区':'西藏','宁夏回族自治区':'宁夏','新疆维吾尔自治区':'新疆','香港特别行政区':'香港','澳门特别行政区':'澳门'}

def address_region(value):
    """Same conservative address-only parser for P/U; no tags or company names."""
    if pd.isna(value):return np.nan,np.nan
    text=unicodedata.normalize('NFKC',str(value));text=re.sub(r'\s+','',text)
    if text.lower() in MISSING or '<' in text:return np.nan,np.nan
    text=re.sub(r'^(?:中华人民共和国|中国)\(?','',text).lstrip('(')
    for city in MUNICIPAL:
        if text.startswith(city):return city,city
    province=np.nan;rest=text
    for full,short in SPECIAL.items():
        if text.startswith(full):province=short;rest=text[len(full):];break
    if pd.isna(province):
        for short in sorted(PROVINCES,key=len,reverse=True):
            if text.startswith(short+'省'):province=short;rest=text[len(short)+1:];break
    match=re.match(r'^([\u4e00-\u9fff]{2,12}?)(市|自治州|地区|盟)',rest)
    city=(match.group(1)+(match.group(2) if match.group(2)!='市' else '')) if match else np.nan
    return province,city

def parse_year(series):
    text=_normalise_text(series).astype('string')
    valid=text.str.fullmatch(r'(?:19|20)\d{2}(?:\.0)?',na=False)
    number=pd.to_numeric(text.where(valid),errors='coerce').astype(float)
    return number.where(number.between(1900,int(REFERENCE_DATE[:4])))

def parse_intro(series):
    text=series.astype('string').str.strip()
    return pd.to_numeric(text.where(text.isin(['0','1'])),errors='coerce').astype(float)

def read_frame(path):
    cols=pd.read_csv(path,nrows=0).columns
    return pd.read_csv(path,dtype={c:str for c in cols if c not in NUMERIC},float_precision='round_trip').reset_index(drop=True)

def enrich(base,raw):
    ids=base.source_record_number.astype(int).to_numpy()-1
    selected=raw.iloc[ids].reset_index(drop=True)
    assert _normalise_credit_code(selected.credit_code).tolist()==base.credit_code_normalized.tolist()
    assert selected.company_name.tolist()==base.company_name.tolist()
    result=base.copy()
    regions=selected.registered_address.map(address_region)
    result['address_province']=regions.map(lambda x:x[0]);result['address_city']=regions.map(lambda x:x[1])
    result['insured_report_year_numeric']=parse_year(selected.insured_count_report_year)
    result['introduction_flag']=parse_intro(selected.has_company_introduction)
    # Keep inherited legacy industry values byte-for-byte equivalent after CSV roundtrip.
    assert result.industry_category_clean.fillna('<NA>').tolist()==selected.industry_category.map(_normalise_industry).fillna('<NA>').tolist()
    return result

def field_audit(gold,mixed,ufit,fold):
    rows=[];allowed=list(dict.fromkeys(x for spec in SPECS.values() for x in spec.all))
    for route,p in [('Gold_only',gold),('Gold_Silver',mixed)]:
        for name in allowed:
            a=float(p[name].notna().mean());b=float(ufit[name].notna().mean())
            if min(a,b)<=0:raise ValueError(f'All missing feature in training role: {fold}, {route}, {name}')
            risks=[]
            if abs(a-b)>.2:risks.append('P_U_nonmissing_gap_over_20pp')
            if name=='industry_category_clean':risks.append('P_U_industry_level_semantics_differ')
            if name in INTRO:risks.append('collection_coverage_not_business_truth')
            if name in GEO:risks.append('address_only_partial_coverage_and_source_geography')
            rows.append({'fold':fold,'route':route,'feature':name,'P_nonmissing':a,'U_fit_nonmissing':b,
              'absolute_gap':abs(a-b),'legacy_gap_guard_pass':abs(a-b)<=.2,'risk':'|'.join(risks),
              'policy':'explicit_diagnostic_exposure' if risks else 'controlled_profile_feature'})
    return pd.DataFrame(rows)

def prepare(project,parent,out):
    if out.exists():raise FileExistsError(out)
    verify_hashes(parent)
    frozen=json.loads((project/'docs/data_versions/v2_raw_001.json').read_text());raws={};inputs=[]
    for dataset in ['P_National','U2']:
        entry=next(e for e in frozen['files'] if e['dataset']==dataset);path=project/entry['registered_path']
        assert sha(path)==entry['sha256']
        raws[dataset]=pd.read_csv(path,dtype=str,keep_default_na=False)
        inputs.append({'dataset':dataset,'path':str(path),'sha256':entry['sha256']})
    frames={}
    for name,dataset in [('gold_development','P_National'),('silver','P_National'),('U_fit_pool','U2'),('U_evaluation','U2')]:
        base=read_features(parent/f'{name}.csv');frames[name]=enrich(base,raws[dataset])
    split=pd.read_csv(parent/'gold_entity_split.csv',dtype={'credit_code_normalized':str})
    reserved=set(split.loc[split.role.eq('final_holdout'),'credit_code_normalized']);assert len(reserved)==621
    allcodes=set()
    for frame in frames.values():
        codes=set(frame.credit_code_normalized);assert not codes&(allcodes|reserved) and len(codes)==len(frame);allcodes|=codes
    gold,silver,ufit=frames['gold_development'],frames['silver'],frames['U_fit_pool']
    fmap=split.set_index('credit_code_normalized').development_fold
    audits=[]
    for fold in range(1,6):
        g=gold.loc[gold.credit_code_normalized.map(fmap).ne(fold)]
        audits.append(field_audit(g,pd.concat([g,silver]),ufit,fold))
    audit=pd.concat(audits,ignore_index=True)
    out.mkdir(parents=True)
    for name,frame in frames.items():save(frame,out/f'{name}.csv')
    save(split,out/'gold_entity_split.csv');save(audit,out/'training_only_feature_audit.csv')
    coverage=[]
    for name,frame in frames.items():
        for field in ['address_province','address_city','industry_category_clean','industry_medium_category_clean','insured_report_year_numeric','introduction_flag']:
            coverage.append({'role':name,'field':field,'entities':len(frame),'nonmissing':int(frame[field].notna().sum()),'nonmissing_ratio':float(frame[field].notna().mean())})
    save(pd.DataFrame(coverage),out/'field_coverage.csv')
    dump({'status':'prepared','created_at':now(),'source_version':'v2_raw_001','reference_date':REFERENCE_DATE,'inputs':inputs,
      'parent_prepared':str(parent),'parent_manifest_sha256':sha(parent/'manifest.json'),'parent_hash_manifest_sha256':sha(parent/'output_hashes.json'),
      'code_sha256':sha(Path(__file__)),'versions':versions(),'counts':{name:len(f) for name,f in frames.items()},'final_Gold_reserved':621,
      'variants':{name:{'numeric':s.numeric,'categorical':s.categorical} for name,s in SPECS.items()},'contrasts':CONTRASTS,
      'field_policy':'Fixed explicit allowlist; legacy missingness guard audited. Medium industry/year high gaps and category-level semantic mismatch are diagnostic exposures, not approved deployment features.',
      'geography':'same address-only deterministic parser for both P and U; no Leads tags, existing region fields or company-name fallback',
      'holdout_features_built_or_scored':False},out/'manifest.json')
    dump(output_hashes(out),out/'output_hashes.json')
    print(json.dumps({'status':'prepared','counts':{n:len(f) for n,f in frames.items()},'risk_fields':sorted(audit.loc[audit.risk.ne(''),'feature'].unique())},ensure_ascii=False),flush=True)

def train(prepared,parentrun,out,smoke=False):
    if out.exists():raise FileExistsError(out)
    verify_hashes(prepared);verify_hashes(parentrun)
    assert json.loads((prepared/'manifest.json').read_text())['code_sha256']==sha(Path(__file__))
    gold=read_frame(prepared/'gold_development.csv');silver=read_frame(prepared/'silver.csv')
    ufit=read_frame(prepared/'U_fit_pool.csv');ueval=read_frame(prepared/'U_evaluation.csv')
    split=pd.read_csv(prepared/'gold_entity_split.csv',dtype={'credit_code_normalized':str});fmap=split.set_index('credit_code_normalized').development_fold
    reserved=set(split.loc[split.role.eq('final_holdout'),'credit_code_normalized'])
    parentfits=pd.read_csv(parentrun/'fit_log.csv');parent_samples=pd.read_csv(parentrun/'paired_U_samples.csv',dtype={'credit_code_normalized':str})
    out.mkdir(parents=True)
    for name in ['prediction_index','predictions','feature_audits']:(out/name).mkdir()
    folds=[1] if smoke else list(range(1,6));seeds=SEEDS[:1] if smoke else SEEDS;repeats=1 if smoke else 10
    protocol={'status':'running','started_at':now(),'stage':'smoke' if smoke else 'full_development','code_sha256':sha(Path(__file__)),
      'support_code_sha256':sha(HERE/'e02_support.py'),'prepared_manifest_sha256':sha(prepared/'manifest.json'),
      'parent_E02_run':str(parentrun),'parent_E02_hashes_sha256':sha(parentrun/'output_hashes.json'),'versions':versions(),
      'variants':['B0_reused']+VARIANTS,'routes':ROUTES,'folds':folds,'seeds':seeds,'repeats':repeats,'expected_new_fits':len(folds)*len(seeds)*repeats*len(VARIANTS)*2,
      'U_scope':'Fixed E02 U2 fit pool and evaluation; same exact parent U draw replayed for every variant/route',
      'weights':'P total = U total = training Gold count; inherited LR C=1 lbfgs tol=1e-4 max_iter=1000',
      'B0_policy':'Reuse existing E02 fold/seed ensemble scores; no redundant refit. Full-run comparison only; smoke B0 marked reference_only.',
      'holdout_scored':False,'scope':'Gold source-internal development; no real U outcome labels, business accuracy or deployment selection'}
    dump(protocol,out/'manifest.json');fits=[];metrics=[];reuse=[];completed=0
    try:
      with threadpool_limits(limits=1):
       for fold in folds:
        gt=gold.loc[gold.credit_code_normalized.map(fmap).ne(fold)].reset_index(drop=True);gv=gold.loc[gold.credit_code_normalized.map(fmap).eq(fold)].reset_index(drop=True)
        mixed=pd.concat([gt,silver],ignore_index=True);target=pd.concat([ueval,gv],ignore_index=True);m=len(gt)
        assert not set(target.credit_code_normalized)&(set(mixed.credit_code_normalized)|set(ufit.credit_code_normalized)|reserved)
        index=target[['credit_code_normalized','company_name','region_group','source','company_size_clean','industry_major_category_clean']].copy()
        index['kind']=np.r_[np.repeat('U_evaluation',len(ueval)),np.repeat('Gold_validation',len(gv))];index.insert(0,'prediction_row',range(len(index)))
        idxpath=out/'prediction_index'/f'fold_{fold}.csv';save(index,idxpath)
        audit=field_audit(gt,mixed,ufit,fold);save(audit,out/'feature_audits'/f'fold_{fold}.csv')
        oldspec,_=common_features(gt,mixed,ufit)
        for seed in seeds:
         start=time.monotonic();values={(variant,route):[] for variant in VARIANTS for route in ROUTES}
         for repeat in range(1,repeats+1):
          ps=parent_samples.loc[parent_samples.fold.eq(fold)&parent_samples.seed.eq(seed)&parent_samples['repeat'].eq(repeat)]
          codes=ps.credit_code_normalized.tolist();sample=ufit.set_index('credit_code_normalized').loc[codes].reset_index()
          sample_seed=seed+fold*10000+repeat-1
          assert len(sample)==m and not set(codes)&set(target.credit_code_normalized)
          oldlogs=parentfits.loc[parentfits.fold.eq(fold)&parentfits.seed.eq(seed)&parentfits['repeat'].eq(repeat)]
          assert len(oldlogs)==2 and oldlogs.U_sample_hash.eq(code_hash(codes)).all()
          for variant,spec in SPECS.items():
           for route,p in [('Gold_only',gt),('Gold_Silver',mixed)]:
            t=time.monotonic();score,info,_=fit_arm(p,sample,target,spec,m,sample_seed)
            values[variant,route].append(score)
            fits.append({'variant':variant,'route':route,'fold':fold,'seed':seed,'repeat':repeat,'sample_seed':sample_seed,
              'feature_fields':'|'.join(spec.all),'U_sample_hash':code_hash(codes),'positive_entity_hash':code_hash(p.credit_code_normalized),
              'target_entity_hash':code_hash(target.credit_code_normalized),'fit_seconds':time.monotonic()-t,**info})
         keys=[(v,r) for v in ALL_VARIANTS for r in ROUTES];means=[];stds=[]
         for variant,route in keys:
          if variant=='B0_reused':
            oldarm='A_Gold_only' if route=='Gold_only' else 'B_Gold_Silver'
            oldpath=parentrun/'predictions'/f'fold_{fold}_seed_{seed}_{oldarm}.csv'
            old=pd.read_csv(oldpath,dtype={'credit_code_normalized':str},float_precision='round_trip').set_index('credit_code_normalized').loc[target.credit_code_normalized]
            means.append(old.score.to_numpy());stds.append(old.repeat_score_std.to_numpy())
            fields=parentfits.loc[parentfits.fold.eq(fold)&parentfits.seed.eq(seed)&parentfits.arm.eq(oldarm),'feature_fields']
            assert fields.eq('|'.join(oldspec.all)).all()
            reuse.append({'variant':variant,'route':route,'fold':fold,'seed':seed,'parent_file':str(oldpath),'sha256':sha(oldpath),'feature_fields':'|'.join(oldspec.all),'reference_only':smoke})
          else:
            a=np.vstack(values[variant,route]);means.append(a.mean(axis=0));stds.append(a.std(axis=0))
         means=np.stack(means);stds=np.stack(stds)
         np.savez_compressed(out/'predictions'/f'fold_{fold}_seed_{seed}.npz',mean=means,std=stds,
            variants=np.array([v for v,r in keys]),routes=np.array([r for v,r in keys]),index_sha256=np.array(sha(idxpath)),fold=np.array(fold),seed=np.array(seed))
         for (variant,route),score in zip(keys,means):
            pred=index.copy();pred['score']=score
            for row in rank_metrics(pred)[0]:metrics.append({'variant':variant,'route':route,'fold':fold,'seed':seed,**row})
         completed+=repeats*len(VARIANTS)*2
         print(json.dumps({'fold':fold,'seed':seed,'new_fits_completed':completed,'expected':protocol['expected_new_fits'],'seconds':round(time.monotonic()-start,2)}),flush=True)
      save(pd.DataFrame(fits),out/'fit_log.csv');save(pd.DataFrame(metrics),out/'cv_metrics.csv');save(pd.DataFrame(reuse),out/'B0_reuse_log.csv')
      protocol.update(status='complete',completed_at=now(),new_fits=len(fits),reused_ensemble_predictions=len(reuse));dump(protocol,out/'manifest.json')
      dump(output_hashes(out),out/'output_hashes.json')
      print(json.dumps({'status':'complete','new_fits':len(fits),'B0_reused':len(reuse),'holdout_scored':False}),flush=True)
    except Exception as e:
      protocol.update(status='failed',error=str(e));dump(protocol,out/'manifest.json');raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='mode',required=True)
    p=sub.add_parser('prepare');p.add_argument('--project',type=Path,required=True);p.add_argument('--parent',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p=sub.add_parser('train');p.add_argument('--prepared',type=Path,required=True);p.add_argument('--parent-run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--smoke',action='store_true')
    a=parser.parse_args()
    if a.mode=='prepare':prepare(a.project.resolve(),a.parent.resolve(),a.out.resolve())
    else:train(a.prepared.resolve(),a.parent_run.resolve(),a.out.resolve(),a.smoke)
