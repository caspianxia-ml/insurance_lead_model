"""Audit existing development inputs and scores. No fitting or holdout scoring."""
import argparse, csv, hashlib, itertools, json, re, unicodedata
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

SEEDS=[42,123,2026]
KS=[100,500,1000]
FIELDS=['industry_category_clean','industry_major_category_clean','industry_medium_category_clean']
EXPECTED=dict(zip(FIELDS,['section','division','group']))
LEVELS=['section','division','group','class']

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def save(p,rows):pd.DataFrame(rows).to_csv(p,index=False,encoding='utf-8-sig')
def read(p):return pd.read_csv(p,dtype={'credit_code_normalized':str},keep_default_na=False)
def norm(x):return re.sub(r'\s+','',unicodedata.normalize('NFKC',str(x)))
def order(score,codes):return np.lexsort((np.asarray(codes,str),-np.asarray(score,float)))

class Taxonomy:
    def __init__(self,reference):
        self.nodes={n['code']:n for n in reference['nodes']};self.names=defaultdict(list)
        for n in reference['nodes']:self.names[n['name']].append(n['code'])
    def match(self,value):
        value=norm(value)
        if not value:return {'status':'missing','codes':[],'levels':[]}
        codes=self.names.get(value,[])
        return dict(status='unmatched' if not codes else 'exact_unique' if len(codes)==1 else 'exact_ambiguous',codes=codes,levels=sorted({self.nodes[c]['level'] for c in codes}))
    def ancestor(self,code,level):
        while code is not None:
            node=self.nodes[code]
            if node['level']==level:return code
            code=node['parent']
        return None
    def canonical(self,values,level):
        """All matched supplied fields must admit the same ancestor. No fuzzy fill."""
        constraints=[]
        for value in values:
            m=self.match(value)
            if m['status'] in ['missing','unmatched']:continue
            aset={a for c in m['codes'] if (a:=self.ancestor(c,level)) is not None}
            # A coarse section also constrains a proposed division; never ignore
            # a contradictory section merely because it has no division ancestor.
            for c in m['codes']:
                if self.ancestor(c,level) is None:
                    aset.update(nc for nc,n in self.nodes.items() if n['level']==level and self.ancestor(nc,self.nodes[c]['level'])==c)
            if aset:constraints.append(aset)
        if not constraints:return '', 'no_reference_evidence'
        common=set.intersection(*constraints)
        if len(common)==1:return next(iter(common)),'reference_consistent'
        return '', 'conflict' if not common else 'ambiguous'

def verify_hashes(root,inputs,checks):
    p=root/'output_hashes.json';obj=json.loads(p.read_text())
    # Historical archives use a flat filename->hash dictionary.
    if 'files' in obj and isinstance(obj['files'],dict):obj=obj['files']
    for rel,expected in obj.items():
        if not isinstance(expected,str) or not re.fullmatch('[0-9a-f]{64}',expected):raise ValueError(f'Unexpected hash format {p}: {rel}')
        target=root/rel;assert target.is_file() and sha(target)==expected,(target,'hash mismatch')
    inputs[str(p)]=sha(p);checks.append(dict(check=f'hashes:{root.name}:{root.parent.name}',passed=True,count=len(obj)))

def industry_audit(project,reference,frames,out):
    taxonomy=Taxonomy(reference);counts=[];profiles=[];details=[]
    for role,df in frames.items():
        for field in FIELDS:
            for value,n in df[field].value_counts(dropna=False).items():
                m=taxonomy.match(value);expected=EXPECTED[field]
                counts.append(dict(role=role,field=field,value=value,n=int(n),status=m['status'],reference_codes='|'.join(m['codes']),reference_levels='|'.join(m['levels']),expected_level=expected,expected_level_possible=expected in m['levels'],definitely_wrong_level=bool(m['codes']) and expected not in m['levels']))
        rows=[]
        for row in df.itertuples(index=False):
            d=row._asdict();values=[d[f] for f in FIELDS]
            section,ss=taxonomy.canonical(values,'section');division,ds=taxonomy.canonical(values,'division')
            rr=dict(role=role,credit_code_normalized=d['credit_code_normalized'],company_name=d['company_name'],source_record_number=d['source_record_number'],**{f:d[f] for f in FIELDS},audit_section_code=section,audit_division_code=division,section_status=ss,division_status=ds)
            rows.append(rr)
        save(out/f'industry_entities_{role}.csv',rows);details+=rows
        profiles.append(dict(role=role,n=len(df),section_status=dict(Counter(x['section_status'] for x in rows)),division_status=dict(Counter(x['division_status'] for x in rows))))
    save(out/'industry_value_audit.csv',counts);dump(out/'industry_role_profiles.json',profiles)
    vc=pd.DataFrame(counts)
    summary=[]
    for (role,field),g in vc.groupby(['role','field'],sort=False):
        total=int(g.n.sum());missing=int(g.loc[g.status.eq('missing'),'n'].sum());wrong=int(g.loc[g.definitely_wrong_level,'n'].sum())
        summary.append(dict(role=role,field=field,n=total,missing=missing,unmatched=int(g.loc[g.status.eq('unmatched'),'n'].sum()),exact_ambiguous=int(g.loc[g.status.eq('exact_ambiguous'),'n'].sum()),definitely_wrong_level=wrong,wrong_fraction_of_nonmissing=wrong/(total-missing) if total>missing else None))
    save(out/'industry_field_summary.csv',summary)
    # Three raw fields by source: this resolves whether level mismatch is introduced locally.
    rawfindings=[]
    wanted={}
    for role,df in frames.items():
        source='P_National' if role in ['Gold_development','Silver'] else 'U2' if role.startswith('U2') else 'U3'
        wanted.setdefault(source,{})
        for r in df.itertuples(index=False):
            wanted[source][int(r.source_record_number)]=tuple(getattr(r,f) for f in FIELDS)
    frozen=json.loads((project/'docs/data_versions/v2_raw_001.json').read_text())
    for item in frozen['files']:
        source=item['dataset']
        if source not in wanted:continue
        p=project/item['registered_path'];assert sha(p)==item['sha256']
        errors=0;seen=0;differences=[]
        with p.open(encoding='utf-8-sig',newline='') as f:
            reader=csv.reader(f,strict=source!='U3');header=next(reader);cols=[header.index(x.replace('_clean','')) for x in FIELDS]
            for num,row in enumerate(reader,1):
                assert len(row)==len(header),(source,num,len(row))
                if num not in wanted[source]:continue
                seen+=1;raw=[row[c] for c in cols]
                # Match legacy _normalise_industry exactly for the comparison.
                def legacy(x):
                    x=str(x).strip()
                    if x.lower() in ['', 'nan','none','null','nat','未知','--','-']:return ''
                    x=re.sub(r'\s+','',x)
                    return unicodedata.normalize('NFKC',x).replace('｜','丨').replace('|','丨')
                expected=tuple(legacy(x) for x in raw)
                if expected!=wanted[source][num]:
                    errors+=1
                    if len(differences)<5:differences.append(dict(record=num,raw=raw,prepared=wanted[source][num]))
        assert seen==len(wanted[source])
        rawfindings.append(dict(source=source,raw_sha256=sha(p),selected_records=seen,raw_to_prepared_differences=errors,examples=differences))
    dump(out/'industry_raw_lineage.json',rawfindings)
    return summary,profiles

def heads(root,prepared,descriptors,out,inputs,checks):
    raw=[];aggregates={};methodkeys=None;headfiles=[];concentration=[];seedoverlap=[];routeoverlap=[];ties=[]
    reserved=set(read(prepared/'gold_entity_split.csv').query("role == 'final_holdout'").credit_code_normalized)
    goldset=set(read(prepared/'gold_development.csv').credit_code_normalized)
    for scope in ['U2','U3']:
        sums=None;codes=None;allgold=[]
        for fold in range(1,6):
            ip=root/'prediction_index'/f'fold_{fold}_{scope}.csv';idx=read(ip);inputs[str(ip)]=sha(ip)
            g=idx.kind.eq('Gold_validation').to_numpy();u=~g
            assert not set(idx.credit_code_normalized)&reserved
            assert set(idx.loc[g,'credit_code_normalized'])<=goldset
            assert set(idx.loc[u,'kind'])==({'U_common'} if scope=='U2' else {'U_common','U_increment'})
            allgold+=idx.loc[g,'credit_code_normalized'].tolist()
            if codes is None:codes=idx.loc[u,'credit_code_normalized'].to_numpy(str);sums=np.zeros((3,12,len(codes)))
            assert codes.tolist()==idx.loc[u,'credit_code_normalized'].tolist()
            for si,seed in enumerate(SEEDS):
                p=root/'predictions'/f'fold_{fold}_{scope}_seed_{seed}.npz';inputs[str(p)]=sha(p)
                with np.load(p,allow_pickle=False) as data:
                    score=data['mean'];keys=list(zip(data['routes'].tolist(),data['methods'].tolist()))
                    assert data['index_sha256'].item()==sha(ip)
                    assert score.shape==(12,len(idx)) and np.isfinite(score).all()
                    if methodkeys is None:methodkeys=keys
                    assert methodkeys==keys
                    sums[si]+=score[:,u]/5
                    for mi,(route,method) in enumerate(keys):
                        ranks=order(score[mi],idx.credit_code_normalized)
                        for k in KS:
                            hits=int(g[ranks[:k]].sum());assert hits<=min(k,int(g.sum()))
                            raw.append(dict(scope=scope,fold=fold,seed=seed,route=route,method=method,top_k=k,gold_hits=hits,gold_total=int(g.sum()),panel_total=len(idx)))
        assert len(allgold)==len(set(allgold))==2484 and set(allgold)==goldset
        checks.append(dict(check=f'{scope}:all_2484_Gold_once_per_seed_no_holdout',passed=True))
        aggregates[scope]=(codes,sums)
        # Existing development U only. All five models per seed excluded U evaluation from fitting.
        desc=descriptors[scope].set_index('credit_code_normalized',verify_integrity=True).loc[codes].reset_index()
        assert not set(codes)&(reserved|goldset)
        assert len(desc)==(11274 if scope=='U2' else 74936)
        for mi,(route,method) in enumerate(methodkeys):
            ranks=np.empty((3,len(codes)),int)
            for si in range(3):ranks[si,order(sums[si,mi],codes)]=np.arange(1,len(codes)+1)
            mean=sums[:,mi].mean(axis=0);overall=order(mean,codes);meanrank=np.empty(len(codes),int);meanrank[overall]=np.arange(1,len(codes)+1)
            if method!='LR':continue
            frame=desc.copy();frame['scope']=scope;frame['route']=route;frame['method']=method;frame['score_mean']=mean;frame['score_seed_std']=sums[:,mi].std(axis=0,ddof=1);frame['rank']=meanrank;frame['rank_seed_min']=ranks.min(axis=0);frame['rank_seed_max']=ranks.max(axis=0)
            for si,seed in enumerate(SEEDS):frame[f'rank_seed_{seed}']=ranks[si]
            for k in KS:frame[f'top{k}_seed_frequency']=(ranks<=k).sum(axis=0)
            frame=frame.sort_values(['rank','credit_code_normalized'])
            fn=f'{scope}_{route}_LR_U_candidate_top1000.csv';save(out/fn,frame.head(1000));headfiles.append(fn)
            for k in KS:
                h=frame.head(k)
                cutoff=mean[overall[k-1]];ties.append(dict(scope=scope,route=route,top_k=k,cutoff_score=float(cutoff),number_with_exact_cutoff_score=int((mean==cutoff).sum()),seed_frequency_3=int(h[f'top{k}_seed_frequency'].eq(3).sum()),seed_frequency_0=int(h[f'top{k}_seed_frequency'].eq(0).sum())))
                for si,sj in itertools.combinations(range(3),2):
                    common=int(((ranks[si]<=k)&(ranks[sj]<=k)).sum());seedoverlap.append(dict(scope=scope,route=route,top_k=k,seed_a=SEEDS[si],seed_b=SEEDS[sj],intersection=common,retained_fraction=common/k))
                for field in ['region_group','company_size_clean','industry_major_category_clean','company_type_clean','industry_category_clean']:
                    base=desc[field].replace('','<missing>').value_counts(normalize=True)
                    for value,n in h[field].replace('','<missing>').value_counts().items():
                        proportion=n/k;ref=float(base.get(value,0));concentration.append(dict(scope=scope,route=route,top_k=k,field=field,value=value,n=int(n),head_fraction=proportion,U_evaluation_fraction=ref,ratio=proportion/ref if ref else None))
                for field in ['company_age','registered_capital_log1p','insured_count_log1p']:
                    x=pd.to_numeric(h[field],errors='coerce');b=pd.to_numeric(desc[field],errors='coerce')
                    concentration.append(dict(scope=scope,route=route,top_k=k,field=field,value='<numeric>',n=int(x.notna().sum()),head_median=float(x.median()) if x.notna().any() else None,U_evaluation_median=float(b.median()) if b.notna().any() else None,head_missing_fraction=float(x.isna().mean()),U_evaluation_missing_fraction=float(b.isna().mean())))
        for k in KS:
            lists=[]
            for route in ['Gold_only','Gold_Silver']:
                mi=methodkeys.index((route,'LR'));lists.append(set(codes[order(sums[:,mi].mean(axis=0),codes)[:k]]))
            routeoverlap.append(dict(scope=scope,top_k=k,intersection=len(lists[0]&lists[1]),retained_fraction=len(lists[0]&lists[1])/k))
    df=pd.DataFrame(raw);save(out/'fixedK_Gold_capture_by_fold_seed.csv',df)
    pooled=df.groupby(['scope','route','method','top_k','seed'],as_index=False).agg(gold_hits=('gold_hits','sum'),gold_total=('gold_total','sum'))
    assert pooled.gold_total.eq(2484).all();pooled['gold_capture']=pooled.gold_hits/pooled.gold_total;save(out/'fixedK_Gold_capture_by_seed.csv',pooled)
    summary=pooled.groupby(['scope','route','method','top_k'],as_index=False).agg(gold_hits_mean=('gold_hits','mean'),gold_capture_mean=('gold_capture','mean'),gold_capture_seed_min=('gold_capture','min'),gold_capture_seed_max=('gold_capture','max'))
    save(out/'fixedK_Gold_capture_summary.csv',summary);save(out/'LR_U_seed_topK_overlap.csv',seedoverlap);save(out/'LR_U_routes_topK_overlap.csv',routeoverlap);save(out/'LR_U_head_concentration.csv',concentration);save(out/'LR_U_cutoff_and_seed_frequency.csv',ties)
    return summary,ties,routeoverlap

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);ap.add_argument('--e04',type=Path,required=True);ap.add_argument('--e05',type=Path,required=True);ap.add_argument('--e06',type=Path,required=True);ap.add_argument('--taxonomy',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    a.out.mkdir(parents=True);inputs={str(a.taxonomy):sha(a.taxonomy)};checks=[]
    for p in [a.e04/'prepared_001',a.e05/'prepared_001',a.e06/'prepared_001',a.e06/'runs/main_002']:verify_hashes(p,inputs,checks)
    ref=json.loads(a.taxonomy.read_text());frames={}
    for role,name in [('Gold_development','gold_development'),('Silver','silver'),('U2_fit','U_fit_pool'),('U2_evaluation','U_evaluation')]:frames[role]=read(a.e05/'prepared_001'/f'{name}.csv')
    two=set(frames['U2_fit'].credit_code_normalized);u3fit=read(a.e04/'prepared_001/U3_fit_pool.csv')
    frames['U3_increment_fit']=u3fit.loc[~u3fit.credit_code_normalized.isin(two)].reset_index(drop=True);frames['U3_increment_evaluation']=read(a.e04/'prepared_001/U_increment_evaluation.csv')
    reserved=set(read(a.e06/'prepared_001/gold_entity_split.csv').query("role == 'final_holdout'").credit_code_normalized)
    assert len(reserved)==621 and all(not reserved&set(f.credit_code_normalized) for f in frames.values())
    summaries,profiles=industry_audit(a.project,ref,frames,a.out)
    descriptors={'U2':frames['U2_evaluation'],'U3':pd.concat([frames['U2_evaluation'],frames['U3_increment_evaluation']],ignore_index=True)}
    summary,ties,overlap=heads(a.e06/'runs/main_002',a.e06/'prepared_001',descriptors,a.out,inputs,checks)
    checks.append(dict(check='all_outputs_use_development_only_no_new_fits',passed=True));save(a.out/'verification_checks.csv',checks)
    dump(a.out/'manifest.json',dict(status='complete',new_fits=0,final_Gold_scored=False,industry_mapping='Reference audit only; not applied to training; ambiguous/unmatched retained as unknown; no raw data overwritten.',head_definition='Fixed K per fold within Gold_validation plus evaluation U; sum Gold hits across five folds and divide by 2484 per seed, then average 3 seeds. This is 5*K evaluation slots, not a deployment budget or U precision.',candidate_definition='U-only development candidates from existing F0 LR; each seed averages five fold scores; seed frequency counts 3 correlated seeds, not independent trials or probability.',tie_policy='score descending, normalized credit code ascending; exact cutoff ties recorded',taxonomy_reference=ref['reference'],inputs=inputs,code_sha256=sha(Path(__file__))))
    hashes={str(p.relative_to(a.out)):sha(p) for p in a.out.rglob('*') if p.is_file()};dump(a.out/'output_hashes.json',hashes)
    print(summary.query("method == 'LR'").to_string(index=False));print(pd.DataFrame(overlap).to_string(index=False));print(pd.DataFrame(summaries).to_string(index=False));print(pd.DataFrame(ties).query('top_k == 100').to_string(index=False))

if __name__=='__main__':main()
