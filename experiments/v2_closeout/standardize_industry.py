"""User-approved industry merge: standard reference, first valid field wins conflicts.

Separate derivative tables only. No retraining or replacement of old features.
"""
import argparse, csv, json, re
from collections import Counter
from pathlib import Path
import pandas as pd
from analyze_closeout import Taxonomy, norm, read, save, dump, sha, FIELDS, LEVELS

POLICY='First valid standard classification in field order category -> major -> medium; hierarchical > and explicit |/丨 tokens preserve original order. Later incompatible labels are logged and skipped. No fuzzy recruitment-label mapping.'

def resolve(values,taxonomy):
    # Keep possible most-detailed standard nodes; an accepted coarse value
    # constrains all subsequent values, including when their own levels differ.
    accepted=[];events=[]
    def compatible(a,b):
        return taxonomy.ancestor(a,taxonomy.nodes[b]['level'])==b or taxonomy.ancestor(b,taxonomy.nodes[a]['level'])==a
    for field,value in zip(FIELDS,values):
        for position,token in enumerate(re.split(r'[>丨|]',norm(value)),1):
            if not token:continue
            m=taxonomy.match(token)
            # Pure standard codes are accepted only if present in the reference.
            if token in taxonomy.nodes:m=dict(status='exact_code',codes=[token],levels=[taxonomy.nodes[token]['level']])
            codes=m['codes']
            event=dict(field=field,token_position=position,token=token,reference_codes='|'.join(codes))
            if not codes:event['decision']='unmatched_retained_unknown'
            elif not accepted:accepted=codes[:];event['decision']='accepted_first_standard'
            else:
                pairs=[(a,b) for a in accepted for b in codes if compatible(a,b)]
                if not pairs:event['decision']='conflict_ignored_later'
                else:
                    accepted=sorted({a if LEVELS.index(taxonomy.nodes[a]['level'])>=LEVELS.index(taxonomy.nodes[b]['level']) else b for a,b in pairs})
                    event['decision']='accepted_compatible'
            events.append(event)
    result={}
    for level in LEVELS:
        ancestors={taxonomy.ancestor(c,level) for c in accepted};ancestors.discard(None)
        # Resolve a coarse/fine same-name ambiguity only if the reference has
        # exactly one admissible branch (e.g. section K -> division 70).
        for c in accepted:
            if taxonomy.ancestor(c,level) is None:
                ancestors.update(nc for nc,n in taxonomy.nodes.items() if n['level']==level and taxonomy.ancestor(nc,taxonomy.nodes[c]['level'])==c)
        if accepted and len(ancestors)==1:
            code=next(iter(ancestors));result[f'standard_{level}_code']=code;result[f'standard_{level}_name']=taxonomy.nodes[code]['name']
        else:result[f'standard_{level}_code']='';result[f'standard_{level}_name']=''
    result['industry_conflict_resolved_first']=any(e['decision']=='conflict_ignored_later' for e in events)
    result['industry_unmatched_tokens']=sum(e['decision']=='unmatched_retained_unknown' for e in events)
    result['industry_standard_status']='resolved' if result['standard_section_code'] else 'unknown_retained'
    return result,events

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--audit',type=Path,required=True);ap.add_argument('--taxonomy',type=Path,required=True);ap.add_argument('--project',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    a.out.mkdir(parents=True);taxonomy=Taxonomy(json.loads(a.taxonomy.read_text()));inputs={};summaries=[];all_events=[]
    frames={}
    for p in a.audit.glob('industry_entities_*.csv'):
        role=p.stem.replace('industry_entities_','');frames[role]=read(p);inputs[str(p)]=sha(p)
    # U1 is an audited subset view of U2. Match identities and all 3 source
    # industry values before constructing that view; do not splice model rows.
    u2=pd.concat([frames['U2_fit'],frames['U2_evaluation']],ignore_index=True).set_index('credit_code_normalized');assert u2.index.is_unique
    frozen=json.loads((a.project/'docs/data_versions/v2_raw_001.json').read_text());meta=next(x for x in frozen['files'] if x['dataset']=='U1');raw=a.project/meta['registered_path'];assert sha(raw)==meta['sha256'];inputs[str(raw)]=sha(raw)
    rows=[];seen=set();source_mismatches=[]
    with raw.open(encoding='utf-8-sig',newline='') as f:
        for number,r in enumerate(csv.DictReader(f),1):
            code=norm(r['credit_code']).upper()
            if code not in u2.index or code in seen:continue
            seen.add(code);rr=u2.loc[code].to_dict();rr['credit_code_normalized']=code;rr['role']='U1_subset';rr['source_record_number']=number
            def clean_raw(x):
                x=norm(x)
                return '' if x.lower() in {'','nan','none','null','nat','未知','-','--'} else x.replace('｜','丨').replace('|','丨')
            if any(clean_raw(r[field.replace('_clean','')])!=norm(rr[field]) for field in FIELDS):source_mismatches.append(code)
            rows.append(rr)
    assert len(rows)==11013 and not source_mismatches,(len(rows),len(source_mismatches));frames['U1_subset']=pd.DataFrame(rows)
    for role,frame in frames.items():
        outputs=[];counts=Counter()
        for row in frame.to_dict('records'):
            result,events=resolve([row[field] for field in FIELDS],taxonomy)
            outputs.append({**row,**result});counts.update([e['decision'] for e in events])
            for event in events:all_events.append(dict(role=role,credit_code_normalized=row['credit_code_normalized'],**event))
        output=pd.DataFrame(outputs);assert len(output)==len(frame) and not output.credit_code_normalized.duplicated().any()
        save(a.out/f'{role}_standard_industry.csv',output)
        summaries.append(dict(role=role,n_before=len(frame),n_after=len(output),standard_section_known=int(output.standard_section_code.ne('').sum()),standard_division_known=int(output.standard_division_code.ne('').sum()),standard_group_known=int(output.standard_group_code.ne('').sum()),standard_class_known=int(output.standard_class_code.ne('').sum()),unknown_retained=int(output.industry_standard_status.eq('unknown_retained').sum()),conflicts_resolved_first=int(output.industry_conflict_resolved_first.sum()),rows_with_unmatched_tokens=int(output.industry_unmatched_tokens.gt(0).sum()),token_decisions=dict(counts)))
    save(a.out/'standardization_events.csv',all_events);dump(a.out/'summary.json',summaries)
    dump(a.out/'manifest.json',dict(status='complete',user_decision='合并就行，按照标准分类；没有像这种应该是一个公司业务多，取第一个',policy=POLICY,scope='Separate industry derivatives; existing models and historical feature tables unchanged. Unknown rows retained; not scored as final model.',taxonomy_sha256=sha(a.taxonomy),inputs=inputs,code_sha256=sha(Path(__file__)),helper_sha256=sha(Path(__file__).with_name('analyze_closeout.py')),U1_subset_U2=True,U1_source_industry_equal=True,U2_subset_U3='Verified previously; U3 consists of unchanged U2 plus disjoint increment in this audit.',new_fits=0,final_Gold_scored=False))
    dump(a.out/'output_hashes.json',{str(p.relative_to(a.out)):sha(p) for p in a.out.rglob('*') if p.is_file()})
    print(json.dumps(summaries,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
