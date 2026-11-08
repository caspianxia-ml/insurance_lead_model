"""Reconcile all three industry source values on development U intersections."""
import argparse,csv,json
from pathlib import Path
import pandas as pd
from analyze_closeout import norm,read,sha,dump,FIELDS

def clean(x):
    x=norm(x)
    return '' if x.lower() in {'','nan','none','null','nat','未知','-','--'} else x.replace('|','丨')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);ap.add_argument('--audit',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    frame=pd.concat([read(a.audit/f'industry_entities_U2_{part}.csv') for part in ['fit','evaluation']],ignore_index=True)
    expected={r['credit_code_normalized']:[r[f] for f in FIELDS] for r in frame.to_dict('records')};assert len(expected)==18790
    freeze=json.loads((a.project/'docs/data_versions/v2_raw_001.json').read_text());results=[]
    for source in ['U1','U2','U3']:
        meta=next(x for x in freeze['files'] if x['dataset']==source);p=a.project/meta['registered_path'];assert sha(p)==meta['sha256'];seen={}
        with p.open(encoding='utf-8-sig',newline='') as f:
            reader=csv.reader(f,strict=source!='U3');header=next(reader);cc=header.index('credit_code');fields=[header.index(x.replace('_clean','')) for x in FIELDS]
            for row in reader:
                assert len(row)==len(header);code=clean(row[cc]).upper()
                if code in expected:seen.setdefault(code,[]).append([clean(row[i]) for i in fields])
        differing=[c for c,records in seen.items() if any(v!=expected[c] for v in records)]
        assert len(seen)==(11013 if source=='U1' else 18790) and not differing
        results.append(dict(source=source,source_sha256=sha(p),matched_development_U=len(seen),duplicate_source_records=sum(len(v)>1 for v in seen.values()),different_industry_entities=len(differing)))
    dump(a.out,dict(status='complete',all_three_industry_fields_equal=True,results=results,new_fits=0,final_Gold_scored=False,code_sha256=sha(Path(__file__))))
    print(json.dumps(results,ensure_ascii=False))

if __name__=='__main__':main()
