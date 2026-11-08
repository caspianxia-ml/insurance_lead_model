"""Descriptive field semantics/coverage evidence; no fitting or model selection."""
from pathlib import Path
import argparse
import pandas as pd
from e05 import read_frame
from e02_support import save,dump,verify_hashes,output_hashes,sha

def describe(prepared,out):
    if out.exists():raise FileExistsError(out)
    verify_hashes(prepared);rows=[];counts=[];sources=[]
    for role in ['gold_development','silver','U_fit_pool']:
        frame=read_frame(prepared/f'{role}.csv')
        rows.append({'role':role,'entities':len(frame),
          'industry_medium_nonmissing':float(frame.industry_medium_category_clean.notna().mean()),
          'insured_year_nonmissing':float(frame.insured_report_year_numeric.notna().mean()),
          'introduction_flag_one':float(frame.introduction_flag.eq(1).mean()),
          'category_equal_major':float(frame.industry_category_clean.eq(frame.industry_major_category_clean).mean())})
        for field in ['industry_category_clean','industry_major_category_clean']:
            for value,n in frame[field].fillna('<missing>').value_counts().items():
                counts.append({'role':role,'field':field,'value':value,'entities':int(n)})
        frame['source_group']=frame.source.fillna('').str.contains('量化Q3榜单',regex=False).map({True:'quantitative_list',False:'other_sources'})
        for source,part in frame.groupby('source_group'):
            sources.append({'role':role,'source_group':source,'entities':len(part),'introduction_flag_one':float(part.introduction_flag.eq(1).mean())})
    out.mkdir(parents=True);save(pd.DataFrame(rows),out/'role_field_statistics.csv');save(pd.DataFrame(counts),out/'industry_value_counts.csv');save(pd.DataFrame(sources),out/'source_coverage_statistics.csv')
    dump({'status':'complete','prepared_hash_manifest_sha256':sha(prepared/'output_hashes.json'),'code_sha256':sha(Path(__file__)),
      'scope':'descriptive evidence only; not causal identification, model fitting, validation or feature selection','final_Gold_used':False},out/'manifest.json')
    dump(output_hashes(out),out/'output_hashes.json');print('Field diagnostics saved; no model fitting.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepared',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();describe(a.prepared.resolve(),a.out.resolve())
