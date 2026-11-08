"""Frozen V2 acceptance adapter. Original toolkit remains unchanged beside this file.

Run: python acceptance.py --project /absolute/insurance_lead_model --out NEW_DIR
Requires pandas; refuses existing output directories and verifies frozen hashes.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import importlib.util
import json
import re
import sys
import unicodedata
import warnings
from datetime import datetime
from decimal import Decimal, InvalidOperation
from itertools import combinations
from zoneinfo import ZoneInfo
import pandas as pd
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('original_audit', HERE/'reference_toolkit/v2_data_audit/audit_v2.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
CHARS = '0123456789ABCDEFGHJKLMNPQRTUWXY'
WEIGHTS = (1,3,9,27,19,26,16,17,20,29,25,13,8,24,10,30,28)
parse_info = {}
frames = {}
changes = []
schema = []

def checksum(code):
    return (len(code)==18 and all(c in CHARS for c in code) and
            code[-1] == CHARS[(31-sum(CHARS.index(c)*w for c,w in zip(code[:17],WEIGHTS))%31)%31])

def text(value):
    val = unicodedata.normalize('NFKC', str(value)).strip()
    return '' if val.lower() in audit.NA_STRINGS | {'nat'} else val

def clean_code(value):
    val = re.sub(r'\s+', '', text(value)).upper()
    if not val: return '', 'missing'
    if re.fullmatch(r'[0-9A-Z]{18}',val): return val,'format_18'
    if re.fullmatch(r'[0-9]+(?:\.[0-9]+)?[Ee][+-]?[0-9]+',val) or re.fullmatch(r'[0-9]+\.[0-9]+',val):
        return val,'suspected_excel_scientific_or_decimal'
    return val,'other_format'

def integer(value, lo, hi):
    val = text(value).replace(',','')
    if not val: return '', 'missing'
    try:
        num = Decimal(val)
        if not num.is_finite() or num != num.to_integral_value(): return '', 'invalid_integer'
        if not lo <= num <= hi: return '', 'out_of_range'
        return str(int(num)), 'ok'
    except InvalidOperation: return '', 'unparseable'

def date(value, timestamp=False):
    val = text(value)
    if not val: return '', 'missing'
    # Only unambiguous year-first dates are accepted. Do not guess day/month.
    if not re.fullmatch(r'\d{4}[-/]\d{1,2}[-/]\d{1,2}(?:[ T]\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?)?', val):
        return '', 'unparseable_or_ambiguous'
    try:
        dt = datetime.fromisoformat(val.replace('/','-'))
        if not timestamp and not datetime(1800,1,1) <= dt <= datetime(2026,10,20,23,59,59):
            return '', 'outside_declared_snapshot_range'
        return dt.isoformat(sep=' ') if timestamp else dt.date().isoformat(), 'ok'
    except ValueError: return '', 'unparseable'

def capital(value):
    val = text(value).replace(',','')
    if not val: return '', '', 'missing'
    m = re.fullmatch(r'([+-]?\d+(?:\.\d+)?)(亿|万)?(?:元)?(?:人民币|CNY|RMB)?',val,re.I)
    if not m:
        foreign = ('美元','港元','港币','欧元','日元','英镑','新加坡','加元','澳元','瑞士','台币','韩元','USD','HKD','EUR','$','€')
        return '', '', 'non_cny' if any(t in val for t in foreign) else 'unparseable'
    num = Decimal(m[1]) * {'亿':Decimal(100000000),'万':Decimal(10000),None:Decimal(1)}[m[2]]
    if num < 0: return '', '', 'negative_amount'
    explicit = any(t in val.upper() for t in ['人民币','CNY','RMB'])
    return format(num,'f'), 'CNY' if explicit else 'CNY_assumed', 'ok' if explicit else 'currency_assumed_review'

def read_csv(path):
    error = None
    with path.open(encoding='utf-8-sig',newline='') as f:
        rd = csv.reader(f, strict=True)
        try:
            for _ in rd: pass
        except csv.Error as exc:
            error={'physical_line':rd.line_num,'message':str(exc)}
    if error and path.name != 'V2_U3_20261020_1200.csv':
        raise ValueError(f'Unexpected strict CSV error: {path}: {error}')
    rows=[]; spans=[]
    with path.open(encoding='utf-8-sig',newline='') as f:
        rd=csv.reader(f,strict=not bool(error)); header=next(rd)
        assert len(header)==len(set(header))
        previous=rd.line_num
        for row in rd:
            if len(row)!=len(header): raise ValueError(f'{path}:{rd.line_num}: wrong width')
            spans.append((previous+1,rd.line_num)); previous=rd.line_num; rows.append(row)
    parse_info[str(path)]={'strict_error':error,'parser':'csv.reader(strict=False)' if error else 'csv.reader(strict=True)',
                           'wrong_width_rows':0,'rows':len(rows),'spans':spans}
    return pd.DataFrame(rows,columns=header,dtype='string'),'utf-8-sig'

original_prepare = audit.prepare
original_quality = audit.data_quality
def raw_quality(label,df):
    columns=[x['field'] for x in schema if x['dataset']==label]
    return original_quality(label,df[columns])
def prepare(label,path):
    df,meta=original_prepare(label,path)
    info=parse_info[str(path)]
    df['record_id']=df['__record_id']
    df['source_record_number']=range(1,len(df)+1)
    df['source_physical_start']=[s[0] for s in info['spans']]
    df['source_physical_end']=[s[1] for s in info['spans']]
    df['credit_code_normalized']=df['__credit_code_normalized']
    df['credit_code_format']=df['__code_format']
    df['credit_code_checksum_status']=df['credit_code_normalized'].map(
        lambda c:'valid' if checksum(c) else 'invalid' if c else 'missing')
    df['entity_key']=df['__entity_code'].map(lambda c:'credit:'+c if c else '')
    df['entity_key_basis']='18_character_format_only_not_verified_identity'
    df['entity_review_required']=~df['credit_code_checksum_status'].eq('valid')
    errorline=(info['strict_error'] or {}).get('physical_line',-1)
    df['csv_quote_review_required']=(df['source_physical_start']<=errorline)&(df['source_physical_end']>=errorline)
    rawcols=meta['columns']
    for field in rawcols:
        normalized=df[field].map(text)
        statuses=pd.Series('ok',index=df.index)
        if field=='credit_code': normalized=df['credit_code_normalized'];statuses=df['credit_code_format']
        elif field in ['leads_company_id','mid_company_clean_id']:
            statuses=normalized.map(lambda v:'ok' if re.fullmatch(r'\d+',v) else 'invalid_id_retained' if v else 'missing')
        elif field in ['insured_count','insured_count_report_year','is_shanghai','is_suzhou','has_company_introduction','has_score_details']:
            limits=(1900,2026) if field.endswith('year') else (0,1) if field.startswith(('is_','has_')) else (0,10000000)
            parsed=df[field].map(lambda v:integer(v,*limits));normalized=parsed.map(lambda x:x[0]);statuses=parsed.map(lambda x:x[1])
        elif field in ['created_time','establishment_date']:
            parsed=df[field].map(lambda v:date(v,field=='created_time'));normalized=parsed.map(lambda x:x[0]);statuses=parsed.map(lambda x:x[1])
        elif field in ['registered_capital','paid_capital']:
            parsed=df[field].map(capital)
            df[field+'_cny_yuan']=parsed.map(lambda x:x[0]);df[field+'_currency']=parsed.map(lambda x:x[1]);statuses=parsed.map(lambda x:x[2])
            # Text normalization is separate from amount and currency interpretation.
        elif field=='company_size': normalized=df[field].map(audit.clean_size)
        df[field+'_normalized']=normalized
        df[field+'_status']=statuses
        for idx in df.index[(df[field]!=normalized) | ~statuses.isin(['ok','format_18'])]:
            changes.append({'dataset':label,'record_id':df.at[idx,'__record_id'],'field':field,
                            'raw_value':df.at[idx,field],'normalized_value':normalized.at[idx],
                            'status':statuses.at[idx],'reason':'derived_only_original_unchanged'})
        schema.append({'dataset':label,'field':field,'raw_type':'string','normalized_column':field+'_normalized',
                       'normalized_type':'ISO date/time' if field in ['created_time','establishment_date'] else
                       'integer encoded as string' if field in ['insured_count','insured_count_report_year'] or field.startswith(('is_','has_')) else 'string',
                       'missing_count':int(df[field].map(audit.is_missing).sum())})
    df['text_contamination_review']=df['registered_address'].str.contains(r'<[^>]+>|style\s*=',case=False,regex=True)|df['region_city'].str.contains(r'<[^>]+>|style\s*=',case=False,regex=True)
    if label=='P_national':
        df['p_quality_original']=df['p_quality']
        df['p_quality_candidate']=df['source'].map(lambda s:'Gold_candidate' if '原力' in s or '量化Q3榜单' in s else 'Silver_candidate')
        df['gold_independent_evidence_status']='not_available_in_snapshot'
        df['gold_confirmed']='unknown'
    # Completeness tie-break uses toolkit's actual source fields only (before derived columns).
    frames[label]=df
    meta.update({k:v for k,v in info.items() if k!='spans'})
    return df,meta

def write(rows,path,columns=None):
    pd.DataFrame(rows,columns=columns).to_csv(path,index=False,encoding='utf-8-sig')

def run(project,out):
    if out.exists(): raise FileExistsError(f'Refusing existing run: {out}')
    frozen_path=project/'docs/data_versions/v2_raw_001.json'
    frozen=json.loads(frozen_path.read_text())
    paths={}; inputs=[]
    for item in frozen['files']:
        label='P_national' if item['dataset']=='P_National' else item['dataset']
        p=project/item['registered_path']
        if audit.sha256(p)!=item['sha256']: raise ValueError(f'Frozen SHA mismatch: {p}')
        paths[label]=p;inputs.append({'dataset':label,'path':str(p),'sha256':item['sha256'],'expected_rows':item['data_rows'],'expected_columns':item['columns']})
    parse_info.clear();frames.clear();changes.clear();schema.clear()
    audit.read_csv=read_csv; audit.clean_code=clean_code; audit.prepare=prepare; audit.data_quality=raw_quality
    result=audit.run(paths,out)
    for item in inputs:
        assert len(frames[item['dataset']])==item['expected_rows']
        loaded_meta=json.loads((out/'run_manifest.json').read_text())['input_files'][item['dataset']]
        assert len(loaded_meta['columns'])==item['expected_columns']
    write(changes,out/'field_transformations.csv')
    write(schema,out/'field_schema.csv')
    quality=[]; comparisons=[]; difference=[]; checks=[]; gold=[]
    for label,df in frames.items():
        kept=df[df['__decision'].eq('kept')]
        codes=kept.loc[kept['__entity_code'].ne(''),'__entity_code']
        checks.append({'check':label+'_conservation','pass':len(df)==len(kept)+int((df['__decision']!='kept').sum())})
        checks.append({'check':label+'_unique_formatted_codes','pass':not codes.duplicated().any()})
        checks.append({'check':label+'_missing_code_retained','pass':int(df['__code_format'].eq('missing').sum())==int(kept['__code_format'].eq('missing').sum())})
        for field in ['credit_code_checksum_status','credit_code_format','csv_quote_review_required','text_contamination_review']:
            for value,n in df[field].value_counts().items(): quality.append({'dataset':label,'check':field,'value':value,'rows':int(n)})
        # Extend the toolkit ledger with deterministic winner links and source physical spans.
        winners=kept.loc[kept['__entity_code'].ne('')].set_index('__entity_code')['__record_id'].to_dict()
        all_winners=df.loc[~df['__decision'].eq('removed_duplicate_credit_code') & df['__entity_code'].ne('')].set_index('__entity_code')['__record_id'].to_dict()
        detail=df[['record_id','source_record_number','source_physical_start','source_physical_end','credit_code_normalized','credit_code_format','credit_code_checksum_status','entity_review_required','csv_quote_review_required','__decision','__review_flags']].copy()
        detail['representative_record_id']=df['__entity_code'].map(all_winners).fillna(df['__record_id'])
        detail['matched_P_record_id']=df['__entity_code'].map(frames['P_national'].loc[frames['P_national']['__entity_code'].ne('')].set_index('__entity_code')['__record_id'].to_dict()).fillna('') if label!='P_national' else ''
        detail.to_csv(out/f'{label}_decision_lineage.csv',index=False,encoding='utf-8-sig')
        if label=='P_national':
            for keys,group in df.groupby(['p_quality_original','p_quality_candidate','credit_code_checksum_status'],dropna=False):
                gold.append(dict(zip(['original_quality','candidate_quality','checksum_status'],keys),rows=len(group)))
    write(quality,out/'additional_quality.csv');write(gold,out/'Gold_Silver_audit.csv')
    p=frames['P_national'];pcodes=set(p['__entity_code'])-{''}
    for phase in ['before','after']:
        sets={k:set(df.loc[(df['__decision'].eq('kept') if phase=='after' else pd.Series(True,index=df.index)) & df['__entity_code'].ne(''),'__entity_code']) for k,df in frames.items()}
        for a,b in combinations(sets,2):
            comparisons.append({'phase':phase,'a':a,'b':b,'a_count':len(sets[a]),'b_count':len(sets[b]),'intersection':len(sets[a]&sets[b]),'a_minus_b':len(sets[a]-sets[b]),'b_minus_a':len(sets[b]-sets[a]),'basis':'18_char_format_code'})
        for a,b in [('U1','U2'),('U2','U3'),('U1','U3')]:
            for code in sorted(sets[a]-sets[b]):difference.append({'phase':phase,'a':a,'b':b,'code':code})
        if phase=='after':
            for u in ['U1','U2','U3']:checks.append({'check':u+'_no_formatted_P_overlap','pass':not bool(sets[u]&pcodes)})
    write(comparisons,out/'overlaps_before_after.csv')
    write(difference,out/'subset_violations.csv',['phase','a','b','code'])
    # Verify output row identity and strict parseability; no skipped records.
    for label,df in frames.items():
        with (out/'cleaned'/f'{label}_cleaned.csv').open(encoding='utf-8-sig',newline='') as f:
            reader=csv.DictReader(f,strict=True); ids=[r['record_id'] for r in reader]
        checks.append({'check':label+'_output_record_ids','pass':ids==df.loc[df['__decision'].eq('kept'),'__record_id'].tolist()})
        checks.append({'check':label+'_input_hash_unchanged','pass':audit.sha256(paths[label])==next(x['sha256'] for x in inputs if x['dataset']==label)})
    write(checks,out/'verification.csv')
    assert all(x['pass'] for x in checks),checks
    manifest=json.loads((out/'run_manifest.json').read_text())
    manifest.update({'pipeline':'v2-frozen-acceptance-1.0.0','generated_at_shanghai':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        'frozen_version':'v2_raw_001','freeze_manifest_sha256':audit.sha256(frozen_path),
        'code_sha256':audit.sha256(Path(__file__)),'toolkit_code_sha256':audit.sha256(HERE/'reference_toolkit/v2_data_audit/audit_v2.py'),
        'python':sys.version,'pandas':pd.__version__,'snapshot_limit':'Declared 2026-10-20 12:00; strict historical as-of and actual export time unverified',
        'rules':{'dedup':'18-character alphanumeric code; original toolkit completeness, newest timestamp, then earliest record; no field merging',
                 'missing_invalid':'Retain missing/non-18 code; 18-char checksum failure explicitly flagged, matching remains format-only per spec',
                 'P_U':'Exclude matching 18-char P codes from U; name-only matches retained for review; P labels remain candidates',
                 'Gold':'Original label preserved; source 原力 or 量化Q3榜单 -> candidate only; independent evidence unavailable',
                 'normalization':'Derived fields only; IDs remain strings; integer ranges and ISO date validation; capital yuan and currency separately recorded; no FX conversion',
                 'region':'Toolkit uses name/address for audit; original province/city flags preserved; no region/status filtering',
                 'U3':'Relaxed parsing only for registered U3; all widths checked; affected source record flagged; output strict-readable'},
        'verification_checks':checks})
    (out/'run_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    # Hash every deliverable except this manifest-of-outputs itself.
    outputs={str(f.relative_to(out)):audit.sha256(f) for f in sorted(out.rglob('*')) if f.is_file()}
    (out/'output_hashes.json').write_text(json.dumps(outputs,indent=2),encoding='utf-8')
    from finalize import finalize
    finalize(project,out)
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args(); print(json.dumps(run(args.project.resolve(),args.out.resolve()),ensure_ascii=False,indent=2))
