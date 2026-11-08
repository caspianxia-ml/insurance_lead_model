"""Apply user-confirmed Gold and review exclusions, then descriptive EDA only.
python eda_next.py --audit run_001 --out eda_001
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from collections import defaultdict,Counter
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd
from acceptance import clean_code,checksum
from reference_toolkit.v2_data_audit.audit_v2 import clean_name

BASE_FIELDS=['record_id','source_record_number','company_name','credit_code_normalized','credit_code_checksum_status',
 'csv_quote_review_required','text_contamination_review','registered_address','region_province_normalized','region_city_normalized',
 'business_status_normalized','establishment_date_normalized','registered_capital_cny_yuan','registered_capital_currency','registered_capital_status',
 'paid_capital_cny_yuan','paid_capital_currency','paid_capital_status','insured_count_normalized','insured_count_status',
 'insured_count_report_year_normalized','industry_major_category_normalized','industry_medium_category_normalized',
 'company_type_normalized','company_size_normalized']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(df,p):
    assert not df.columns.duplicated().any()
    df.to_csv(p,index=False,encoding='utf-8-sig')
def run(audit,out):
    if out.exists():raise FileExistsError(out)
    hashes=json.loads((audit/'output_hashes.json').read_text())
    manifest=json.loads((audit/'run_manifest.json').read_text())
    for name in ['run_manifest.json','P_U_name_review.csv']+[f'cleaned/{k}_cleaned.csv' for k in manifest['input_files']]:
        assert sha(audit/name)==hashes[name],name
    names=defaultdict(set);namecounts=Counter()
    for label,meta in manifest['input_files'].items():
        path=Path(meta['path']);assert sha(path)==meta['sha256']
        with path.open(encoding='utf-8-sig',newline='') as f:
            for row in csv.DictReader(f):
                n=clean_name(row['company_name'])
                if n:names[n].add(clean_code(row['credit_code'])[0]);namecounts[n]+=1
    ambiguous={n for n,c in names.items() if len(c)>1 or ('' in c and namecounts[n]>1)}
    review=pd.read_csv(audit/'P_U_name_review.csv',dtype=str,keep_default_na=False)
    reviewids=set(review['record_id'])
    out.mkdir(parents=True);(out/'eligible').mkdir()
    summaries=[];ledgers=[];pools={};header_notes=[]
    for label in manifest['input_files']:
        p=audit/'cleaned'/f'{label}_cleaned.csv'
        # Previous writer emitted the scale-normalized column twice. Verify both
        # copies are identical before selecting one; new outputs require unique headers.
        with p.open(encoding='utf-8-sig',newline='') as f:
            rd=csv.reader(f);header=next(rd);groups=defaultdict(list)
            for i,c in enumerate(header):groups[c].append(i)
            duplicate={k:v for k,v in groups.items() if len(v)>1}
            for row in rd:
                for positions in duplicate.values():assert len({row[i] for i in positions})==1
        header_notes.append({'dataset':label,'redundant_identical_headers':list(duplicate),'action':'verify_identical_then_select_one'})
        fields=BASE_FIELDS+(['p_quality_original','source'] if label=='P_national' else [])
        df=pd.read_csv(p,dtype=str,keep_default_na=False,usecols=lambda c:c in fields)[fields]
        reasons=pd.Series('',index=df.index)
        def flag(mask,reason):
            nonlocal reasons
            reasons.loc[mask]=reasons.loc[mask].map(lambda x:x+'|'+reason if x else reason)
        flag(df.credit_code_normalized.eq(''),'missing_credit_code')
        flag(df.credit_code_normalized.ne('') & ~df.credit_code_checksum_status.eq('valid'),'invalid_credit_code')
        normalized_names=df.company_name.map(clean_name)
        flag(normalized_names.isin(ambiguous),'unresolved_name_code_conflict')
        flag(df.record_id.isin(reviewids),'unresolved_P_name_conflict')
        flag(df.csv_quote_review_required.eq('True'),'unresolved_CSV_quote')
        # A dirty address is audited; it does not become a modelling feature.
        # No row is excluded for ordinary missing profile attributes or operating status.
        if label=='P_national':
            df['p_quality']=df.p_quality_original
            quantitative=df.source.str.contains('量化Q3榜单',regex=False)
            df.loc[quantitative,'p_quality']='Gold'
            df['gold_confirmed']=df.p_quality.map({'Gold':'true','Silver':'false'}).fillna('unknown')
            df['gold_confirmation_basis']=df.p_quality_original.map({'Gold':'user_confirmed_historical_business_verification','Silver':'original_Silver_label'}).fillna('unknown_label')
            df.loc[quantitative,'gold_confirmation_basis']='user_prior_decision_quantitative_list_is_Gold'
            flag(~df.p_quality_original.isin(['Gold','Silver']),'unknown_P_quality')
        keep=reasons.eq('');eligible=df.loc[keep].copy();pools[label]=eligible
        save(eligible,out/'eligible'/f'{label}.csv')
        ledger=df[['record_id','company_name','credit_code_normalized']].copy();ledger.insert(0,'dataset',label)
        ledger['eligible_for_EDA_and_training']=keep;ledger['exclusion_reasons']=reasons
        ledgers.append(ledger)
        summaries.append({'dataset':label,'cleaned_rows':len(df),'excluded_rows':int((~keep).sum()),'eligible_rows':len(eligible),
            'Gold_eligible':int(eligible.p_quality.eq('Gold').sum()) if label=='P_national' else '',
            'Silver_eligible':int(eligible.p_quality.eq('Silver').sum()) if label=='P_national' else ''})
    save(pd.DataFrame(summaries),out/'population_summary.csv');save(pd.concat(ledgers),out/'eligibility_decisions.csv')
    reasons=[]
    for ledger in ledgers:
        c=Counter(reason for val in ledger.exclusion_reasons if val for reason in val.split('|'))
        for r,n in c.items():reasons.append({'dataset':ledger.dataset.iloc[0],'reason':r,'rows':n,'note':'reasons_can_overlap'})
    save(pd.DataFrame(reasons),out/'exclusion_reason_counts.csv')
    # Both sides of unresolved names are excluded globally, so subset membership
    # is checked again rather than inferred from the earlier audit.
    codes={k:set(v.credit_code_normalized) for k,v in pools.items()};checks=[]
    for k,df in pools.items():
        checks += [{'check':k+'_unique_codes','pass':not df.credit_code_normalized.duplicated().any()},
                   {'check':k+'_checksum_valid','pass':all(checksum(c) for c in df.credit_code_normalized)},
                   {'check':k+'_no_ambiguous_name','pass':not df.company_name.map(clean_name).isin(ambiguous).any()}]
        if k!='P_national':checks.append({'check':k+'_no_P_overlap','pass':not bool(codes[k]&codes['P_national'])})
    subsets=[]
    for a,b in [('U1','U2'),('U2','U3'),('U1','U3')]:
        subsets.append({'a':a,'b':b,'a_minus_b':len(codes[a]-codes[b]),'subset':codes[a]<=codes[b]})
    save(pd.DataFrame(subsets),out/'subset_after_eligibility.csv')
    assert all(x['pass'] for x in checks)
    save(pd.DataFrame(checks),out/'verification.csv')
    p=pools['P_national'];groups={'P_all':p,'P_Gold':p[p.p_quality.eq('Gold')],'P_Silver':p[p.p_quality.eq('Silver')],
      **{k:v for k,v in pools.items() if k!='P_national'}}
    # U groups overlap; increments avoid counting the same entity repeatedly.
    groups['U2_minus_U1']=pools['U2'][~pools['U2'].credit_code_normalized.isin(codes['U1'])]
    groups['U3_minus_U2']=pools['U3'][~pools['U3'].credit_code_normalized.isin(codes['U2'])]
    miss=[];numeric=[];cats=[]
    numcols=['company_age_years','registered_capital_cny_yuan','paid_capital_cny_yuan','insured_count_normalized']
    catcols=['industry_major_category_normalized','industry_medium_category_normalized','company_type_normalized','company_size_normalized','business_status_normalized','region_province_normalized']
    for group,df in groups.items():
        df=df.copy()
        dates=pd.to_datetime(df.establishment_date_normalized,errors='coerce')
        df['company_age_years']=(pd.Timestamp('2026-10-20')-dates).dt.days/365.25
        for field in numcols+catcols:
            series=df[field]
            if field in numcols:
                series=pd.to_numeric(series,errors='coerce')
                if field.endswith('_cny_yuan'):
                    prefix=field.removesuffix('_cny_yuan')
                    series=series.where(df[prefix+'_status'].eq('ok') & df[prefix+'_currency'].eq('CNY'))
                missing=series.isna()
                valid=series.dropna()
                numeric.append({'group':group,'field':field,'rows':len(df),'valid_n':len(valid),
                    **{k:(float(valid.quantile(q)) if len(valid) else '') for k,q in [('p10',.1),('median',.5),('p90',.9)]}})
            else:
                missing=series.isin(['','Unknown'])
                for val,n in series.mask(missing,'<missing>').value_counts().items():
                    cats.append({'group':group,'field':field,'category':val,'rows':int(n),'share':int(n)/len(df) if len(df) else 0})
            miss.append({'group':group,'field':field,'rows':len(df),'unavailable_n':int(missing.sum()),'unavailable_rate':float(missing.mean()) if len(df) else ''})
    save(pd.DataFrame(miss),out/'feature_missingness.csv');save(pd.DataFrame(numeric),out/'numeric_distribution.csv');save(pd.DataFrame(cats),out/'categorical_distribution.csv')
    policy={'generated_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),'upstream_audit':str(audit.resolve()),
      'input_version':'v2_raw_001','user_decision':'Gold is historically business-confirmed; unresolved code conflicts excluded from EDA and training',
      'Gold_rule':'Original p_quality=Gold OR source contains 量化Q3榜单 per explicit prior user decision; original labels retained; Gold business-confirmed per current user',
      'exclude':'missing/invalid checksum codes; globally ambiguous same-name code groups (both sides); P name review; unresolved CSV quote; unknown P label',
      'retain':'raw and cleaned history untouched; ordinary missing profile fields and operating status do not trigger row exclusion',
      'model_warning':'Eligible means identity-screened only; no model fit, imputation, encoding or holdout split yet; source/labels/IDs/region/free text are not predictive features',
      'currency':'EDA CNY amount uses explicit CNY only; assumed-currency and foreign amounts count unavailable',
      'input_header_repair':header_notes,'python_pandas':pd.__version__,'code_sha256':sha(Path(__file__)),
      'upstream_output_hashes_sha256':sha(audit/'output_hashes.json'),'checks':checks}
    (out/'manifest.json').write_text(json.dumps(policy,ensure_ascii=False,indent=2))
    lines=['# V2 下一阶段：样本准入与 EDA','',
      'Gold 口径已按用户当前及历史确认更正：历史 Gold 是经过业务确认的正例；用户还明确量化榜单也应标 Gold，因此原 Gold 加量化Q3榜单为本轮有效 Gold。原始标签另列保留。先前“只认原始 Gold、不升级榜单”和“Gold 证据未知”的表述由本次口径替代。','',
      '## 本轮完成','',
      '原始数据及上一轮清洗数据保留。另建 EDA/训练准入池：代码缺失、代码校验失败、同名异码/缺码的疑似冲突涉及的两侧企业，以及未确认的 CSV 引号异常记录均不进入分析或训练。不是把这些企业从原始档案删除。','',
      '| 数据集 | 上轮清洗行 | 暂不准入 | 可用于 EDA/后续训练的行 |','|---|---:|---:|---:|']
    for x in summaries:lines.append(f"| {x['dataset']} | {x['cleaned_rows']:,} | {x['excluded_rows']:,} | {x['eligible_rows']:,} |")
    ps=summaries[0];lines += ['',f"P 中当前准入 Gold {ps['Gold_eligible']:,} 家，Silver {ps['Silver_eligible']:,} 家。历史 Gold 因缺代码等原因未准入的记录仍然是 Gold，仅暂不参与本轮研究。",'',
      '准入池已重新核验信用代码唯一、校验通过、P/U 互斥，以及三组 U 的包含关系，结果见 verification.csv 和 subset_after_eligibility.csv。','',
      '## EDA 结果','',
      '已分别分析 Gold、Silver、P 总体、U1/U2/U3 和两个增量 U；未把 U 当真实负例。资本数值只用币种明确的人民币，字段缺失与真实零分开。',
      '| 组别 | 参保人数有效值中位数 | 行业大类不可用率 | 实缴资本不可用率 |','|---|---:|---:|---:|']
    for group in ['P_Gold','P_Silver','P_all','U1','U2','U3']:
        n=next(x for x in numeric if x['group']==group and x['field']=='insured_count_normalized')
        ind=next(x for x in miss if x['group']==group and x['field']=='industry_major_category_normalized')
        paid=next(x for x in miss if x['group']==group and x['field']=='paid_capital_cny_yuan')
        lines.append(f"| {group} | {n['median']} | {ind['unavailable_rate']:.1%} | {paid['unavailable_rate']:.1%} |")
    lines += ['','完整结果见 feature_missingness.csv、numeric_distribution.csv、categorical_distribution.csv。差异可能来自采集覆盖和样本选择，不能直接解释成保险福利效应。','',
      '## 下一步','',
      '最新决定：E01 独立复现暂缓，已有 V1.1 四模型和验证结果保留为历史基准。下一组正式实验为 E02 Gold/Silver 对照。',
      '1. 先统一 E02 的最终准入规则（含旧经营状态过滤）、旧特征方案及 Gold 留出/开发折。当前身份准入池不是最终训练集。训练折内拟合缺失填充与编码。',
      '2. 固定全国 U2、特征和评价设计，比较 Gold-only 与 Gold+Silver；必要的同口径比较组训练属于 E02，不额外安排独立 E01。随后再比较 U1/U2/U3。U 是未标注企业，不作真实负例。',
      '3. 检查留出 Gold 的 Top 10%/20% 捕获率、跨来源与地区的稳定性，并对比 Gold-only 和 Gold+Silver，判断 Silver 是否改善排序。不能直接汇报为真实准确率。',
      '4. 缺失填充、编码、选特征和任何模型调参只使用训练部分；固定留出集仅用于最终评估。本轮没有训练模型。','',
      '复现：python eda_next.py --audit run_001 --out eda_002（新输出目录必须不存在）。','']
    (out/'EDA进度与下一步.md').write_text('\n'.join(lines),encoding='utf-8')
    (out/'output_hashes.json').write_text(json.dumps({str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()},indent=2))
    print(json.dumps(summaries,ensure_ascii=False,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--audit',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();run(args.audit,args.out)
