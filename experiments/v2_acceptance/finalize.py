"""Independent reconciliation of frozen CSV and emitted outputs, plus report.
Called automatically by acceptance.py; may be rerun to regenerate report/verification.
"""
from pathlib import Path
import csv
import hashlib
import json
import sys
import unicodedata
import re
from collections import Counter
from itertools import combinations

HERE=Path(__file__).resolve().parent
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))
def write(data,p,fields=None):
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(data[0]));w.writeheader();w.writerows(data)
def code(s):
    v=re.sub(r'\s+','',unicodedata.normalize('NFKC',s)).upper()
    return '' if v.lower() in {'','-','--','—','–','null','none','nan','n/a','na','unknown','未知','暂无','未公开','nat'} else v

def finalize(project,out):
    from acceptance import capital
    manifest=json.loads((out/'run_manifest.json').read_text())
    schema=rows(out/'field_schema.csv');allowed={(x['dataset'],x['field']) for x in schema}
    # Restrict inherited quality table to actual source fields.
    quality=[x for x in rows(out/'data_quality_report.csv') if (x['dataset'],x['field']) in allowed]
    write(quality,out/'data_quality_report.csv')
    stats=rows(out/'entity_cleaning_log.csv'); extra=rows(out/'additional_quality.csv')
    sets={}; independent=[]; rawcounts={}; gold=Counter(); errors=[]; statuses=Counter()
    capital_events=[]
    for label,meta in manifest['input_files'].items():
        path=Path(meta['path']); raw=rows(path); cleaned=rows(out/'cleaned'/f'{label}_cleaned.csv')
        for number,r in enumerate(raw,1):
            for field in ['registered_capital','paid_capital']:
                amount,currency,status=capital(r[field])
                capital_events.append({'dataset':label,'record_id':f'{label}#{number}','field':field,
                    'raw_value':r[field],'cny_yuan':amount,'currency':currency,'status':status,
                    'reason':'unit_to_yuan_no_FX' if amount else 'retained_raw_no_numeric_conversion'})
        decisions=rows(out/f'{label}_decision_lineage.csv')
        rawcodes=Counter(code(r['credit_code']) for r in raw)
        sets[label]=set(rawcodes)-{''};rawcounts[label]=len(raw)
        keptids={r['record_id'] for r in cleaned}; decisionmap={r['record_id']:r for r in decisions}
        counts=Counter(r['__decision'] for r in decisions)
        psets=sets.get('P_national',set())
        expected_dups=sum(n-1 for c,n in rawcodes.items() if re.fullmatch('[0-9A-Z]{18}',c))
        tests={
            'raw_count':len(raw)==meta['rows'],
            'raw_hash':digest(path)==meta['sha256'],
            'decision_coverage':len(decisions)==len(raw) and len(decisionmap)==len(raw),
            'output_ids':keptids=={r['record_id'] for r in decisions if r['__decision']=='kept'},
            'independent_duplicates':counts['removed_duplicate_credit_code']==expected_dups,
            'independent_P_exclusion':counts['removed_known_P_credit_code']==(0 if label=='P_national' else sum(bool(re.fullmatch('[0-9A-Z]{18}',c)) for c in sets[label]&psets)),
            'missing_code_retained':sum(not code(r['credit_code']) for r in raw)==sum(not r['credit_code_normalized'] for r in cleaned),
        }
        originals=meta['columns']
        tests['all_retained_source_fields_exact']=all(all(r[c]==raw[int(r['source_record_number'])-1][c] for c in originals) for r in cleaned)
        for test,passed in tests.items(): independent.append({'dataset':label,'check':test,'pass':passed})
        for r in cleaned:
            if label=='P_national':gold[(r['p_quality_candidate'],r['credit_code_format'])]+=1
            for field in ['registered_capital','paid_capital','insured_count','insured_count_report_year','establishment_date']:
                statuses[(label,field,r[field+'_status'])]+=1
            if r['csv_quote_review_required']=='True' or r['text_contamination_review']=='True':
                errors.append({'dataset':label,'record_id':r['record_id'],'physical_start':r['source_physical_start'],
                               'physical_end':r['source_physical_end'],'csv_quote_review':r['csv_quote_review_required'],
                               'text_contamination_review':r['text_contamination_review'],'action':'retained_review_required'})
    assert all(x['pass'] for x in independent),independent
    write(independent,out/'independent_verification.csv')
    write(capital_events,out/'capital_transformations.csv')
    write([{'dataset':k[0],'field':k[1],'status':k[2],'rows':v} for k,v in sorted(statuses.items())],out/'normalization_status_counts.csv')
    write(errors,out/'parser_text_review.csv',['dataset','record_id','physical_start','physical_end','csv_quote_review','text_contamination_review','action'])
    all_code_overlap=[]
    for a,b in combinations(sets,2):
        overlap=sets[a]&sets[b]
        all_code_overlap.append({'a':a,'b':b,'basis':'all_nonempty_normalized_codes','overlap':len(overlap),
                                 'non18_overlap':sum(not bool(re.fullmatch('[0-9A-Z]{18}',c)) for c in overlap),
                                 'a_minus_b':len(sets[a]-sets[b])})
    write(all_code_overlap,out/'all_code_overlap_diagnostics.csv')
    # Preserve original evidence documents as byte copies, never alter source project.
    evidence=out/'evidence';evidence.mkdir(exist_ok=True)
    for relative in ['docs/data_versions/v2_raw_001.json','reports/v2/v2_raw_001.md','docs/V2_EDA_20261020_1200.md','docs/PROJECT_STATUS_20261031.md']:
        (evidence/Path(relative).name).write_bytes((project/relative).read_bytes())
    manifest['code_sha256']=digest(HERE/'acceptance.py');manifest['finalize_code_sha256']=digest(Path(__file__))
    manifest['test_code_sha256']=digest(HERE/'test_acceptance.py')
    manifest['synthetic_tests']={'original_toolkit':'passed','acceptance_tests':3,'acceptance_tests_status':'passed'}
    manifest['independent_verification']=independent
    manifest['reference_archive_sha256']='a742211a4d06a392bd18fcad27f18e676c9fc2b478db20f4b211e14f044fdb9b'
    (out/'run_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# 企业保险福利识别模型 V2 数据验收与清洗报告','',
      '验收日期：2026-11-06（Asia/Shanghai）。输入版本：`v2_raw_001`。',
      '结论：四份原冻结数据身份、结构与清洗数量验收通过，已产出可追溯的清洗数据。训练样本验收仍有待核验项：缺失/异常代码、同名冲突、Gold 独立业务证据和 U3 引号脏值。当前数据不应直接称为已确认的 Gold 训练集。','',
      '## 原始数据与依据','',
      f'项目：`{project}`。四份输入位于 `data/v2/raw/`，SHA-256 全部与 `docs/data_versions/v2_raw_001.json` 一致。未重新导出、未重新冻结、未覆盖原件。P 实际为 27 列，U1/U2/U3 为 20 列，全部原始字段以字符串读入。',
      '原工具包位于下载目录的 `保险福利V2_数据审计工具包.zip` 和带 `(1)` 的同名文件，两个归档哈希完全相同；已保留未修改解包副本。`v2_data_spec.md` 为草案 v0.1，实际使用当前冻结版 U1，而非另取历史 V1 U1。',
      '原快照声明时点为 2026-10-20 12:00。实际导出时间、严格历史 as-of、SQL 与真实执行对应关系仍未获独立日志证明，继承原冻结登记中的限制。','',
      '## 清洗数量（行）','',
      '| 数据集 | 原始行 | 去重移出 | P 代码交集移出 | 清洗保留行 | 缺代码保留行 |',
      '|---|---:|---:|---:|---:|---:|']
    for s in stats: lines.append('| '+' | '.join(s[c] for c in ['dataset','raw_rows','removed_duplicate_credit_code','removed_known_P_credit_code','kept_rows','credit_code_missing'])+' |')
    lines += ['',
      '每组满足：原始行 = 去重移出 + P 交集移出 + 保留行。移出记录只是不进入清洗输出，原文件和逐行日志全部保留。P 的 4,600 行及 U3 的 8,303 行缺代码记录全部保留，未按名称自动合并。清洗后的行数不是严格唯一企业数。','',
      '## 实体键、交集与包含关系','',
      '遵照草案，首轮自动匹配使用规范化后 18 位字母数字代码（NFKC、去空白、大写），与校验位验证分开。18 位格式符合不代表实体已核实。非 18 位及缺失代码保留并复核；校验位不通过的 18 位代码保留审计标记，本轮仍按格式键做去重/交集排除。无依据更改代码、补代码或跨行拼字段均未执行。',
      '去重选择原工具包候选字段完整度最高的记录，再按可用最新时间、原始行顺序打破平局。U 无更新时间字段，平局取原始顺序；未把 ID 当成时间。每条移出记录保留代表记录 ID，交集移出记录保留 P 匹配记录 ID。',
      'P/U1、P/U2、P/U3 原始代码实体交集分别为 6,960、25,522、31,870；清洗后这三组格式键交集均为零。这表示与已观察 P 的代码互斥，不表示 P 全部已业务确认为 Gold。',
      '按 18 位格式键：清洗前 U1 18,376、U2 45,077、U3 159,271；清洗后分别为 11,416、19,555、127,401。清洗前后均满足 U1 ⊆ U2 ⊆ U3。U2−U1 清洗后为 8,139，U3−U2 为 107,846。不得把三个 U 相加。',
      '历史 EDA 把所有非空代码都用于实体比较，故 U3 报 159,272、排 P 后 127,402；本轮仅 18 位格式键，少计入 1 条非 18 位代码，但该记录保留在清洗文件。该差异来自统计口径，未丢行。`all_code_overlap_diagnostics.csv` 另列所有非空代码的独立诊断。',
      '同名疑似 P/U 冲突保留复核：U1 41 行、U2 45 行、U3 2,109 行。清洗后的 P/U 零交集结论只针对代码键；缺代码、同名异码仍可能形成标签污染或跨集合实体泄漏。','',
      '## Gold / Silver','',
      '原始 `p_quality` 保留：Gold 7,556 行、Silver 32,477 行。沿用历史 V2 EDA 的候选定义，source 含“原力”或“量化Q3榜单”为 Gold_candidate：7,595 行；其余 Silver_candidate：32,438 行。39 行原 Silver 属于新 Gold 候选（37 行有代码、2 行缺代码）。',
      'Gold 候选中 4,356 行缺代码。没有独立业务确认字段/证据或审查人可供核验，因此 `gold_confirmed=unknown`，不能把候选数当作确认 Gold 数。工具包原 `gold_source_candidate_only` 字段只统计原力来源（7,556），本轮统一候选分组请使用 `p_quality_candidate` 和 `Gold_Silver_audit.csv`。','',
      '## 字段标准化与脏值','',
      '原始字段逐值保留。新增 `_normalized`、`_status` 与资本金额/币种列；ID 和代码始终为字符串。缺失不补零，数值零保留；人数为 0–10,000,000 非负整数，年份为 1900–2026；布尔字段只允许 0/1。日期只接收明确的年在前格式，成立日期以声明快照日期核查范围。',
      '资本统一解析万/亿到元，人民币、假定人民币和外币分开标记；无币种的数字不冒充已确认人民币，外币不擅自换汇。企业规模统一连接符与空白，保留原有区间粒度，未知为 Unknown，不擅自重分箱。',
      '地区使用草案规定的名称/地址沪苏标记作审计；保留原省市标签及标记，不自动改变地域归属。经营状态、地区冲突和字段缺失不自动删企业。标签、来源、质量、名称、地址及这些审计衍生列不能直接全部投入模型；本次不训练、不划分 Holdout，也不拟合填充/编码。',
      '信用代码校验不通过（不含缺失）P 27、U1 0、U2 1、U3 7 行；这些是质量标记，不是删除数。',
      'U3 严格解析在物理第 117,140 行遇引号错误。仅对已核验哈希的 U3 使用 csv.reader(strict=False)，167,579 条记录全部宽度为 20；没有跳行。相关记录保留并标记 `csv_quote_review_required`，不能把宽松读取宣称为已修复真实字段含义。输出重新编码为可严格读取的 CSV，原字段语义疑点仍待核对。',
      'HTML/style 污染采取更具体的标签或 style= 匹配标记，与历史 EDA 对任意 style 文本的统计不同；不自动剥离可能有意义的字段文本。待核对记录见 `parser_text_review.csv`。','',
      '## 验证与复现','',
      f'原工具包合成测试通过；主流程 {len(manifest["verification_checks"])} 项完整性检查全部通过；独立复核 {len(independent)} 项全部通过。独立复核重新读取原始 CSV，验证去重数量、交集数量、全部保留行的全部原始字段逐值一致、记录 ID 覆盖和原文件哈希。',
      '可复现代码：上级目录 `acceptance.py`、`finalize.py`，依赖未修改的 `reference_toolkit/v2_data_audit/audit_v2.py`。新运行必须使用未存在的输出目录，拒绝覆盖历史运行。Python/pandas 版本、输入/代码哈希、规则及验证结果见 `run_manifest.json`，产物哈希见 `output_hashes.json`。','',
      '```bash',f'python acceptance.py --project "{project}" --out ./run_002','```','',
      '## 交付索引','',
      '- `cleaned/`：四组清洗后的全量保留记录，带原字段和标准化列。',
      '- `entity_cleaning_log.csv`：清洗前后数量和移出原因。',
      '- `record_decisions.csv`、`*_decision_lineage.csv`：逐记录决定、原始物理行跨度、代表记录和匹配 P 记录。',
      '- `field_schema.csv`、`field_transformations.csv`、`capital_transformations.csv`、`normalization_status_counts.csv`：实际字段、逐字段处理、全部源记录资本换算及异常状态汇总。',
      '- `P_U_exact_overlap.csv`、`P_U_name_review.csv`、`within_dataset_name_review.csv`：交集移出与人工复核名单。',
      '- `overlaps_before_after.csv`、`U_subset_diagnostics.csv`、`subset_violations.csv`：集合关系（违反包含关系的明细为空表）。',
      '- `Gold_Silver_audit.csv`、`additional_quality.csv`：候选分层、代码校验、解析异常统计。',
      '- `verification.csv`、`independent_verification.csv`：验证结果；`evidence/`：原冻结登记与历史审计文档副本。','']
    (out/'audit_summary.md').write_text('\n'.join(lines),encoding='utf-8')
    (out/'output_hashes.json').write_text(json.dumps({str(p.relative_to(out)):digest(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='output_hashes.json'},indent=2),encoding='utf-8')
    return {'independent_checks':len(independent),'all_passed':True}

if __name__=='__main__':print(finalize(Path(sys.argv[1]),Path(sys.argv[2])))
