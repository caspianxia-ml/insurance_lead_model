"""Read-only extraction of the supplied schema and official NBS workbook."""
import argparse, hashlib, json, re, unicodedata
from pathlib import Path
from openpyxl import load_workbook

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def norm(x):
    return re.sub(r'\s+', '', unicodedata.normalize('NFKC', str(x or '')))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--nbs',type=Path,required=True);ap.add_argument('--schema',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    wb=load_workbook(a.nbs,read_only=True,data_only=True);nodes=[];section=None
    for row,values in enumerate(wb['Sheet1'].iter_rows(max_col=5,values_only=True),1):
        # Group and its sole class can share a row. Preserve BOTH nodes.
        for value in values[:2]:
            code=str(value).strip() if value is not None else ''
            if not re.fullmatch(r'[A-T]|\d{2,4}',code):continue
            level='section' if code.isalpha() else {2:'division',3:'group',4:'class'}[len(code)]
            if level=='section':section=code
            nodes.append(dict(code=code,name=norm(values[3]),raw_name=str(values[3]),level=level,parent=None if level=='section' else section if level=='division' else code[:-1],sheet='Sheet1',row=row))
    assert nodes and len({n['code'] for n in nodes})==len(nodes)
    codes={n['code'] for n in nodes}
    assert all(n['parent'] is None or n['parent'] in codes for n in nodes)
    wb.close();evidence=[];wb=load_workbook(a.schema,read_only=True,data_only=True)
    for sheet in ['AI外呼筛选字段','全部字段','leads_company']:
        for row,values in enumerate(wb[sheet].iter_rows(values_only=True),1):
            if any('industry_' in str(v) for v in values if v is not None):
                cells={f'{__import__("openpyxl").utils.get_column_letter(i)}{row}':str(v) for i,v in enumerate(values,1) if v is not None}
                evidence.append(dict(sheet=sheet,row=row,cells=cells))
    wb.close()
    result=dict(reference=dict(nbs_source='https://www.stats.gov.cn/sj/tjbz/gmjjhyfl/202302/t20230213_1902780.html',attachment='https://www.stats.gov.cn/sj/tjbz/gmjjhyfl/202302/P020230213403084213497.xlsx',nbs_sha256=sha(a.nbs),schema_sha256=sha(a.schema),interpretation='Official comparison reference only; does not establish upstream ETL version.'),nodes=nodes,schema_evidence=evidence)
    a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    from collections import Counter
    print(json.dumps(dict(nodes=len(nodes),levels=Counter(n['level'] for n in nodes),schema_evidence_rows=len(evidence)),ensure_ascii=False))

if __name__=='__main__':main()
