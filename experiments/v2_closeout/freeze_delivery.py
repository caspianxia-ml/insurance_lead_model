"""Lock code, documentation and archive hash manifests without touching data."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from v2 import digest

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();project=a.project.resolve()
    if a.out.exists():raise FileExistsError(a.out)
    reg=json.loads((project/'experiments/v2_closeout/registry.json').read_text());files=set();manifests=[]
    for folder in ['src','experiments/e02_gold_silver','experiments/e03_positive_scope','experiments/e04_unlabeled_scope','experiments/e05_feature_ablation','experiments/e06_nnpu_prior','experiments/v2_acceptance','experiments/v2_closeout']:
        for p in (project/folder).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix in ['.py','.md','.json','.txt'] and p.resolve()!=a.out.resolve():files.add(p)
    files.add(project/'experiments/v2.py');files.add(project/'README.md')
    for p in (project/'docs').glob('*.md'):files.add(p)
    files.add(project/'docs/data_versions/v2_raw_001.json')
    for record in reg['experiments'].values():
        base=project/record['archive']
        for rel in ['prepared_001',record['formal_run'],record['formal_report']]:manifests.append(base/rel/'output_hashes.json')
    for rel in ['run_001','eda_002']:manifests.append(project/reg['acceptance']/rel/'output_hashes.json')
    for rel in reg['closeout_formal'].values():manifests.append(project/reg['closeout']/rel/'output_hashes.json')
    for p in manifests:files.add(p)
    for name in ['source_relationships.json','runtime_versions.json']:files.add(project/reg['closeout']/name)
    obj=dict(version=reg['version'],files={str(p.relative_to(project)):digest(p) for p in sorted(files)},output_hash_manifests=[str(p.relative_to(project)) for p in manifests],purpose='Local reproducibility lock; Git commit and backup bind this file. Not an authenticity signature.')
    a.out.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');print(len(files),'locked files;',len(manifests),'archive hash manifests')

if __name__=='__main__':main()
