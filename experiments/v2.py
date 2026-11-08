"""One local entry for V2 verification and explicit, append-only reproduction."""
import argparse, hashlib, json, os, subprocess, sys
from pathlib import Path

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def new_path(p,project):
    project=project.expanduser().resolve()
    p=p.expanduser().resolve()
    if p.exists():raise FileExistsError(f'Output already exists: {p}')
    protected=[project/'data',project/'src',project/'experiments',project/'docs',project/'.git']
    if any(p==r or r in p.parents for r in protected):raise ValueError('Output must not be inside data/src/experiments/docs/.git')
    return p

def verify(project,reg,source_only=False):
    counts={'frozen_files':0,'locked_files':0,'archived_files':0}
    lock=json.loads((project/'experiments/v2_closeout/snapshot_lock.json').read_text())
    for rel,expected in lock['files'].items():
        if digest(project/rel)!=expected:raise ValueError(f'Locked file mismatch: {rel}')
        counts['locked_files']+=1
    if not source_only:
        freeze=json.loads((project/'docs/data_versions/v2_raw_001.json').read_text())
        needed=[project/x['registered_path'] for x in freeze['files']]
        if any(not x.is_file() for x in needed):
            raise SystemExit('Business data are local-only. Use verify --source-only for this public checkout; full replay requires the authorized local data and run archives.')
        for item in freeze['files']:
            if digest(project/item['registered_path'])!=item['sha256']:raise ValueError('Frozen hash mismatch: '+item['registered_path'])
            counts['frozen_files']+=1
        manifests=lock.get('output_hash_manifests',[])
        if lock.get('scope')=='public_source_only':
            manifests=[]
            for record in reg['experiments'].values():
                manifests += [str(Path(record['archive'])/rel/'output_hashes.json') for rel in ['prepared_001',record['formal_run'],record['formal_report']]]
            manifests += [str(Path(reg['acceptance'])/rel/'output_hashes.json') for rel in ['run_001','eda_002']]
            manifests += [str(Path(reg['closeout'])/rel/'output_hashes.json') for rel in reg['closeout_formal'].values()]
        for rel in manifests:
            manifest=project/rel
            if not manifest.is_file():raise SystemExit('Missing local run archive: '+rel)
            for name,expected in json.loads(manifest.read_text()).items():
                if digest(manifest.parent/name)!=expected:raise ValueError(f'Archive hash mismatch: {rel}: {name}')
                counts['archived_files']+=1
    print(json.dumps(dict(status='verified_sources' if source_only else 'verified',**counts,new_fits=0,final_Gold_scored=False),ensure_ascii=False))

def command(project,reg,action,experiment,out,run_override=None,prepared_override=None):
    r=reg['experiments'][experiment];code=project/r['code'];base=project/r['archive'];prep=base/'prepared_001';parent=lambda e:project/reg['experiments'][e]['archive']
    prep=prepared_override or prep
    if action=='train':
        args=[str(code/r['train_script']),'train','--prepared',str(prep),'--out',str(out)]
        if experiment=='E05':args+=['--parent-run',str(parent('E02')/'runs/main_001')]
        if experiment=='E06':args+=['--parent-e04',str(parent('E04')/'runs/main_001'),'--parent-e05',str(parent('E05')/'runs/main_001')]
    elif action=='report':
        args=[str(code/r['report_script']),'--prepared',str(prep),'--run',str(run_override or base/r['formal_run']),'--out',str(out)]
        if experiment=='E04':args+=['--e02-run',str(parent('E02')/'runs/main_001')]
        if experiment=='E05':args+=['--parent-run',str(parent('E02')/'runs/main_001')]
        if experiment=='E06':args+=['--parent-e04',str(parent('E04')/'runs/main_001'),'--parent-e05',str(parent('E05')/'runs/main_001')]
    elif action=='prepare':
        args=[str(code/r['train_script']),'prepare','--project',str(project),'--out',str(out)]
        if experiment in ['E02','E03','E04']:args+=['--upstream',str(project/reg['acceptance']/'eda_002')]
        if experiment in ['E03','E04']:args+=['--e02-prepared',str(parent('E02')/'prepared_001')]
        if experiment=='E05':args+=['--parent',str(parent('E02')/'prepared_001')]
        if experiment=='E06':args+=['--e05-prepared',str(parent('E05')/'prepared_001'),'--e04-prepared',str(parent('E04')/'prepared_001')]
    else:raise ValueError(action)
    return args

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[1]);sub=ap.add_subparsers(dest='action',required=True)
    sub.add_parser('status');sub.add_parser('verify').add_argument('--source-only',action='store_true')
    for action in ['prepare','train','report']:
        p=sub.add_parser(action);p.add_argument('--experiment',choices=['E02','E03','E04','E05','E06'],required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--dry-run',action='store_true')
        if action=='report':p.add_argument('--run',type=Path)
        if action in ['train','report']:p.add_argument('--prepared',type=Path)
    p=sub.add_parser('closeout');p.add_argument('--out',type=Path,required=True)
    p=sub.add_parser('audit-data');p.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();project=a.project.resolve();reg=json.loads((project/'experiments/v2_closeout/registry.json').read_text())
    if a.action=='status':print(json.dumps(reg,ensure_ascii=False,indent=2));return
    if a.action=='verify':verify(project,reg,a.source_only);return
    out=new_path(a.out,project)
    env=os.environ.copy();env['PYTHONHASHSEED']='0'
    def execute(args):subprocess.run([sys.executable,*map(str,args)],cwd=project,env=env,check=True)
    if a.action in ['prepare','train','report']:
        args=command(project,reg,a.action,a.experiment,out,getattr(a,'run',None),getattr(a,'prepared',None))
        if a.dry_run:print(json.dumps([sys.executable,*args],ensure_ascii=False));return
        verify(project,reg);execute(args)
    elif a.action=='closeout':
        verify(project,reg);out.mkdir(parents=True);code=project/'experiments/v2_closeout';reference=project/reg['closeout']/'reference/taxonomy_complete.json'
        execute([code/'analyze_closeout.py','--project',project,'--e04',project/reg['experiments']['E04']['archive'],'--e05',project/reg['experiments']['E05']['archive'],'--e06',project/reg['experiments']['E06']['archive'],'--taxonomy',reference,'--out',out/'analysis'])
        execute([code/'standardize_industry.py','--project',project,'--audit',out/'analysis','--taxonomy',reference,'--out',out/'industry_standardized'])
    elif a.action=='audit-data':
        verify(project,reg);out.mkdir(parents=True);code=project/'experiments/v2_acceptance'
        execute([code/'acceptance.py','--project',project,'--out',out/'audit'])
        execute([code/'eda_next.py','--audit',out/'audit','--out',out/'eligibility'])

if __name__=='__main__':main()
