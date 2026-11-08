"""Independent E05 checks and prespecified feature contrasts; no model fitting."""
from pathlib import Path
import argparse,json,math,os,tempfile
import numpy as np
import pandas as pd
from e05 import (SPECS,VARIANTS,ALL_VARIANTS,ROUTES,CONTRASTS,read_frame)
from e02_support import sha,code_hash,save,dump,verify_hashes,output_hashes,read_features,FEATURES

def report(prepared,run,parent,out):
    if out.exists():raise FileExistsError(out)
    for root in [prepared,run,parent]:verify_hashes(root)
    protocol=json.loads((run/'manifest.json').read_text())
    assert protocol['status']=='complete' and protocol['stage']=='full_development' and not protocol['holdout_scored']
    assert sha(Path(__file__).with_name('e05.py'))==protocol['code_sha256']
    assert sha(Path(__file__).with_name('e02_support.py'))==protocol['support_code_sha256']
    assert sha(prepared/'manifest.json')==protocol['prepared_manifest_sha256']
    assert sha(parent/'output_hashes.json')==protocol['parent_E02_hashes_sha256']
    gold=read_frame(prepared/'gold_development.csv');silver=read_frame(prepared/'silver.csv');ufit=read_frame(prepared/'U_fit_pool.csv');ueval=read_frame(prepared/'U_evaluation.csv')
    split=pd.read_csv(prepared/'gold_entity_split.csv',dtype={'credit_code_normalized':str});fmap=split.set_index('credit_code_normalized').development_fold
    reserved=set(split.loc[split.role.eq('final_holdout'),'credit_code_normalized'])
    logs=pd.read_csv(run/'fit_log.csv');metrics=pd.read_csv(run/'cv_metrics.csv');reused=pd.read_csv(run/'B0_reuse_log.csv')
    oldlogs=pd.read_csv(parent/'fit_log.csv');old_samples=pd.read_csv(parent/'paired_U_samples.csv',dtype={'credit_code_normalized':str})
    checks=[]
    def check(name,passed,detail=''):
        checks.append({'check':name,'passed':bool(passed),'detail':str(detail)})
        if not passed:raise AssertionError(name+': '+str(detail))
    check('all_input_and_run_hashes',True)
    check('2700_converged_new_fits',len(logs)==2700 and logs.converged.all() and logs.max_n_iter.lt(1000).all())
    check('621_final_Gold_and_no_role_overlap',len(reserved)==621 and not reserved&(set(gold.credit_code_normalized)|set(silver.credit_code_normalized)|set(ufit.credit_code_normalized)|set(ueval.credit_code_normalized)))
    check('new_features_no_final_Gold_file',not any('holdout' in p.name.lower() for p in prepared.glob('*.csv')))
    pm=json.loads((prepared/'manifest.json').read_text())
    # Prefer the supplied E02 archive's sibling preparation directory. Its hashes
    # must still match the recorded lineage; this survives workspace relocation.
    sibling=parent.parent.parent/'prepared_001'
    parentprepared=sibling if sibling.is_dir() else Path(pm['parent_prepared'])
    verify_hashes(parentprepared)
    check('parent_prepared_lineage',sha(parentprepared/'manifest.json')==pm['parent_manifest_sha256'] and sha(parentprepared/'output_hashes.json')==pm['parent_hash_manifest_sha256'])
    for name,frame in [('gold_development',gold),('silver',silver),('U_fit_pool',ufit),('U_evaluation',ueval)]:
        old=read_features(parentprepared/f'{name}.csv')
        check(name+'_same_entities_and_order',frame.credit_code_normalized.tolist()==old.credit_code_normalized.tolist())
        same=True
        for field in FEATURES:
            a=frame[field];b=old[field]
            same=same and (np.array_equal(a.to_numpy(dtype=float),b.to_numpy(dtype=float),equal_nan=True) if field in ['company_age','registered_capital_log1p','paid_capital_log1p','insured_count_log1p'] else a.fillna('<NA>').tolist()==b.fillna('<NA>').tolist())
        check(name+'_unchanged_old_features',same)
    indices={};predictions=[];usum=np.zeros((len(ALL_VARIANTS)*2,len(ueval)));count=0
    expectedkeys=[(v,r) for v in ALL_VARIANTS for r in ROUTES]
    for path in sorted((run/'prediction_index').glob('*.csv')):
        fold=int(path.stem.split('_')[1]);index=pd.read_csv(path,dtype=str,keep_default_na=False)
        wanted=gold.loc[gold.credit_code_normalized.map(fmap).eq(fold),'credit_code_normalized'].tolist()
        check(f'fold{fold}_index_order_and_members',index.credit_code_normalized.tolist()==ueval.credit_code_normalized.tolist()+wanted)
        check(f'fold{fold}_unique_indices',not index.credit_code_normalized.duplicated().any() and index.prediction_row.astype(int).tolist()==list(range(len(index))))
        check(f'fold{fold}_final_Gold_isolated',not set(index.credit_code_normalized)&reserved)
        check(f'fold{fold}_kind_count',index.kind.eq('U_evaluation').sum()==11274 and index.kind.eq('Gold_validation').sum()==len(wanted))
        indices[fold]=index
    for path in sorted((run/'predictions').glob('*.npz')):
        with np.load(path,allow_pickle=False) as data:
            fold=int(data['fold']);seed=int(data['seed']);scores=data['mean'];std=data['std'];index=indices[fold]
            check(path.stem+'_index_hash',str(data['index_sha256'].item())==sha(run/'prediction_index'/f'fold_{fold}.csv'))
            check(path.stem+'_key_order',list(zip(data['variants'].tolist(),data['routes'].tolist()))==expectedkeys)
            check(path.stem+'_float64_format',scores.dtype==std.dtype==np.float64 and scores.shape==std.shape==(20,len(index)))
            check(path.stem+'_finite',np.isfinite(scores).all() and np.isfinite(std).all() and (std>=0).all())
        gm=index.kind.eq('Gold_validation').to_numpy();codes=index.credit_code_normalized.to_numpy(dtype=str)
        usum+=scores[:,:len(ueval)];count+=1
        for key,(variant,route) in enumerate(expectedkeys):
            order=np.lexsort((codes,-scores[key]));ranks=np.empty(len(index),dtype=int);ranks[order]=np.arange(1,len(index)+1)
            rows=metrics.loc[metrics.fold.eq(fold)&metrics.seed.eq(seed)&metrics.variant.eq(variant)&metrics.route.eq(route)]
            check(f'{path.stem}_{variant}_{route}_metric_rows',len(rows)==4)
            for row in rows.to_dict('records'):
                k=math.ceil(len(index)*row['top_fraction']);hits=int(gm[order[:k]].sum())
                check(f'{path.stem}_{variant}_{route}_{row["top_fraction"]}',k==row['top_k'] and hits==row['captured_Gold'] and int(gm.sum())==row['validation_Gold'] and len(index)==row['evaluation_rows'] and np.isclose(hits/gm.sum(),row['capture_rate']))
            frame=index.loc[gm,['credit_code_normalized','company_name','region_group','source','company_size_clean','industry_major_category_clean']].copy()
            frame['source_group']=np.where(frame.source.str.contains('量化Q3榜单',regex=False),'quantitative_list','historical_Gold')
            frame['variant']=variant;frame['route']=route;frame['fold']=fold;frame['seed']=seed;frame['score']=scores[key,gm];frame['rank']=ranks[gm];frame['evaluation_rows']=len(index)
            predictions.append(frame)
            if variant=='B0_reused':
                oldarm='A_Gold_only' if route=='Gold_only' else 'B_Gold_Silver'
                oldpath=parent/'predictions'/f'fold_{fold}_seed_{seed}_{oldarm}.csv'
                old=pd.read_csv(oldpath,dtype={'credit_code_normalized':str},float_precision='round_trip').set_index('credit_code_normalized').loc[index.credit_code_normalized]
                check(f'{path.stem}_{route}_B0_exact_reuse',np.array_equal(scores[key],old.score.to_numpy()) and np.array_equal(std[key],old.repeat_score_std.to_numpy()))
                rec=reused.loc[reused.fold.eq(fold)&reused.seed.eq(seed)&reused.route.eq(route)]
                check(f'{path.stem}_{route}_reuse_source',len(rec)==1 and rec.sha256.iloc[0]==sha(oldpath) and not rec.reference_only.any())
    check('15_prediction_archives',count==15)
    for (fold,seed,repeat),group in logs.groupby(['fold','seed','repeat']):
        traincodes=set(gold.loc[gold.credit_code_normalized.map(fmap).ne(fold),'credit_code_normalized']);m=len(traincodes)
        parentgroup=oldlogs.loc[oldlogs.fold.eq(fold)&oldlogs.seed.eq(seed)&oldlogs['repeat'].eq(repeat)]
        sample=old_samples.loc[old_samples.fold.eq(fold)&old_samples.seed.eq(seed)&old_samples['repeat'].eq(repeat),'credit_code_normalized']
        check(f'group_{fold}_{seed}_{repeat}_common_U_and_budget',len(group)==18 and len(sample)==m and len(set(sample))==m and set(sample)<=set(ufit.credit_code_normalized)
          and group.U_sample_hash.eq(code_hash(sample)).all() and parentgroup.U_sample_hash.eq(code_hash(sample)).all()
          and group.U_rows.eq(m).all() and np.allclose(group.positive_weight_sum,m) and np.allclose(group.U_weight_sum,m) and np.allclose(group.total_weight,2*m))
        check(f'group_{fold}_{seed}_{repeat}_same_target_and_seed',group.target_entity_hash.eq(code_hash(indices[fold].credit_code_normalized)).all() and group.sample_seed.eq(seed+fold*10000+repeat-1).all())
        for row in group.to_dict('records'):
            p=traincodes if row['route']=='Gold_only' else traincodes|set(silver.credit_code_normalized)
            check(f'{fold}_{seed}_{repeat}_{row["variant"]}_{row["route"]}_P_and_spec',row['positive_entity_hash']==code_hash(p) and row['positive_rows']==len(p) and not p&reserved
              and row['feature_fields']=='|'.join(SPECS[row['variant']].all))
    gp=pd.concat(predictions,ignore_index=True);coverage=gp.groupby(['variant','route','seed']).credit_code_normalized.agg(['count','nunique'])
    check('all_Gold_OOF_coverage',len(coverage)==60 and coverage['count'].eq(2484).all() and coverage['nunique'].eq(2484).all())
    check('1200_capture_metrics',len(metrics)==1200)
    out.mkdir(parents=True);save(pd.DataFrame(checks),out/'independent_verification.csv');save(gp,out/'Gold_development_OOF_predictions.csv')
    # Archive the exact reused draw log as evidence, without generating new draws.
    save(old_samples,out/'E02_replayed_U_samples.csv')
    summarize(prepared,run,out,metrics,gp,ueval,usum/count,len(checks))

def summarize(prepared,run,out,metrics,gp,ueval,umean,nchecks):
    byseed=metrics.groupby(['variant','route','seed','top_fraction'],as_index=False).agg(captured_Gold=('captured_Gold','sum'),Gold_n=('validation_Gold','sum'))
    byseed['capture_rate']=byseed.captured_Gold/byseed.Gold_n;save(byseed,out/'capture_by_seed.csv')
    summary=byseed.groupby(['variant','route','top_fraction'],as_index=False).agg(mean_capture=('capture_rate','mean'),min_capture=('capture_rate','min'),max_capture=('capture_rate','max'));save(summary,out/'capture_summary.csv')
    paired=[];pooled=[]
    for high,low,name in CONTRASTS:
        a=metrics[metrics.variant.eq(high)].merge(metrics[metrics.variant.eq(low)],on=['route','fold','seed','top_fraction'],suffixes=('_added','_base'),validate='one_to_one');a['contrast']=name;a['difference_pp']=100*(a.capture_rate_added-a.capture_rate_base);paired.append(a)
        b=byseed[byseed.variant.eq(high)].merge(byseed[byseed.variant.eq(low)],on=['route','seed','top_fraction'],suffixes=('_added','_base'),validate='one_to_one');b['contrast']=name;b['difference_pp']=100*(b.capture_rate_added-b.capture_rate_base);pooled.append(b)
    paired=pd.concat(paired,ignore_index=True);pooled=pd.concat(pooled,ignore_index=True)
    save(paired,out/'paired_feature_differences.csv');save(pooled,out/'pooled_feature_differences_by_seed.csv')
    effects=pooled.groupby(['route','contrast','top_fraction'],as_index=False).agg(mean_difference_pp=('difference_pp','mean'),min_seed_difference_pp=('difference_pp','min'),max_seed_difference_pp=('difference_pp','max'));save(effects,out/'feature_effect_summary.csv')
    directions=paired.groupby(['route','contrast','top_fraction'],as_index=False).agg(improved=('difference_pp',lambda x:int((x>0).sum())),ties=('difference_pp',lambda x:int((x==0).sum())),comparisons=('difference_pp','size'),min_difference_pp=('difference_pp','min'),max_difference_pp=('difference_pp','max'));save(directions,out/'feature_effect_directions.csv')
    subgroup=[]
    for field in ['region_group','source_group','company_size_clean','industry_major_category_clean']:
        for fraction in [.1,.2]:
            frame=gp.copy();frame['captured']=frame['rank'].le(np.ceil(frame.evaluation_rows*fraction));frame['group']=frame[field].replace('', '<missing>')
            rows=frame.groupby(['variant','route','seed','group'],as_index=False).agg(Gold_n=('captured','size'),captured_Gold=('captured','sum'))
            rows['group_field']=field;rows['top_fraction']=fraction;rows['capture_rate']=rows.captured_Gold/rows.Gold_n;subgroup.append(rows)
    sub=pd.concat(subgroup,ignore_index=True);save(sub,out/'subgroup_capture_by_seed.csv')
    subsum=sub.groupby(['variant','route','group_field','group','top_fraction'],as_index=False).agg(Gold_n=('Gold_n','first'),mean_capture=('capture_rate','mean'));save(subsum,out/'subgroup_capture_summary.csv')
    subdiff=[]
    for high,low,name in CONTRASTS:
        a=subsum[subsum.variant.eq(high)].merge(subsum[subsum.variant.eq(low)],on=['route','group_field','group','top_fraction'],suffixes=('_added','_base'),validate='one_to_one');a['contrast']=name;a['difference_pp']=100*(a.mean_capture_added-a.mean_capture_base);subdiff.append(a)
    save(pd.concat(subdiff,ignore_index=True),out/'subgroup_feature_differences.csv')
    overlap=[];details=[];corr=[];keymap={(v,r):i for i,(v,r) in enumerate((v,r) for v in ALL_VARIANTS for r in ROUTES)}
    for route in ROUTES:
        wide=ueval[['credit_code_normalized','company_name','region_group']].copy();ranks={}
        for variant in ALL_VARIANTS:
            values=umean[keymap[variant,route]];order=np.lexsort((wide.credit_code_normalized.to_numpy(dtype=str),-values));rank=np.empty(len(wide),dtype=int);rank[order]=np.arange(1,len(wide)+1)
            wide[variant+'_score']=values;wide[variant+'_rank']=rank;ranks[variant]=rank
        save(wide,out/f'{route}_U_development_scores_and_ranks.csv')
        for high,low,name in CONTRASTS:
            corr.append({'route':route,'contrast':name,'Spearman_rank_correlation':float(pd.Series(ranks[high]).corr(pd.Series(ranks[low]),method='pearson'))})
            for k in [100,500,1000]:
                added=ranks[high]<=k;base=ranks[low]<=k;intersection=int((added&base).sum())
                overlap.append({'route':route,'contrast':name,'top_k':k,'intersection':intersection,'retained_fraction':intersection/k})
                if k==100:
                    d=wide.loc[added^base,['credit_code_normalized','company_name',high+'_rank',low+'_rank']].copy();d['route']=route;d['contrast']=name;d['direction']=np.where(added[added^base],'added_only','base_only');details.append(d.rename(columns={high+'_rank':'added_rank',low+'_rank':'base_rank'}))
    save(pd.DataFrame(overlap),out/'TopK_feature_overlap.csv');save(pd.DataFrame(corr),out/'feature_rank_correlations.csv');save(pd.concat(details,ignore_index=True),out/'Top100_feature_difference_details.csv')
    draw(out,effects)
    narrative(prepared,run,out,summary,effects,directions,nchecks)

def draw(out,effects):
    os.environ.setdefault('MPLCONFIGDIR',tempfile.mkdtemp(prefix='e05-mpl-'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=[('size','Size'),('paid_capital','Paid capital'),('industry_with_size','Industry (diagnostic)'),('insured_year','Report year (diagnostic)'),('address_geography','Address geography'),('introduction_coverage','Introduction flag')]
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained',sharex=True)
    for ax,route,title in zip(axes,ROUTES,['Gold only','Gold + Silver']):
        for offset,fraction,color in [(-.17,.1,'#3477ad'),(.17,.2,'#d28a32')]:
            vals=[float(effects.loc[effects.route.eq(route)&effects.contrast.eq(name)&effects.top_fraction.eq(fraction),'mean_difference_pp'].iloc[0]) for name,label in labels]
            ax.barh(np.arange(len(labels))+offset,vals,.33,label=f'Top {fraction:.0%}',color=color)
        ax.set_yticks(range(len(labels)),[label for name,label in labels]);ax.invert_yaxis();ax.axvline(0,color='#555',linewidth=.8);ax.set_title(title);ax.set_xlabel('Paired change in Gold capture (percentage points)');ax.legend(frameon=False);ax.spines[['top','right']].set_visible(False)
    fig.savefig(out/'E05_feature_effects.png',dpi=180);plt.close(fig)

def narrative(prepared,run,out,summary,effects,directions,nchecks):
    def effect(route,name,fraction):return float(effects.loc[effects.route.eq(route)&effects.contrast.eq(name)&effects.top_fraction.eq(fraction),'mean_difference_pp'].iloc[0])
    lines=['# E05：固定全国 P/U2 的特征消融','',
      '开发对照已完成：2,700 次新增拟合，30 份 B0 折/种子集成评分直接复用 E02；621 家最终 Gold 留出未构造新增特征、未训练、未评分。两条正例路线分别解释，不能以 Gold 来源内评价较高决定真实 U 总体业务模型。','',
      '## 固定内容与方案','',
      '全国开发 Gold 2,484、Silver 30,851，U2 拟合候选 7,516、独立评价 11,274；完全继承 E02 五折、三个种子和每次 U 抽样实体及顺序。所有特征方案共用同一训练/验证企业、样本量、P/U 类总权重及 LR 参数。填充、标准化和编码只在训练部分拟合。',
      '本轮不重跑原 B0。其逐企业均值和抽样标准差按原 float64 评分精确复用、独立核验；B0 的原八字段与 F2_safe 的差异含经营状态和编码顺序，不能解释为纯经营状态效应。','',
      '| 方案 | 特征 |','|---|---|',
      '| F0 | 年龄、注册资本、参保人数、行业大类、企业类型 |',
      '| F1_size | F0＋企业规模 |','| F1_industry | F0＋行业门类/中类；诊断 |',
      '| F1 | F0＋规模＋行业门类/中类；诊断 |','| F2_safe | F1_size＋实缴资本 |',
      '| F2_year | F2_safe＋参保报告年份；诊断 |','| F2 | F1＋实缴资本＋年份；原全组合，诊断 |',
      '| F3 | F2_safe＋注册地址统一解析省市 |','| F4 | F2_safe＋简介有无；采集覆盖诊断 |','',
      'safe 仅表示不包含本次已明确的行业层级/年份风险，不证明无采集偏差或可以部署。F3/F4 固定基于 F2_safe，与训练前方案一致。','',
      '## 全方案 Gold 找回率','',
      '每折将同一批验证 Gold 和 U2 独立评价企业一起排序。指标为相应前段找回的验证 Gold 比例；每种子汇总五折后，取三个种子均值。不是 U 中真正有福利企业的发现率。','',
      '| 方案 | Gold-only 前10% | Gold-only 前20% | 混合 前10% | 混合 前20% |','|---|---:|---:|---:|---:|']
    for variant in ALL_VARIANTS:
        vals=[f'{float(summary.loc[summary.variant.eq(variant)&summary.route.eq(route)&summary.top_fraction.eq(f),"mean_capture"].iloc[0]):.1%}' for route in ROUTES for f in [.1,.2]]
        lines.append('| '+variant+' | '+' | '.join(vals)+' |')
    lines+=['','## 单项配对变化','',
      '下表每个差值都相对于预先规定的同样本参照，单位为百分点；不同增量参照详见括号。不能将多项变化相加，忽略特征相互作用。','',
      '| 新增项与参照 | Gold-only 前10% | Gold-only 前20% | 混合 前10% | 混合 前20% |','|---|---:|---:|---:|---:|']
    names=[('size','规模：F1_size−F0'),('paid_capital','实缴资本：F2_safe−F1_size'),('industry_with_size','行业门类/中类：F1−F1_size（诊断）'),('insured_year','年份：F2_year−F2_safe（诊断）'),('address_geography','地址省市：F3−F2_safe'),('introduction_coverage','简介有无：F4−F2_safe（覆盖诊断）')]
    for name,label in names:
        lines.append('| '+label+' | '+' | '.join(f'{effect(route,name,f):+.2f}' for route in ROUTES for f in [.1,.2])+' |')
    lines+=['','15 个折/种子配对方向与逐种子差值见 feature_effect_directions.csv 和 pooled_feature_differences_by_seed.csv。这些比较有关联，不当作独立试验作显著性推断。多方案开发探索尚未经过最终留出确认。','',
      '## 数据风险与分组','',
      '- 原 P 地区字段优先业务标签，U 地区字段主要地址提取；本轮均只从注册地址按同一规则解析，未使用现成地区标签、公司名、Gold 来源或任何打标证据。无法解析保留缺失，未补全地址；地区覆盖见 field_coverage.csv。',
      '- 行业中类/报告年份的 Gold/U 非缺失率差异较大；行业门类原字段还存在 P/U 层级混用。这些新字段是明确的诊断性暴露，不表示旧 20 个百分点缺失准入限制已撤销。每折两个 P 路线的覆盖与风险见 training_only_feature_audit.csv。',
      '- 简介标志只表示是否采集到简介。即使提高 Gold 找回，也可能反映历史来源/采集过程；没有福利真值证据不能解释为有效福利信号。',
      '- subgroup_feature_differences.csv 保存地区、规模、行业大类及历史 Gold/量化榜单来源分组差值和样本数；小来源分组属于探索，不证明独立来源或跨地区泛化。',
      '- TopK_feature_overlap.csv 及 Top100_feature_difference_details.csv 保存同一 U2 面板上的名单改变；名单仍只是开发排序，不是业务确认企业。','',
      '## 核验与交付','',
      f'2,700 次拟合均正常收敛，4 项针对性测试通过，{nchecks:,} 项独立核验全部通过。独立核验重新计算 1,200 条捕获指标，检查全部预测索引、数据/源码哈希、固定 U 抽样、正例指纹、特征清单、权重和最终留出隔离；B0 均值及标准差与 E02 精确相同。',
      '逐企业评分以同一索引和无损 float64 数组保存；原始文件、旧运行和最终留出保持原样。全部代码、固定方案、字段审计、分组指标和复制哈希归档到本地项目。未发布逐企业数据，未输出全量生产 U 评分。','',
      '## 当前解释与下一步','',
      '特征是否入模要结合来源独立性和采集一致性，不能只按本表最大分数挑选。两条正例路线继续保留；风险字段如出现提升，先保留为诊断结果，而非批准成为主模型特征。',
      'E05 开发消融到此结束；下一项 E06 为稳定画像方案上的 nnPU/LR 对照及类别先验敏感性，需要预先固定比较方案与先验依据。当前未启动 E06，也未使用最终 Gold 留出筛选。','']
    (out/'E05结果报告.md').write_text('\n'.join(lines),encoding='utf-8')
    dump({'status':'complete','source_run':str(run),'source_run_manifest_sha256':sha(run/'manifest.json'),'report_code_sha256':sha(Path(__file__)),
      'all_passed':True,'independent_checks':nchecks,'new_fits':2700,'final_Gold_scored':False,'scope':'source-internal development only; diagnostic features not approved for deployment'},out/'manifest.json')
    dump(output_hashes(out),out/'output_hashes.json')
    print(json.dumps({'status':'complete','independent_checks':nchecks,'report':str(out),'main_effects':effects.loc[effects.contrast.isin([n for n,l in names])&effects.top_fraction.eq(.1)].to_dict('records')},ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepared',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--parent-run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();report(a.prepared.resolve(),a.run.resolve(),a.parent_run.resolve(),a.out.resolve())
