"""Independently verify a completed E02 run and write descriptive results.
Does not load or score final Gold holdout features.
"""
from pathlib import Path
import argparse
from collections import Counter
import hashlib,json,math
import numpy as np
import pandas as pd
from e02 import read_features,code_hash,sha,dump,save,verify_hashes,output_hashes

def report(prepared,run,out):
    if out.exists():raise FileExistsError(out)
    verify_hashes(prepared);verify_hashes(run)
    protocol=json.loads((run/'manifest.json').read_text());pm=json.loads((prepared/'manifest.json').read_text())
    assert protocol['status']=='complete' and protocol['stage']=='full_development'
    assert not protocol['holdout_scored'] and not protocol['final_Gold_holdout_scored']
    splits=pd.read_csv(prepared/'gold_entity_split.csv',dtype={'credit_code_normalized':str})
    reserved=set(splits.loc[splits.role.eq('final_holdout'),'credit_code_normalized'])
    gold=read_features(prepared/'gold_development.csv');silver=read_features(prepared/'silver.csv')
    ufit=read_features(prepared/'U_fit_pool.csv');ueval=read_features(prepared/'U_evaluation.csv')
    metrics=pd.read_csv(run/'cv_metrics.csv');fits=pd.read_csv(run/'fit_log.csv');samples=pd.read_csv(run/'paired_U_samples.csv',dtype={'credit_code_normalized':str})
    checks=[]
    def check(name,condition,detail=''):
        checks.append({'check':name,'passed':bool(condition),'detail':detail})
        if not condition:raise AssertionError(name+': '+detail)
    check('run_and_preparation_hashes',True)
    check('code_at_fit_matches_delivered_code',sha(Path(__file__).with_name('e02.py'))==protocol['code_sha256'])
    check('full_fit_count',len(fits)==300)
    check('all_fits_converged',bool(fits.converged.all()) and fits.max_n_iter.lt(1000).all())
    check('no_final_holdout_used',not set(gold.credit_code_normalized)&reserved and not protocol['final_Gold_holdout_scored'])
    predframes=[];subgroups=[]
    for path in sorted((run/'predictions').glob('*.csv')):
        df=pd.read_csv(path,dtype={'credit_code_normalized':str},float_precision='round_trip')
        fold=int(df.fold.iloc[0]);seed=int(df.seed.iloc[0]);arm=df.arm.iloc[0]
        expected_gold=set(splits.loc[splits.development_fold.eq(fold),'credit_code_normalized'])
        actual_gold=set(df.loc[df.kind.eq('Gold_validation'),'credit_code_normalized'])
        check(path.name+'_validation_Gold_ids',actual_gold==expected_gold)
        check(path.name+'_U_evaluation_ids',set(df.loc[df.kind.eq('U_evaluation'),'credit_code_normalized'])==set(ueval.credit_code_normalized))
        check(path.name+'_no_final_holdout',not bool(set(df.credit_code_normalized)&reserved))
        check(path.name+'_finite_unique_predictions',np.isfinite(df.score).all() and not df.credit_code_normalized.duplicated().any())
        check(path.name+'_rank_order',df['rank'].tolist()==list(range(1,len(df)+1)) and df.score.is_monotonic_decreasing)
        for metric in metrics.loc[metrics.fold.eq(fold)&metrics.seed.eq(seed)&metrics.arm.eq(arm)].to_dict('records'):
            k=math.ceil(len(df)*metric['top_fraction'])
            captured=int((df.kind.eq('Gold_validation')&df['rank'].le(k)).sum())
            check(f'{path.name}_capture_{metric["top_fraction"]}',k==metric['top_k'] and captured==metric['captured_Gold'] and len(expected_gold)==metric['validation_Gold'])
        goldpred=df[df.kind.eq('Gold_validation')].copy()
        goldpred['source_group']=np.where(goldpred.source.fillna('').str.contains('量化Q3榜单',regex=False),'quantitative_list','historical_Gold')
        for field in ['region_group','industry_major_category_clean','company_size_clean','source_group']:
            for category,g in goldpred.groupby(goldpred[field].fillna('<missing>'),dropna=False):
                for fraction in [.1,.2]:
                    k=math.ceil(len(df)*fraction)
                    subgroups.append({'arm':arm,'seed':seed,'fold':fold,'group_field':field,'group':category,
                      'top_fraction':fraction,'Gold_n':len(g),'captured_Gold':int(g['rank'].le(k).sum())})
        predframes.append(df)
    check('prediction_file_count',len(predframes)==30)
    for (fold,seed,repeat),sample in samples.groupby(['fold','seed','repeat']):
        matching=fits[fits.fold.eq(fold)&fits.seed.eq(seed)&fits['repeat'].eq(repeat)]
        expected_ptrain=set(splits.loc[splits.role.eq('development')&~splits.development_fold.eq(fold),'credit_code_normalized'])
        sample_codes=set(sample.credit_code_normalized)
        check(f'sample_{fold}_{seed}_{repeat}',len(sample)==len(sample_codes)==len(expected_ptrain) and sample_codes<=set(ufit.credit_code_normalized))
        check(f'pair_{fold}_{seed}_{repeat}',len(matching)==2 and matching.U_sample_hash.nunique()==1 and matching.U_sample_hash.iloc[0]==code_hash(sample_codes)
          and matching.feature_fields.nunique()==1 and matching.target_entity_hash.nunique()==1
          and np.allclose(matching.total_weight,2*len(expected_ptrain)) and np.allclose(matching.positive_weight_sum,matching.U_weight_sum))
        for fit in matching.to_dict('records'):
            positives=expected_ptrain if fit['arm']=='A_Gold_only' else expected_ptrain|set(silver.credit_code_normalized)
            check(f'P_ids_{fold}_{seed}_{repeat}_{fit["arm"]}',fit['positive_entity_hash']==code_hash(positives))
    check('paired_sample_group_count',samples.groupby(['fold','seed','repeat']).ngroups==150)
    allpred=pd.concat(predframes,ignore_index=True)
    coverage=allpred[allpred.kind.eq('Gold_validation')].groupby(['arm','seed']).credit_code_normalized.agg(['count','nunique'])
    check('Gold_development_OOF_coverage',coverage['count'].eq(2484).all() and coverage['nunique'].eq(2484).all())
    out.mkdir(parents=True)
    save(pd.DataFrame(checks),out/'independent_verification.csv')
    byseed=metrics.groupby(['arm','seed','top_fraction'],as_index=False).agg(captured_Gold=('captured_Gold','sum'),Gold_n=('validation_Gold','sum'))
    byseed['capture_rate']=byseed.captured_Gold/byseed.Gold_n
    save(byseed,out/'capture_by_seed.csv')
    summary=byseed.groupby(['arm','top_fraction'],as_index=False).agg(mean_capture=('capture_rate','mean'),min_capture=('capture_rate','min'),max_capture=('capture_rate','max'))
    save(summary,out/'capture_summary.csv')
    paired=metrics.pivot(index=['fold','seed','top_fraction'],columns='arm',values='capture_rate').reset_index()
    paired['B_minus_A_pp']=100*(paired.B_Gold_Silver-paired.A_Gold_only)
    save(paired,out/'paired_fold_differences.csv')
    paired_summary=paired.groupby('top_fraction',as_index=False).agg(mean_difference_pp=('B_minus_A_pp','mean'),min_difference_pp=('B_minus_A_pp','min'),max_difference_pp=('B_minus_A_pp','max'),B_better=('B_minus_A_pp',lambda x:int((x>0).sum())),ties=('B_minus_A_pp',lambda x:int((x==0).sum())),comparisons=('B_minus_A_pp','size'))
    save(paired_summary,out/'paired_difference_summary.csv')
    subgroup=pd.DataFrame(subgroups).groupby(['arm','seed','group_field','group','top_fraction'],as_index=False).agg(Gold_n=('Gold_n','sum'),captured_Gold=('captured_Gold','sum'))
    subgroup['capture_rate']=subgroup.captured_Gold/subgroup.Gold_n
    save(subgroup,out/'subgroup_capture_by_seed.csv')
    sgsummary=subgroup.groupby(['arm','group_field','group','top_fraction'],as_index=False).agg(Gold_n=('Gold_n','first'),mean_capture=('capture_rate','mean'),min_capture=('capture_rate','min'),max_capture=('capture_rate','max'))
    save(sgsummary,out/'subgroup_capture_summary.csv')
    u=allpred[allpred.kind.eq('U_evaluation')]
    averaged=u.groupby(['arm','credit_code_normalized','company_name'],as_index=False).agg(development_ensemble_score=('score','mean'),between_fit_score_std=('score','std'),model_count=('score','count'))
    tables={}
    for arm,g in averaged.groupby('arm'):
        g=g.sort_values(['development_ensemble_score','credit_code_normalized'],ascending=[False,True]).copy()
        g['rank']=np.arange(1,len(g)+1);tables[arm]=g
        save(g,out/f'{arm}_U_evaluation_ranking.csv')
    comparison=tables['A_Gold_only'].merge(tables['B_Gold_Silver'],on=['credit_code_normalized','company_name'],suffixes=('_A','_B'),validate='one_to_one')
    comparison['rank_change_A_minus_B']=comparison.rank_A-comparison.rank_B
    save(comparison,out/'U_evaluation_ranking_comparison.csv')
    overlap=[];differences=[]
    for k in [100,500,1000,math.ceil(.1*len(ueval)),math.ceil(.2*len(ueval))]:
        a=set(tables['A_Gold_only'].head(k).credit_code_normalized);b=set(tables['B_Gold_Silver'].head(k).credit_code_normalized)
        overlap.append({'top_k':k,'intersection':len(a&b),'jaccard':len(a&b)/len(a|b),'retained_fraction':len(a&b)/k})
        for direction,codes in [('A_only',a-b),('B_only',b-a)]:
            for row in comparison.loc[comparison.credit_code_normalized.isin(codes)].to_dict('records'):
                differences.append({'top_k':k,'direction':direction,**row})
    save(pd.DataFrame(overlap),out/'TopK_overlap.csv');save(pd.DataFrame(differences),out/'TopK_difference_details.csv')
    gold_oof=allpred[allpred.kind.eq('Gold_validation')].groupby(['arm','credit_code_normalized','company_name','fold'],as_index=False).agg(score_mean_across_seeds=('score','mean'),seeds=('seed','nunique'))
    save(gold_oof,out/'Gold_development_OOF_scores.csv')
    specrows=[]
    for path in sorted((run/'feature_audits').glob('*.csv')):
        audit=pd.read_csv(path);audit.insert(0,'fold',int(path.stem.split('_')[1]));specrows.append(audit)
    feature_audit=pd.concat(specrows,ignore_index=True);save(feature_audit,out/'feature_eligibility_all_folds.csv')
    # Export a static figure; no interactive tool or new modelling is involved.
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
        fractions=[.01,.05,.1,.2];x=np.arange(4);width=.35
        for offset,arm,label,color in [(-.5,'A_Gold_only','Gold only','#3778b8'),(.5,'B_Gold_Silver','Gold + Silver','#e18a35')]:
            values=summary[summary.arm.eq(arm)].set_index('top_fraction').loc[fractions,'mean_capture'].to_numpy()*100
            axes[0].bar(x+offset*width,values,width,label=label,color=color)
        axes[0].set_xticks(x,['Top 1%','Top 5%','Top 10%','Top 20%']);axes[0].set_ylabel('Development Gold capture (%)');axes[0].legend(frameon=False)
        for fraction,color in [(.1,'#3778b8'),(.2,'#e18a35')]:
            g=paired[paired.top_fraction.eq(fraction)].sort_values(['fold','seed'])
            axes[1].plot(np.arange(1,len(g)+1),g.B_minus_A_pp,'o-',label=f'Top {fraction:.0%}',color=color,linewidth=1)
        axes[1].axhline(0,color='#555',linewidth=.8);axes[1].set_xlabel('Paired fold / seed comparisons');axes[1].set_ylabel('Gold+Silver minus Gold-only (pp)');axes[1].legend(frameon=False)
        for ax in axes:ax.spines[['top','right']].set_visible(False)
        fig.savefig(out/'E02_development_comparison.png',dpi=180);plt.close(fig)
    except ImportError:pass
    top10=paired_summary[paired_summary.top_fraction.eq(.1)].iloc[0]
    top20=paired_summary[paired_summary.top_fraction.eq(.2)].iloc[0]
    conclusion=('混合组在当前 Gold 来源内开发排序较高；目标 U 总体效用尚未确定' if top10.B_better==15 and top20.B_better==15 else
      'Gold-only 在当前 Gold 来源内开发排序较高；目标 U 总体效用尚未确定' if top10.B_better==0 and top20.B_better==0 and top10.mean_difference_pp<0 and top20.mean_difference_pp<0 else
      '两组在当前 Gold 来源内存在折间或指标差异；目标 U 总体效用尚未确定')
    lines=['# E02：Gold-only 与 Gold＋Silver 开发实验结果','',
      '状态：五折开发比较已完成；没有评分或使用 621 家最终留出 Gold。E01 独立复现未执行。','',
      f'结论：{conclusion}。这是对已观察 Gold 的开发集排序比较，尚非最终验证或真实新线索发现率。','',
      '## 样本和运行','',
      '复用 v2_raw_001 与 eda_002 的实体准入，另共同沿用旧经营状态过滤。Gold 3,105 家中固定留出 621，开发部分 2,484；Silver 30,851；U2 18,790，其中 7,516 为拟合候选池、11,274 为独立评价池。',
      'A 每折使用约 1,987/1,988 家训练 Gold；B 使用相同 Gold 加全部准入 Silver。验证 Gold 每折 496/497 家，两组评价池和排名分母相同。',
      '旧画像特征、LR 参数固定；5 折 × 3 个预先固定种子（42、123、2026）× 10 次配对 U 抽样 × 2 组，共 300 次 LR 拟合。单独的小范围运行只检查管道，未更改主实验参数。','',
      '## 主要开发结果','',
      '下表先在每个种子内累计五折命中的 Gold/2,484，再对三个种子取均值；括号为种子最小–最大。','',
      '| 排名前段 | A：Gold-only | B：Gold＋Silver |','|---|---:|---:|']
    for fraction in fractions:
        values=[]
        for arm in ['A_Gold_only','B_Gold_Silver']:
            row=summary[summary.arm.eq(arm)&summary.top_fraction.eq(fraction)].iloc[0]
            values.append(f'{row.mean_capture:.1%}（{row.min_capture:.1%}–{row.max_capture:.1%}）')
        lines.append(f'| Top {fraction:.0%} | '+' | '.join(values)+' |')
    lines += ['',f'Top 10% 的 15 个配对折/种子中，B 优于 A {int(top10.B_better)} 次；Top 20% 为 {int(top20.B_better)} 次。配对差值（百分点）见 paired_fold_differences.csv。重复种子/交叉验证折存在依赖，不把这 15 个比较当独立样本做夸大的显著性结论。','',
      '## 控制变量和验证','',
      '- 每折每次重复，两组使用完全相同的 U 企业、数量和评价对象。',
      '- P 与 U 的总权重在两组中各自相等，跨组总权重相同；相同 C 下避免样本量变化额外改变正则尺度。B 内 Gold/Silver 先等权。',
      '- 共同特征清单只按训练 Gold、混合 P 与 U 拟合候选池的可用性确定，取两组交集；预处理分别在各自训练数据拟合。',
      '- 行业中类未通过共同特征准入（缺失率差超过旧规则 20 个百分点）；其余八个旧画像字段在五折共同使用。没有加入来源、标签、地域、名称、代码或简介覆盖特征。',
      '- 所有 300 次拟合均正常收敛。独立核验每份预测的企业集合、捕获数量、训练 P 指纹、U 配对、特征一致、权重以及最终 Gold 隔离。',
      f'- 独立验证检查 {len(checks)} 项全部通过，见 independent_verification.csv；输入、代码与结果哈希均有记录。','',
      '## 分组和名单','',
      'subgroup_capture_summary.csv 分别记录地区、行业、规模与历史 Gold/量化榜单来源组的捕获率。量化榜单开发样本仅 35 家，其组内差异需谨慎解读，不能与大组同等看待。Gold 是业务确认正例，但其来源选择仍可能限制对全部真实正例的代表性。',
      'TopK_overlap.csv、TopK_difference_details.csv 比较相同 11,274 家独立评价 U 的开发模型集成排名。这里不是全量 U2 的生产候选名单；U 拟合候选池未评分。Gold 最终留出也未评分。','',
      '## 如何使用结果','',
      '本实验回答在固定开发评价规则下加入当前 Silver 的净效果，不单独识别 Silver 的标签噪声率，也不能直接解释成真实新线索准确率。最终 Gold 留出继续封存，供 V2 后续候选方案统一评价。',
      'Gold 可靠标签不等于其来源代表全部真实正例。随机实体留出避免企业泄漏，但不消除来源选择偏差；本轮评价不能分离来源、训练组成、样本量、组权重与标签质量。混合组中 Gold 约占正例总权重 6%，两组候选均保留，不据当前 Gold 来源内结果直接选定业务模型。后续需来源隔离诊断或更独立的业务核验；未经核验的 Silver 留出不能直接当真值。最终 Gold 留出保持封存，本轮不追加训练。','',
      '## 复现与记录','',
      '上级目录 e02.py 为执行代码，legacy/src 为旧方法代码的哈希归档；test_e02.py 为四项针对配对权重、共同字段、排序与泄漏字段隔离的测试。准备、训练和报告均拒绝覆盖已有输出目录。',
      f'准备版本：{prepared.name}；正式运行：{run.name}。manifest.json 记录依赖、参数、种子和规则；output_hashes.json 记录交付哈希。','']
    (out/'E02结果报告.md').write_text('\n'.join(lines),encoding='utf-8')
    dump({'status':'complete','source_run':str(run.resolve()),'source_run_manifest_sha256':sha(run/'manifest.json'),'source_run_output_hashes_sha256':sha(run/'output_hashes.json'),
      'report_code_sha256':sha(Path(__file__)),'independent_checks':len(checks),'all_passed':True,'final_holdout_scored':False,
      'scope':'development_only; independent U evaluation pool; no final selection based on reserved Gold','conclusion':conclusion},out/'manifest.json')
    dump(output_hashes(out),out/'output_hashes.json')
    print(json.dumps({'report':str(out),'independent_checks':len(checks),'conclusion':conclusion,'capture_summary':summary.to_dict('records')},ensure_ascii=False,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepared',type=Path,required=True);ap.add_argument('--run',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();report(args.prepared.resolve(),args.run.resolve(),args.out.resolve())
