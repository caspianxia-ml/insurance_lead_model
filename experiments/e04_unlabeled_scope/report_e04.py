"""Independent verification and same-panel E04 reporting. No final Gold scoring."""
from pathlib import Path
import argparse,json,math,os,tempfile
import numpy as np
import pandas as pd
from e04 import ARMS,PANELS,PAIRS
from e02_support import sha,code_hash,save,dump,verify_hashes,output_hashes,read_features

def report(prepared,run,e02run,out):
    if out.exists():raise FileExistsError(out)
    verify_hashes(prepared);verify_hashes(run);verify_hashes(e02run)
    protocol=json.loads((run/'manifest.json').read_text())
    assert protocol['status']=='complete' and protocol['stage']=='full_development' and not protocol['final_Gold_holdout_scored']
    assert protocol['code_sha256']==sha(Path(__file__).with_name('e04.py'))
    assert protocol['support_code_sha256']==sha(Path(__file__).with_name('e02_support.py'))
    split=pd.read_csv(prepared/'gold_entity_split.csv',dtype={'credit_code_normalized':str})
    fmap=split.set_index('credit_code_normalized').development_fold
    reserved=set(split.loc[split.role.eq('final_holdout'),'credit_code_normalized'])
    gold=read_features(prepared/'gold_development.csv');silver=read_features(prepared/'silver.csv')
    u2fit=read_features(prepared/'U2_fit_pool.csv');u3fit=read_features(prepared/'U3_fit_pool.csv')
    common=read_features(prepared/'U_common_evaluation.csv');extra=read_features(prepared/'U_increment_evaluation.csv')
    fits=pd.read_csv(run/'fit_log.csv');metrics=pd.read_csv(run/'cv_metrics.csv')
    samples=pd.read_csv(run/'paired_U_samples.csv',dtype={'credit_code_normalized':str})
    checks=[]
    def check(name,passed,detail=''):
        checks.append({'check':name,'passed':bool(passed),'detail':detail})
        if not passed:raise AssertionError(name+': '+detail)
    check('all_original_input_and_run_hashes',True)
    check('600_converged_fits',len(fits)==600 and fits.converged.all() and fits.max_n_iter.lt(1000).all())
    check('final_Gold_ID_count',len(reserved)==621)
    indices={};gold_predictions=[];subgroups=[];baseline=[]
    usum=np.zeros((4,len(common)+len(extra)));models=0
    for path in sorted((run/'prediction_index').glob('*.csv')):
        fold=int(path.stem.split('_')[1]);index=pd.read_csv(path,dtype={'credit_code_normalized':str})
        expected=set(gold.loc[gold.credit_code_normalized.map(fmap).eq(fold),'credit_code_normalized'])
        check(f'fold{fold}_Gold_ids',set(index.loc[index.kind.eq('Gold_validation'),'credit_code_normalized'])==expected)
        check(f'fold{fold}_common_U_ids',set(index.loc[index.kind.eq('U_common'),'credit_code_normalized'])==set(common.credit_code_normalized))
        check(f'fold{fold}_increment_U_ids',set(index.loc[index.kind.eq('U_increment'),'credit_code_normalized'])==set(extra.credit_code_normalized))
        check(f'fold{fold}_unique_finite_index',not index.credit_code_normalized.duplicated().any() and index.prediction_row.tolist()==list(range(len(index))))
        check(f'fold{fold}_final_Gold_excluded',not set(index.credit_code_normalized)&reserved)
        indices[fold]=index
    for path in sorted((run/'predictions').glob('*.npz')):
        with np.load(path,allow_pickle=False) as data:
            fold=int(data['fold']);seed=int(data['seed']);mean=data['mean'];std=data['std']
            index=indices[fold]
            check(path.stem+'_index_hash',str(data['index_sha256'].item())==sha(run/'prediction_index'/f'fold_{fold}.csv'))
            check(path.stem+'_array_format',data['arms'].tolist()==ARMS and mean.dtype==np.float64 and mean.shape==(4,len(index)) and std.shape==mean.shape)
            check(path.stem+'_finite_predictions',np.isfinite(mean).all() and np.isfinite(std).all() and (std>=0).all())
        goldmask=index.kind.eq('Gold_validation').to_numpy();commonmask=index.kind.ne('U_increment').to_numpy()
        u_mask=~goldmask
        check(path.stem+'_fixed_U_order',index.loc[u_mask,'credit_code_normalized'].tolist()==common.credit_code_normalized.tolist()+extra.credit_code_normalized.tolist())
        usum+=mean[:,u_mask];models+=1
        for armid,arm in enumerate(ARMS):
            for panel in PANELS:
                mask=commonmask if panel=='common_U2' else np.ones(len(index),dtype=bool)
                rowids=np.flatnonzero(mask);codes=index.loc[mask,'credit_code_normalized'].to_numpy(dtype=str)
                order=np.lexsort((codes,-mean[armid,mask]));rankids=rowids[order]
                ranks=np.zeros(len(index),dtype=int);ranks[rankids]=np.arange(1,len(rowids)+1)
                gm=metrics[metrics.fold.eq(fold)&metrics.seed.eq(seed)&metrics.arm.eq(arm)&metrics.panel.eq(panel)]
                check(f'{path.stem}_{arm}_{panel}_four_metrics',len(gm)==4)
                for metric in gm.to_dict('records'):
                    k=math.ceil(len(rowids)*metric['top_fraction']);hit=int(goldmask[rankids[:k]].sum())
                    check(f'{path.stem}_{arm}_{panel}_{metric["top_fraction"]}',metric['top_k']==k and metric['evaluation_rows']==len(rowids) and metric['captured_Gold']==hit and metric['validation_Gold']==int(goldmask.sum()))
                g=index.loc[goldmask,['credit_code_normalized','company_name','region_group','industry_major_category_clean','company_size_clean','source']].copy()
                g['source_group']=np.where(g.source.fillna('').str.contains('量化Q3榜单',regex=False),'quantitative_list','historical_Gold')
                g['arm']=arm;g['panel']=panel;g['fold']=fold;g['seed']=seed;g['score']=mean[armid,goldmask];g['rank']=ranks[goldmask]
                gold_predictions.append(g)
                for field in ['region_group','industry_major_category_clean','company_size_clean','source_group']:
                    for category,group in g.groupby(g[field].fillna('<missing>'),dropna=False):
                        for fraction in [.1,.2]:
                            subgroups.append({'arm':arm,'panel':panel,'fold':fold,'seed':seed,'group_field':field,'group':category,
                              'top_fraction':fraction,'Gold_n':len(group),'captured_Gold':int(group['rank'].le(math.ceil(len(rowids)*fraction)).sum())})
            # Independent continuity check: existing U2 control on common panel.
            if arm in ['A_Gold_U2','B_mixed_U2']:
                oldarm='A_Gold_only' if arm=='A_Gold_U2' else 'B_Gold_Silver'
                old=pd.read_csv(e02run/'predictions'/f'fold_{fold}_seed_{seed}_{oldarm}.csv',dtype={'credit_code_normalized':str},float_precision='round_trip').set_index('credit_code_normalized')
                newcodes=index.loc[commonmask,'credit_code_normalized']
                old_values=old.loc[newcodes,'score'].to_numpy()
                maxdiff=float(np.max(np.abs(mean[armid,commonmask]-old_values)))
                fields=fits.loc[fits.arm.eq(arm)&fits.fold.eq(fold)&fits.seed.eq(seed),'feature_fields'].iloc[0]
                old_fields=pd.read_csv(e02run/'fit_log.csv').query('fold==@fold and seed==@seed and arm==@oldarm').feature_fields.iloc[0]
                samefields=fields==old_fields
                if samefields:check(f'{path.stem}_{arm}_E02_continuity',np.allclose(mean[armid,commonmask],old_values,rtol=1e-6,atol=1e-6),str(maxdiff))
                baseline.append({'fold':fold,'seed':seed,'arm':arm,'same_feature_fields_as_E02':samefields,'max_absolute_score_difference':maxdiff})
    check('15_prediction_archives',models==15)
    for (fold,seed,repeat),logs in fits.groupby(['fold','seed','repeat']):
        ng=set(gold.loc[gold.credit_code_normalized.map(fmap).ne(fold),'credit_code_normalized']);mixed=ng|set(silver.credit_code_normalized)
        check(f'four_arm_{fold}_{seed}_{repeat}',len(logs)==4 and logs.target_entity_hash.nunique()==1 and logs.feature_fields.nunique()==1
          and np.allclose(logs.total_weight,2*len(ng)) and np.allclose(logs.positive_weight_sum,logs.U_weight_sum))
        for scope,pool in [('U2',u2fit),('U3',u3fit)]:
            sample=samples[samples.fold.eq(fold)&samples.seed.eq(seed)&samples['repeat'].eq(repeat)&samples.scope.eq(scope)]
            codes=set(sample.credit_code_normalized);slog=logs[logs.scope.eq(scope)]
            check(f'U_{scope}_{fold}_{seed}_{repeat}',len(sample)==len(codes)==len(ng) and codes<=set(pool.credit_code_normalized) and not codes&reserved
              and slog.U_sample_hash.nunique()==1 and slog.U_sample_hash.iloc[0]==code_hash(codes))
        for row in logs.to_dict('records'):
            p=ng if row['arm'].startswith(('A_','C_')) else mixed
            check(f'P_{row["arm"]}_{fold}_{seed}_{repeat}',row['positive_entity_hash']==code_hash(p) and not p&reserved)
    check('300_U_sample_groups',samples.groupby(['fold','seed','repeat','scope']).ngroups==300)
    gp=pd.concat(gold_predictions,ignore_index=True)
    coverage=gp.groupby(['arm','panel','seed']).credit_code_normalized.agg(['count','nunique'])
    check('2484_development_Gold_OOF_per_arm_panel_seed',len(coverage)==24 and coverage['count'].eq(2484).all() and coverage['nunique'].eq(2484).all())
    out.mkdir(parents=True)
    save(pd.DataFrame(checks),out/'independent_verification.csv');save(pd.DataFrame(baseline),out/'E02_control_continuity.csv')
    save(gp,out/'Gold_development_OOF_predictions.csv')
    summarize(prepared,run,out,metrics,pd.DataFrame(subgroups),indices[1].loc[~indices[1].kind.eq('Gold_validation')].copy(),usum/models,len(checks))

def summarize(prepared,run,out,metrics,subgroups,uindex,umean,checks_count):
    """Same-panel summaries, ranking diagnostics and narrative (no model fitting)."""
    byseed=metrics.groupby(['arm','panel','seed','top_fraction'],as_index=False).agg(captured_Gold=('captured_Gold','sum'),Gold_n=('validation_Gold','sum'))
    byseed['capture_rate']=byseed.captured_Gold/byseed.Gold_n;save(byseed,out/'capture_by_seed.csv')
    summary=byseed.groupby(['arm','panel','top_fraction'],as_index=False).agg(mean_capture=('capture_rate','mean'),min_capture=('capture_rate','min'),max_capture=('capture_rate','max'))
    save(summary,out/'capture_summary.csv')
    paired=[];pooled=[]
    for higher,lower,name in PAIRS:
        a=metrics[metrics.arm.eq(higher)].merge(metrics[metrics.arm.eq(lower)],on=['panel','fold','seed','top_fraction'],suffixes=('_U3','_U2'),validate='one_to_one')
        a['contrast']=name;a['difference_pp']=100*(a.capture_rate_U3-a.capture_rate_U2);paired.append(a)
        b=byseed[byseed.arm.eq(higher)].merge(byseed[byseed.arm.eq(lower)],on=['panel','seed','top_fraction'],suffixes=('_U3','_U2'),validate='one_to_one')
        b['contrast']=name;b['difference_pp']=100*(b.capture_rate_U3-b.capture_rate_U2);pooled.append(b)
    paired=pd.concat(paired,ignore_index=True);pooled=pd.concat(pooled,ignore_index=True)
    save(paired,out/'paired_U_scope_differences.csv');save(pooled,out/'pooled_U_scope_differences_by_seed.csv')
    differences=paired.groupby(['panel','contrast','top_fraction'],as_index=False).agg(mean_fold_difference_pp=('difference_pp','mean'),min_difference_pp=('difference_pp','min'),max_difference_pp=('difference_pp','max'),
      U3_better=('difference_pp',lambda x:int((x>0).sum())),ties=('difference_pp',lambda x:int((x==0).sum())),comparisons=('difference_pp','size'))
    save(differences,out/'U_scope_difference_summary.csv')
    effects=pooled.groupby(['panel','contrast','top_fraction'],as_index=False).agg(mean_difference_pp=('difference_pp','mean'),min_seed_difference_pp=('difference_pp','min'),max_seed_difference_pp=('difference_pp','max'))
    save(effects,out/'pooled_U_scope_effects.csv')
    sg=subgroups.groupby(['arm','panel','seed','group_field','group','top_fraction'],as_index=False).agg(Gold_n=('Gold_n','sum'),captured_Gold=('captured_Gold','sum'))
    sg['capture_rate']=sg.captured_Gold/sg.Gold_n;save(sg,out/'subgroup_capture_by_seed.csv')
    sg=sg.groupby(['arm','panel','group_field','group','top_fraction'],as_index=False).agg(Gold_n=('Gold_n','first'),mean_capture=('capture_rate','mean'),min_capture=('capture_rate','min'),max_capture=('capture_rate','max'))
    save(sg,out/'subgroup_capture_summary.csv')
    subgroup_differences=[]
    for higher,lower,name in PAIRS:
        a=sg[sg.arm.eq(higher)].merge(sg[sg.arm.eq(lower)],on=['panel','group_field','group','top_fraction'],suffixes=('_U3','_U2'))
        a['contrast']=name;a['difference_pp']=100*(a.mean_capture_U3-a.mean_capture_U2);subgroup_differences.append(a)
    save(pd.concat(subgroup_differences,ignore_index=True),out/'subgroup_U_scope_differences.csv')
    overlap=[];correlations=[];details=[]
    for panel in PANELS:
        mask=uindex.kind.eq('U_common').to_numpy() if panel=='common_U2' else np.ones(len(uindex),dtype=bool)
        ranks={}
        for armid,arm in enumerate(ARMS):
            frame=uindex.loc[mask,['credit_code_normalized','company_name','kind','region_group']].copy()
            frame['development_ensemble_score']=umean[armid,mask]
            frame=frame.sort_values(['development_ensemble_score','credit_code_normalized'],ascending=[False,True]);frame['rank']=np.arange(1,len(frame)+1)
            ranks[arm]=frame;save(frame,out/f'{panel}_{arm}_U_ranking.csv')
        for higher,lower,name in PAIRS:
            joined=ranks[higher].merge(ranks[lower],on=['credit_code_normalized','company_name','kind','region_group'],suffixes=('_U3','_U2'),validate='one_to_one')
            joined['rank_change_U2_minus_U3']=joined.rank_U2-joined.rank_U3
            save(joined,out/f'{panel}_{name}_U_ranking_comparison.csv')
            correlations.append({'panel':panel,'contrast':name,'entities':len(joined),'Spearman_rank_correlation':float(joined[['rank_U3','rank_U2']].corr(method='spearman').iloc[0,1])})
            for k in sorted(set([100,500,1000,math.ceil(.1*len(joined)),math.ceil(.2*len(joined))])):
                a=set(ranks[higher].head(k).credit_code_normalized);b=set(ranks[lower].head(k).credit_code_normalized)
                overlap.append({'panel':panel,'contrast':name,'top_k':k,'intersection':len(a&b),'retained_fraction':len(a&b)/k,'jaccard':len(a&b)/len(a|b),
                  'new_U3_increment_in_U3trained_top':int(ranks[higher].head(k).kind.eq('U_increment').sum()),'new_U3_increment_in_U2trained_top':int(ranks[lower].head(k).kind.eq('U_increment').sum())})
                if k in [100,1000]:
                    for direction,codes in [('U3_only',a-b),('U2_only',b-a)]:
                        for r in joined.loc[joined.credit_code_normalized.isin(codes)].to_dict('records'):
                            details.append({'panel':panel,'contrast':name,'top_k':k,'direction':direction,**r})
    save(pd.DataFrame(overlap),out/'TopK_U_scope_overlap.csv');save(pd.DataFrame(details),out/'TopK_U_scope_difference_details.csv');save(pd.DataFrame(correlations),out/'U_scope_rank_correlations.csv')
    if 'MPLCONFIGDIR' not in os.environ:os.environ['MPLCONFIGDIR']=tempfile.mkdtemp(prefix='e04-mpl-')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,3.8),layout='constrained')
    labels=['Gold / U2','Mixed / U2','Gold / U3','Mixed / U3'];colors=['#3677b5','#e4943b','#21562b','#b34444'];fractions=[.01,.05,.1,.2];x=np.arange(4);width=.19
    for i,arm in enumerate(ARMS):
        values=summary[summary.arm.eq(arm)&summary.panel.eq('common_U2')].set_index('top_fraction').loc[fractions,'mean_capture']*100
        axes[0].bar(x+(i-1.5)*width,values,width,label=labels[i],color=colors[i])
    axes[0].set_xticks(x,['Top 1%','Top 5%','Top 10%','Top 20%']);axes[0].set_ylabel('Gold capture in common U2 panel (%)');axes[0].legend(frameon=False,fontsize=8)
    categories=[];gold_effect=[];mixed_effect=[]
    for panel,prefix in [('common_U2','Common'),('expanded_U3','Expanded')]:
        for fraction in [.1,.2]:
            categories.append(f'{prefix}\nTop {fraction:.0%}')
            for contrast,target in [('C_minus_A',gold_effect),('D_minus_B',mixed_effect)]:
                target.append(float(effects.loc[effects.panel.eq(panel)&effects.contrast.eq(contrast)&effects.top_fraction.eq(fraction),'mean_difference_pp'].iloc[0]))
    axes[1].bar(x-.17,gold_effect,.34,label='Gold: U3 minus U2',color='#21562b');axes[1].bar(x+.17,mixed_effect,.34,label='Mixed: U3 minus U2',color='#b34444')
    axes[1].axhline(0,color='#555',linewidth=.8);axes[1].set_xticks(x,categories);axes[1].set_ylabel('Within-panel change (percentage points)');axes[1].legend(frameon=False,fontsize=8)
    for ax in axes:ax.spines[['top','right']].set_visible(False)
    fig.savefig(out/'E04_U_scope_comparison.png',dpi=180);plt.close(fig)
    write_narrative(prepared,run,out,summary,effects,differences,checks_count)

def write_narrative(prepared,run,out,summary,effects,differences,checks_count):
    def effect(panel,contrast,fraction):
        return float(effects.loc[effects.panel.eq(panel)&effects.contrast.eq(contrast)&effects.top_fraction.eq(fraction),'mean_difference_pp'].iloc[0])
    lines=['# E04：固定全国 P、比较 U2 与 U3 训练总体','',
      '状态：四组开发主对照完成；全部 621 家最终留出 Gold 未评分。主结论只在固定 U2 面板内比较，扩展面板为补充；不据此选定目标 U 总体的业务模型。','',
      '## 样本与受控扩围','',
      '全国开发 Gold 2,484、Silver 30,851，五折与最终 Gold 留出完全承接 E02。每折约 1,987/1,988 家训练 Gold、496/497 家验证 Gold。',
      'U2 保留 18,790 家，原拟合候选 7,516、评价 11,274；原 U3 身份准入和经营状态过滤后 124,892 家，包含全部 U2，新增 106,102 家。新增部分固定分为拟合 42,440、评价 63,662。因此受控 U3 拟合池 49,956，扩展评价 U 74,936。',
      '重合实体采用 E02 U2 的完整记录，新增实体采用冻结 U3 的完整记录，不跨行拼字段；共有九个旧画像字段的核验见 shared_U2_U3_feature_equality.csv（准备目录）。原文件均未修改，U3 逻辑记录未跳行。','',
      '| 组 | 全国训练正例 | 临时对照 U 来源 |','|---|---|---|',
      '| A | Gold-only | U2 拟合池 |','| B | Gold＋Silver | 同一 U2 抽样 |',
      '| C | 与 A 相同的 Gold-only | U3 拟合池 |','| D | 与 B 相同的 Gold＋Silver | 同一 U3 抽样 |','',
      '主比较 C−A 和 D−B，分别保留两条正例路线。每次四组 U 样本量相同，约等于训练 Gold 数；P/U 类总权重和跨组总权重一致。U2/U3 实体可以不同，这是本实验改变的因素。不是增加每次 U 训练量的实验。','',
      '## 固定 U2 主评价面板','',
      '每折使用同一批验证 Gold＋11,274 家 U2 评价企业；四组排序分母相同。先按每种子累计五折命中数/2,484，再取三个种子的均值。','',
      '| 排名前段 | A Gold/U2 | B 混合/U2 | C Gold/U3 | D 混合/U3 |','|---|---:|---:|---:|---:|']
    for fraction in [.01,.05,.1,.2]:
        vals=[f'{summary.loc[summary.arm.eq(arm)&summary.panel.eq("common_U2")&summary.top_fraction.eq(fraction),"mean_capture"].iloc[0]:.1%}' for arm in ARMS]
        lines.append(f'| Top {fraction:.0%} | '+' | '.join(vals)+' |')
    lines += ['','## 扩展竞争面板（补充）','',
      '每折使用相同验证 Gold＋74,936 家共同扩展评价 U，四组在该面板内可比较。它没有新增经核验的 U 标签，不能称为独立业务验证；不同面板的同一百分比对应不同名额，不能把两张表的原始捕获率差当成性能提升。','',
      '| 排名前段 | A Gold/U2 | B 混合/U2 | C Gold/U3 | D 混合/U3 |','|---|---:|---:|---:|---:|']
    for fraction in [.01,.05,.1,.2]:
        vals=[f'{summary.loc[summary.arm.eq(arm)&summary.panel.eq("expanded_U3")&summary.top_fraction.eq(fraction),"mean_capture"].iloc[0]:.1%}' for arm in ARMS]
        lines.append(f'| Top {fraction:.0%} | '+' | '.join(vals)+' |')
    lines += ['','## 面板内配对差值：U3 训练减 U2 训练','','| 面板与路线 | Top 1% | Top 10% | Top 20% |','|---|---:|---:|---:|']
    for panel,name in [('common_U2','固定 U2 主面板'),('expanded_U3','扩展面板')]:
        for contrast,route in [('C_minus_A','Gold-only'),('D_minus_B','Gold＋Silver')]:
            vals=[f'{effect(panel,contrast,f):+.2f} 个百分点' for f in [.01,.1,.2]]
            lines.append(f'| {name}／{route} | '+' | '.join(vals)+' |')
    lines += ['']
    for contrast,route in [('C_minus_A','Gold-only'),('D_minus_B','混合')]:
        rows=differences[differences.panel.eq('common_U2')&differences.contrast.eq(contrast)]
        a=rows[rows.top_fraction.eq(.1)].iloc[0];b=rows[rows.top_fraction.eq(.2)].iloc[0]
        lines.append(f'固定主面板的 {route} 路线：Top 10% 的 15 个配对折/种子中，U3 较高 {int(a.U3_better)} 次、相同 {int(a.ties)} 次；Top 20% 较高 {int(b.U3_better)} 次、相同 {int(b.ties)} 次。')
    lines += ['','## 运行与独立核验','',
      '5 折 × 3 个预先固定种子（42、123、2026）× 10 次重复 × 4 组，共 600 次 LR 拟合，均正常收敛。字段准入取四组训练部分与相应 U 拟合池的交集；预处理只在训练数据拟合，未加入来源、标签、ID、地域或完整度变量。',
      '逐企业预测使用同一索引＋无损 float64 压缩数组保存。独立复核验证了索引哈希、面板企业、排名捕获计数、四组正例指纹、共用 U、抽样量、权重和最终 Gold 隔离。',
      f'独立检查 {checks_count} 项全部通过。E02_control_continuity.csv 记录共同字段相同时 U2 控制组与 E02 原评分的逐企业数值差异；原准备、拟合及报告产物均有哈希。','',
      '## 名单、分组与解释边界','',
      '- 分组见 subgroup_U_scope_differences.csv，含地区、行业、规模、历史 Gold/量化榜单来源；小样本保留样本量，不作为独立总体结论。',
      '- 两面板内均有相同 U 集合的开发集成排名、排名相关和 Top-K 重合；新增 U 排在前面不代表已确认有保险福利。未生成全部拟合 U 的生产评分。',
      '- Gold 标签可靠性与代表性分开。这里测量的是现有 Gold 来源内排名，不能从较高指标推出真实未标注总体的发现率，也不能据此淘汰 Gold/Silver 路线。',
      '- U3 训练改变 U 抽样构成，可能改变企业规模、行业和覆盖分布。保持预算一致排除了“多抽 U”的混杂，但不单独识别某个企业属性或类别先验的因果作用。',
      '- 15 次折/种子比较有关联，不当作独立重复试验夸大统计显著性；差值小也不自动构成等效性证明。',
      '- 最终 621 家 Gold 留出继续封存。独立来源诊断、业务核验以及真实福利概率校准尚未完成。','',
      '## 当前结论与后续','',
      f'在固定 U2 主面板，Gold-only 的 Top 10%/20% 配对均值差为 {effect("common_U2","C_minus_A",.1):+.2f}/{effect("common_U2","C_minus_A",.2):+.2f} 个百分点；混合路线为 {effect("common_U2","D_minus_B",.1):+.2f}/{effect("common_U2","D_minus_B",.2):+.2f}。以这两个来源内结果说明 U 抽样范围的影响，不直接宣称总体业务优劣。',
      'E04 主开发比较到此结束，没有因本轮结果追加 U 抽样调参或标签调权。下一项按总计划是 E05 特征消融，开始前需固定样本/评价面板；现阶段保留两条正例路线及两个 U 范围的结果，不在最终留出上筛选方案。','',
      '执行入口 e04.py、共享方法 e02_support.py、legacy/src 均保留源码哈希；test_e04.py 四项针对面板隔离、U3 字段准入、无损预测和权重控制的测试已通过。准备、训练与报告均拒绝覆盖旧目录。','']
    (out/'E04结果报告.md').write_text('\n'.join(lines),encoding='utf-8')
    dump({'status':'complete','source_run':str(run.resolve()),'source_run_manifest_sha256':sha(run/'manifest.json'),
      'report_code_sha256':sha(Path(__file__)),'independent_checks':checks_count,'all_passed':True,'final_Gold_scored':False,
      'primary_panel':'common_U2','scope':'Gold source-internal development only; no global business model selection',
      'effects':effects.to_dict('records')},out/'manifest.json')
    dump(output_hashes(out),out/'output_hashes.json')
    print(json.dumps({'status':'complete','independent_checks':checks_count,'report':str(out),
      'primary_effects':effects[effects.panel.eq('common_U2')&effects.top_fraction.isin([.1,.2])].to_dict('records')},ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepared',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--e02-run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();report(a.prepared.resolve(),a.run.resolve(),a.e02_run.resolve(),a.out.resolve())
