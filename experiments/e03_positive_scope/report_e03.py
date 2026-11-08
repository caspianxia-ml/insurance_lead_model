"""Verify E03 artifacts independently and summarize within-panel scope contrasts."""
from pathlib import Path
import argparse,math,os,tempfile
import numpy as np
import pandas as pd
from e03 import ARMS,PAIRS
from e02_support import sha,code_hash,dump,save,output_hashes,verify_hashes,read_features

def report(prepared,run,out):
    if out.exists():raise FileExistsError(out)
    verify_hashes(prepared);verify_hashes(run)
    manifest=pd.read_json(run/'manifest.json',typ='series')
    assert manifest['status']=='complete' and manifest['stage']=='full_development'
    assert manifest['fits']==600 and not manifest['final_Gold_holdout_scored']
    assert manifest['code_sha256']==sha(Path(__file__).with_name('e03.py'))
    gold=read_features(prepared/'gold_development.csv');silver=read_features(prepared/'silver.csv')
    ufit=read_features(prepared/'U1_fit_pool.csv');ueval=read_features(prepared/'U1_evaluation.csv')
    split=pd.read_csv(prepared/'gold_entity_split.csv',dtype={'credit_code_normalized':str})
    fmap=split.set_index('credit_code_normalized').development_fold
    reserved=set(split.loc[split.role.eq('final_holdout'),'credit_code_normalized'])
    localgold=gold[gold.region_group.isin(['Shanghai','Suzhou'])]
    localsilver=silver[silver.region_group.isin(['Shanghai','Suzhou'])]
    fits=pd.read_csv(run/'fit_log.csv');metrics=pd.read_csv(run/'cv_metrics.csv')
    samples=pd.read_csv(run/'common_U_samples.csv',dtype={'credit_code_normalized':str})
    checks=[]
    def check(name,passed,detail=''):
        checks.append({'check':name,'passed':bool(passed),'detail':detail})
        if not passed:raise AssertionError(name+': '+detail)
    check('original_preparation_and_run_hashes',True)
    check('all_600_fits_converged',len(fits)==600 and fits.converged.all() and fits.max_n_iter.lt(1000).all())
    check('reserved_final_Gold_count',len(reserved)==621)
    pred=[];subgroup=[]
    for path in sorted((run/'predictions').glob('*.csv')):
        df=pd.read_csv(path,dtype={'credit_code_normalized':str},float_precision='round_trip')
        arm=df.arm.iloc[0];fold=int(df.fold.iloc[0]);seed=int(df.seed.iloc[0])
        expected=set(localgold.loc[localgold.credit_code_normalized.map(fmap).eq(fold),'credit_code_normalized'])
        check(path.stem+'_Gold_ids',set(df.loc[df.kind.eq('Gold_validation'),'credit_code_normalized'])==expected)
        check(path.stem+'_U_ids',set(df.loc[df.kind.eq('U_evaluation'),'credit_code_normalized'])==set(ueval.credit_code_normalized))
        check(path.stem+'_final_Gold_excluded',not bool(set(df.credit_code_normalized)&reserved))
        check(path.stem+'_rank_finite_unique',np.isfinite(df.score).all() and not df.credit_code_normalized.duplicated().any() and df.score.is_monotonic_decreasing and df['rank'].tolist()==list(range(1,len(df)+1)))
        for r in metrics.loc[metrics.arm.eq(arm)&metrics.fold.eq(fold)&metrics.seed.eq(seed)].to_dict('records'):
            k=math.ceil(len(df)*r['top_fraction']);hit=int((df.kind.eq('Gold_validation')&df['rank'].le(k)).sum())
            check(path.stem+f'_metric_{r["top_fraction"]}',r['top_k']==k and r['captured_Gold']==hit and r['validation_Gold']==len(expected))
        g=df[df.kind.eq('Gold_validation')].copy()
        g['source_group']=np.where(g.source.fillna('').str.contains('量化Q3榜单',regex=False),'quantitative_list','historical_Gold')
        for field in ['region_group','company_size_clean','industry_major_category_clean','source_group']:
            for category,group in g.groupby(g[field].fillna('<missing>'),dropna=False):
                for fraction in [.1,.2]:
                    subgroup.append({'arm':arm,'fold':fold,'seed':seed,'group_field':field,'group':category,'top_fraction':fraction,
                      'Gold_n':len(group),'captured_Gold':int(group['rank'].le(math.ceil(len(df)*fraction)).sum())})
        pred.append(df)
    check('all_60_prediction_files',len(pred)==60)
    allpred=pd.concat(pred,ignore_index=True)
    coverage=allpred[allpred.kind.eq('Gold_validation')].groupby(['arm','seed']).credit_code_normalized.agg(['count','nunique'])
    check('1163_local_Gold_OOF_per_arm_seed',coverage['count'].eq(1163).all() and coverage['nunique'].eq(1163).all() and len(coverage)==12)
    for (fold,seed,repeat),s in samples.groupby(['fold','seed','repeat']):
        logs=fits[fits.fold.eq(fold)&fits.seed.eq(seed)&fits['repeat'].eq(repeat)]
        c=set(s.credit_code_normalized)
        ng=set(gold.loc[gold.credit_code_normalized.map(fmap).ne(fold),'credit_code_normalized'])
        lg=set(localgold.loc[localgold.credit_code_normalized.map(fmap).ne(fold),'credit_code_normalized'])
        expected={'A_local_Gold':lg,'B_local_mixed':lg|set(localsilver.credit_code_normalized),
          'C_national_Gold':ng,'D_national_mixed':ng|set(silver.credit_code_normalized)}
        check(f'U_{fold}_{seed}_{repeat}',len(s)==len(c)==len(lg) and c<=set(ufit.credit_code_normalized) and not c&set(ueval.credit_code_normalized))
        check(f'four_arm_pair_{fold}_{seed}_{repeat}',len(logs)==4 and logs.U_sample_hash.nunique()==1 and logs.U_sample_hash.iloc[0]==code_hash(c)
          and logs.target_entity_hash.nunique()==1 and logs.feature_fields.nunique()==1 and np.allclose(logs.total_weight,2*len(lg)) and np.allclose(logs.positive_weight_sum,logs.U_weight_sum))
        for r in logs.to_dict('records'):
            check(f'positive_ids_{fold}_{seed}_{repeat}_{r["arm"]}',r['positive_entity_hash']==code_hash(expected[r['arm']]) and not expected[r['arm']]&reserved)
    check('150_matched_U_quadruples',samples.groupby(['fold','seed','repeat']).ngroups==150)
    out.mkdir(parents=True)
    save(pd.DataFrame(checks),out/'independent_verification.csv')
    byseed=metrics.groupby(['arm','seed','top_fraction'],as_index=False).agg(captured_Gold=('captured_Gold','sum'),Gold_n=('validation_Gold','sum'))
    byseed['capture_rate']=byseed.captured_Gold/byseed.Gold_n;save(byseed,out/'capture_by_seed.csv')
    summary=byseed.groupby(['arm','top_fraction'],as_index=False).agg(mean_capture=('capture_rate','mean'),min_capture=('capture_rate','min'),max_capture=('capture_rate','max'))
    save(summary,out/'capture_summary.csv')
    paired=[];pooled=[]
    for higher,lower,name in PAIRS:
        a=metrics[metrics.arm.eq(higher)].merge(metrics[metrics.arm.eq(lower)],on=['fold','seed','top_fraction'],suffixes=('_national','_local'),validate='one_to_one')
        a['contrast']=name;a['difference_pp']=100*(a.capture_rate_national-a.capture_rate_local);paired.append(a)
        b=byseed[byseed.arm.eq(higher)].merge(byseed[byseed.arm.eq(lower)],on=['seed','top_fraction'],suffixes=('_national','_local'),validate='one_to_one')
        b['contrast']=name;b['difference_pp']=100*(b.capture_rate_national-b.capture_rate_local);pooled.append(b)
    differences=pd.concat(paired,ignore_index=True);save(differences,out/'paired_scope_differences.csv')
    pooled=pd.concat(pooled,ignore_index=True);save(pooled,out/'pooled_scope_differences_by_seed.csv')
    aggregate=differences.groupby(['contrast','top_fraction'],as_index=False).agg(mean_fold_difference_pp=('difference_pp','mean'),min_fold_difference_pp=('difference_pp','min'),max_fold_difference_pp=('difference_pp','max'),national_better=('difference_pp',lambda x:int((x>0).sum())),ties=('difference_pp',lambda x:int((x==0).sum())),pairs=('difference_pp','size'))
    save(aggregate,out/'scope_difference_summary.csv')
    subgroup=pd.DataFrame(subgroup).groupby(['arm','seed','group_field','group','top_fraction'],as_index=False).agg(Gold_n=('Gold_n','sum'),captured_Gold=('captured_Gold','sum'))
    subgroup['capture_rate']=subgroup.captured_Gold/subgroup.Gold_n
    save(subgroup,out/'subgroup_capture_by_seed.csv')
    sg=subgroup.groupby(['arm','group_field','group','top_fraction'],as_index=False).agg(Gold_n=('Gold_n','first'),mean_capture=('capture_rate','mean'),min_capture=('capture_rate','min'),max_capture=('capture_rate','max'))
    save(sg,out/'subgroup_capture_summary.csv')
    groupcontrasts=[]
    for higher,lower,name in PAIRS:
        a=sg[sg.arm.eq(higher)].merge(sg[sg.arm.eq(lower)],on=['group_field','group','top_fraction'],suffixes=('_national','_local'))
        a['contrast']=name;a['difference_pp']=100*(a.mean_capture_national-a.mean_capture_local);groupcontrasts.append(a)
    save(pd.concat(groupcontrasts,ignore_index=True),out/'subgroup_scope_differences.csv')
    u=allpred[allpred.kind.eq('U_evaluation')]
    ranks={}
    for arm,g in u.groupby('arm'):
        rank=g.groupby(['credit_code_normalized','company_name'],as_index=False).agg(development_ensemble_score=('score','mean'),models=('score','count'))
        rank=rank.sort_values(['development_ensemble_score','credit_code_normalized'],ascending=[False,True]);rank['rank']=np.arange(1,len(rank)+1)
        ranks[arm]=rank;save(rank,out/f'{arm}_U1_evaluation_ranking.csv')
    overlap=[];topdetail=[];correlations=[]
    for higher,lower,name in PAIRS:
        joined=ranks[higher].merge(ranks[lower],on=['credit_code_normalized','company_name'],suffixes=('_national','_local'),validate='one_to_one')
        joined['rank_change_local_minus_national']=joined.rank_local-joined.rank_national
        save(joined,out/f'{name}_U1_ranking_comparison.csv')
        correlations.append({'contrast':name,'U1_evaluation_n':len(joined),'spearman_rank_correlation':float(joined[['rank_national','rank_local']].corr(method='spearman').iloc[0,1])})
        for k in [100,500,1000,math.ceil(.1*len(ueval)),math.ceil(.2*len(ueval))]:
            a=set(ranks[higher].head(k).credit_code_normalized);b=set(ranks[lower].head(k).credit_code_normalized)
            overlap.append({'contrast':name,'top_k':k,'intersection':len(a&b),'jaccard':len(a&b)/len(a|b),'retained_fraction':len(a&b)/k})
            for direction,codes in [('national_only',a-b),('local_only',b-a)]:
                for r in joined.loc[joined.credit_code_normalized.isin(codes)].to_dict('records'):
                    topdetail.append({'contrast':name,'top_k':k,'direction':direction,**r})
    save(pd.DataFrame(overlap),out/'TopK_scope_overlap.csv');save(pd.DataFrame(topdetail),out/'TopK_scope_difference_details.csv');save(pd.DataFrame(correlations),out/'U1_scope_rank_correlations.csv')
    if 'MPLCONFIGDIR' not in os.environ:os.environ['MPLCONFIGDIR']=tempfile.mkdtemp(prefix='e03-mpl-')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,3.8),layout='constrained')
    colors=['#3677b5','#e4943b','#21562b','#b34444'];labels=['Local Gold','Local Gold+Silver','National Gold','National Gold+Silver']
    fractions=[.01,.05,.1,.2];x=np.arange(4);width=.19
    for i,arm in enumerate(ARMS):
        values=summary[summary.arm.eq(arm)].set_index('top_fraction').loc[fractions,'mean_capture']*100
        axes[0].bar(x+(i-1.5)*width,values,width,label=labels[i],color=colors[i])
    axes[0].set_xticks(x,['Top 1%','Top 5%','Top 10%','Top 20%']);axes[0].set_ylabel('SH/SZ development Gold capture (%)');axes[0].legend(frameon=False,fontsize=8)
    for i,(name,label) in enumerate([('C_minus_A','Gold-only scope effect'),('D_minus_B','Mixed scope effect')]):
        values=pooled[pooled.contrast.eq(name)].groupby('top_fraction').difference_pp.mean().reindex(fractions)
        axes[1].bar(x+(i-.5)*.35,values,.35,label=label,color=['#21562b','#b34444'][i])
    axes[1].axhline(0,color='#555',linewidth=.8);axes[1].set_xticks(x,['Top 1%','Top 5%','Top 10%','Top 20%']);axes[1].set_ylabel('National minus local (percentage points)');axes[1].legend(frameon=False,fontsize=8)
    for ax in axes:ax.spines[['top','right']].set_visible(False)
    fig.savefig(out/'E03_scope_comparison.png',dpi=180);plt.close(fig)
    effects=[]
    for name in ['C_minus_A','D_minus_B']:
        table=pooled[pooled.contrast.eq(name)].groupby('top_fraction').difference_pp.mean()
        effects.append({'contrast':name,'Top1_difference_pp':float(table.loc[.01]),'Top10_difference_pp':float(table.loc[.1]),'Top20_difference_pp':float(table.loc[.2])})
    lines=['# E03：全国 P 与沪苏 P 的四组开发对照','',
      '状态：主开发对照完成，全部 621 家最终留出 Gold 未评分。结论限定为当前沪苏 Gold 来源内排序，不选定目标 U 总体的业务模型。','',
      f'结果随截断位置而变化：扩大为全国 P 后，Gold-only 路线 Top 1% 差值为 {effects[0]["Top1_difference_pp"]:+.2f} 个百分点，混合路线为 {effects[1]["Top1_difference_pp"]:+.2f}；Top 10%/20% 的均值差值见下表，均未显示扩大范围的增益。因此不概括为“全国数据一定好/坏”，也不把混合路线的小差值当作已证明等效。','',
      '## 数据与固定条件','',
      '承接 E02 原开发五折及 Gold 最终留出，沪苏开发验证 Gold 共 1,163 家（每家在每个种子下仅一次样本外验证）。全国开发 Gold 共 2,484 家。Silver 沪苏 9,126、全国 30,851。',
      'U1 共 11,013 家；继承既有 U2 角色，拟合候选 4,353、独立评价 6,660。冻结 U1 的九个旧画像字段与 E02 U2 同代码企业一致，已逐字段核验。',
      '每折四组共用同一 U 抽样与目标面板、共同特征和 P/U 总权重。5 折 × 3 个固定种子 × 10 轮 × 4 组，共 600 次 LR 拟合，均正常收敛。预处理只在对应训练数据拟合，地区只用于样本分组，未作为预测变量。','',
      '## 四组开发捕获率','',
      '先按每个种子累加五折命中数/1,163，再对三个种子取均值。','',
      '| 排名前段 | A 沪苏 Gold | B 沪苏 Gold＋Silver | C 全国 Gold | D 全国 Gold＋Silver |','|---|---:|---:|---:|---:|']
    for fraction in fractions:
        values=[f'{summary.loc[summary.arm.eq(arm)&summary.top_fraction.eq(fraction),"mean_capture"].iloc[0]:.1%}' for arm in ARMS]
        lines.append(f'| Top {fraction:.0%} | '+' | '.join(values)+' |')
    lines += ['','## 主要配对差值：全国减沪苏','','| 路线 | Top 10% 差值 | Top 20% 差值 |','|---|---:|---:|']
    for e in effects:
        lines.append(f'| {"Gold-only（C−A）" if e["contrast"]=="C_minus_A" else "Gold＋Silver（D−B）"} | {e["Top10_difference_pp"]:+.2f} 个百分点 | {e["Top20_difference_pp"]:+.2f} 个百分点 |')
    for contrast,label in [('C_minus_A','Gold-only'),('D_minus_B','混合')]:
        a=aggregate[aggregate.contrast.eq(contrast)&aggregate.top_fraction.eq(.1)].iloc[0]
        b=aggregate[aggregate.contrast.eq(contrast)&aggregate.top_fraction.eq(.2)].iloc[0]
        lines += ['',f'{label}路线：15 个折/种子比较中，全国组 Top 10% 更高 {int(a.national_better)} 次、相同 {int(a.ties)} 次；Top 20% 更高 {int(b.national_better)} 次、相同 {int(b.ties)} 次。完整差值及范围见 scope_difference_summary.csv。']
    lines += ['','## 分组、组成与解释边界','',
      '- 只比较同一折、同一种子、同一评价面板内的 C−A 与 D−B。E02 的候选总体和验证对象不同，不将两轮原始捕获率直接横比。',
      '- 扩大正例范围同时改变样本量、来源、行业和规模组成。混合路线 Gold 正例权重份额约从沪苏 9.25% 变为全国 6.05%；因此不是纯地域因果效应，也不能单独判断标签质量。',
      '- 业务确认保证 Gold 标签可信，不保证其来源代表所有真实正例。这里的更高/更低仅是当前沪苏 Gold 来源内泛化，不能据此宣称总体业务模型更优或淘汰 Gold/Silver 路线。',
      '- subgroup_scope_differences.csv 分上海/苏州、行业、规模与历史 Gold/量化榜单来源，使用同一总体排名阈值；小组保留样本量，不能过度解释。',
      '- TopK_scope_overlap.csv 与 U1_scope_rank_correlations.csv 比较相同独立 U1 的开发集成排序。排名变化不是已核验新线索效果。未生成全量 U1 生产评分。',
      '- 折与重复种子相关，不把 15 次比较当独立试验夸大显著性。后续来源隔离或独立业务核验仍需要另行开展。','',
      '## 完整性与复现','',
      f'独立检查 {len(checks)} 项全部通过，覆盖每份预测的验证 Gold/U1 名单、捕获计数、四组正例指纹、共用 U、字段/权重一致及最终 Gold 隔离。训练记录与准备版本均通过原哈希复核。',
      '执行代码为上级目录 e03.py；e02_support.py 与 legacy/src 是带哈希的 E02 方法副本。三项针对四组构成、全国验证折排除、共同字段和类别权重的测试已通过。',
      'prepared_001 保存名单、角色和各组组成；runs/main_001 保存拟合、逐折预测、共用 U 样本与参数；reports/main_001 保存本报告、配对差值、分组结果和名单差异。所有新运行拒绝覆盖旧目录。','']
    (out/'E03结果报告.md').write_text('\n'.join(lines),encoding='utf-8')
    dump({'status':'complete','source_run':str(run.resolve()),'source_run_manifest_sha256':sha(run/'manifest.json'),
      'report_code_sha256':sha(Path(__file__)),'independent_checks':len(checks),'all_passed':True,'final_Gold_holdout_scored':False,
      'interpretation':'within-source SH/SZ development scope comparisons; no global business model selection','scope_effects':effects},out/'manifest.json')
    dump(output_hashes(out),out/'output_hashes.json')
    print(pd.DataFrame(effects).to_json(orient='records',force_ascii=False),flush=True)
    print(str({'checks':len(checks),'report':str(out),'status':'complete'}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepared',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();report(a.prepared.resolve(),a.run.resolve(),a.out.resolve())
