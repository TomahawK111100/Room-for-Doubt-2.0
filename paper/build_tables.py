"""Regenerate manuscript tables from preserved compact experiment outputs."""
import csv
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNS = [('c100_s42', 'C100, base 42'), ('c100_s43', 'C100, base 43'), ('c10_s42', 'C10 Worse, base 42')]
METHODS = [('base','Base'), ('tta','Uniform TTA'), ('msp','MSP weighting'), ('gap','Gap weighting'), ('entropy','Entropy weighting'), ('resnet18__early_mean','Early-5, ResNet18'), ('base_resnet50__best_margin','Static, base features'), ('base_resnet50__history_mean','History-50, base features'), ('base_resnet50__early_mean','Early-5, base features')]

def read(run, name):
    with (ROOT/'evidence'/run/(name+'.csv')).open() as f:
        return list(csv.DictReader(f))

def table(name, columns, header, rows):
    text = '\\begin{tabular}{'+columns+'}\n\\toprule\n'+header+' \\\\\n\\midrule\n'
    text += '\n'.join(' & '.join(row)+' \\\\' for row in rows)
    text += '\n\\bottomrule\n\\end{tabular}\n'
    (ROOT/'revision/tables'/name).write_text(text)

def main():
    tables = {}
    long = []
    for run, title in RUNS:
        aggregate = {r['method']:r for r in read(run,'aggregate')}
        final = read(run,'test_validation_selected')
        selected = {r['method']:r for r in read(run,'validation_selected')}
        for r in final:
            if r['method']!='base':
                assert (r['views'],r['temperature']) == (selected[r['method']]['views'],selected[r['method']]['temperature'])
        for method, label in METHODS:
            a = aggregate[method]
            members = [r for r in final if r['method']==method or r['method'].startswith(method+'__')]
            assert len(members)==(3 if '__' in method else 1)
            assert abs(statistics.mean(float(r['acc']) for r in members)-float(a['acc']))<1e-10
            row = dict(run=run,method=method,acc=float(a['acc']),acc_sd=float(a['acc_sd']),auroc=float(a['error_auroc']),ece=statistics.mean(float(r['ece']) for r in members),nll=statistics.mean(float(r['nll']) for r in members),brier=statistics.mean(float(r['brier']) for r in members),selected_views=a['selected_views'])
            long.append(row)
        tables[run] = aggregate
    rows=[]
    for method,label in METHODS:
        cells=[label]
        for run,_ in RUNS:
            r=tables[run][method];value=f"{100*float(r['acc']):.2f}"
            if int(r['n_meta_seeds']):value+=f" $\\pm$ {100*float(r['acc_sd']):.2f}"
            cells.append(value)
        rows.append(cells)
    table('accuracy.tex','lccc','Method & C100 (42) & C100 (43) & C10 Worse (42)',rows)
    rows=[]
    for run,title in RUNS:
        for method in ['base','tta','entropy','base_resnet50__best_margin','base_resnet50__history_mean','base_resnet50__early_mean']:
            r=next(r for r in long if r['run']==run and r['method']==method)
            label=dict(METHODS)[method].replace(', base features','')
            rows.append([title,label,f"{r['auroc']:.4f}",f"{r['ece']:.4f}",f"{r['nll']:.4f}",f"{r['brier']:.4f}"])
    table('reliability.tex','llrrrr','Run & Method & AUROC $\\uparrow$ & ECE $\\downarrow$ & NLL $\\downarrow$ & Brier $\\downarrow$',rows)
    rows=[]
    for run,title in RUNS:
        for r in sorted(read(run,'test_validation_selected'),key=lambda r:r['method']):
            if r['method'].startswith('base_resnet50__early_mean__'):
                rows.append([title,r['method'].split('__')[-1],r['views'],r['temperature'],f"{float(r['gain_pp_vs_base']):+.2f}",f"[{float(r['ci_low_pp_vs_base']):+.2f}, {float(r['ci_high_pp_vs_base']):+.2f}]",f"[{float(r['ci_low_pp_vs_tta']):+.2f}, {float(r['ci_high_pp_vs_tta']):+.2f}]"])
    table('intervals.tex','llrrrrr','Run & Head seed & Views & $T$ & Gain & CI vs base & CI vs same-view TTA',rows)
    rows=[]
    for target,label in [('early_mean_2','First 2'),('early_mean','First 5'),('early_mean_10','First 10'),('early_mean_20','First 20'),('history_mean','All 50'),('late_mean_5','Last 5'),('best_margin','Static checkpoint')]:
        row=[label]
        for encoder in ['resnet18','base_resnet50']:
            a=tables['c100_s42'][encoder+'__'+target]
            row.extend([f"{100*float(a['acc']):.2f}",f"{float(a['error_auroc']):.4f}"])
        rows.append(row)
    table('windows.tex','lrrrr','Target & R18 acc. & R18 AUROC & Base acc. & Base AUROC',rows)
    rows=[]
    for target,label in [('early_mean','Early-5'),('best_margin','Static'),('history_mean','History-50')]:
        row=[label]
        for run,_ in RUNS:
            rec=[r for r in read(run,'reconstruction') if r['encoder']=='base_resnet50' and r['target']==target and r['partition']=='detector_eval']
            assert len(rec)==3 and all(int(r['samples'])==6750 for r in rec)
            row.extend([f"{statistics.mean(float(r['mse']) for r in rec):.4f}",f"{statistics.mean(float(r['r2']) for r in rec):.3f}"])
        rows.append(row)
    table('reconstruction.tex','lrrrrrr','Target & C100(42) MSE & $R^2$ & C100(43) MSE & $R^2$ & C10(42) MSE & $R^2$',rows)
    with (ROOT/'evidence/combined_results.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(long[0]));w.writeheader();w.writerows(long)
    print('Generated 5 tables; verified policy selections and 27 aggregate rows.')

if __name__=='__main__':main()
