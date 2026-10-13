"""Complete finite descriptive tables; no outcome-dependent exclusions."""
import json
import numpy as np
from .common import *

def candidates():
    seedrows=[];mappingrows=[]
    for seed in SEEDS:
        root=OUT/'B1'/f'seed_{seed}'
        for filename,outrows,nkey,suffix in [('candidate_agreement.csv',seedrows,'n','_agree'),
                                          ('mapping_agreement.csv',mappingrows,'n_mappings','_full_mapping_agree')]:
            data=csv_read(root/filename)
            if filename.startswith('candidate'):data=[r for r in data if r['query_type']=='-1']
            for cross in (0,1):
                for treatment in ('matched','wrong'):
                    subsets=('all','receiver_initial_distinct','receiver_donor_distinct','initial_donor_distinct','all_three_distinct') if treatment=='matched' else ('all','receiver_donor_distinct')
                    for subset in subsets:
                        for s in ('ALL',*SUFFIXES):
                            rows=[r for r in data if int(r['cross'])==cross and r['treatment']==treatment and
                                  r['subset']==subset and (s=='ALL' or r['S']==s)]
                            n=sum(int(r[nkey]) for r in rows)
                            row={'seed':seed,'cross':cross,'treatment':treatment,'subset':subset,'S':s,
                                 'n':n,'direction_cells':len(rows),'direction_cells_nonempty':sum(int(r[nkey])>0 for r in rows)}
                            metrics=[cn+suffix for cn in ('receiver','initial','donor_rule')]
                            if filename.startswith('candidate'):metrics += [cn+'_CE' for cn in ('receiver','initial','donor_rule')]
                            else:metrics+=['item_bijection','position_bijection','mutually_inverse']
                            for metric in metrics:
                                valid=[r for r in rows if r[metric]!='' and int(r[nkey])>0]
                                den=sum(int(r[nkey]) for r in valid)
                                row[metric]=sum(float(r[metric])*int(r[nkey]) for r in valid)/den if den else ''
                                row[metric+'_equal_direction']=float(np.mean([float(r[metric]) for r in valid])) if valid else ''
                            outrows.append(row)
    csv_write(OUT/'candidate_seed_summary.csv',seedrows)
    csv_write(OUT/'candidate_mapping_seed_summary.csv',mappingrows)
    overall=[]
    for rows,name in [(seedrows,'query'),(mappingrows,'mapping')]:
        keys={k for r in rows for k in r if k not in ('seed','cross','treatment','subset','S','n','direction_cells','direction_cells_nonempty')}
        for cross in (0,1):
            for treatment in ('matched','wrong'):
                subsets=sorted({r['subset'] for r in rows if r['treatment']==treatment})
                for subset in subsets:
                    for s in ('ALL',*SUFFIXES):
                        selected=[r for r in rows if r['cross']==cross and r['treatment']==treatment and r['subset']==subset and r['S']==s]
                        overall.append({'unit':name,'cross':cross,'treatment':treatment,'subset':subset,'S':s,
                            'n_observations_all_seeds':sum(r['n'] for r in selected),'seeds':len(selected),
                            **{k:float(np.mean([r[k] for r in selected if r[k]!=''])) if any(r[k]!='' for r in selected) else '' for k in sorted(keys)}})
    # Query and mapping columns differ, so use a union with blanks.
    keys=list(dict.fromkeys(k for r in overall for k in r))
    csv_write(OUT/'candidate_overall_summary.csv',[{k:r.get(k,'') for k in keys} for r in overall])

def native_summary():
    summaries=[];strata=[]
    for cohort in ('B1','B_reserved'):
        for seed in SEEDS:
            root=OUT/cohort/f'seed_{seed}'
            rows=[r for d in (0,10) for s in SUFFIXES for r in csv_read(root/f'native_d{d}_S{s}.csv')]
            for r in rows:strata.append({'cohort':cohort,**r})
            for treatment in ('native','native_translated','sham'):
                selected=[r for r in rows if r['treatment']==treatment and r['query_type']=='-1']
                fine=[r for r in rows if r['treatment']==treatment and r['query_type']!='-1']
                minimum=min(fine,key=lambda r:float(r['literal_target_accuracy'] if treatment=='native_translated' else r['original_target_accuracy']))
                summaries.append({'cohort':cohort,'seed':seed,'treatment':treatment,
                    'original_target_accuracy':float(np.mean([float(r['original_target_accuracy']) for r in selected])),
                    'literal_target_accuracy':float(np.mean([float(r['literal_target_accuracy']) for r in selected])),
                    'original_target_CE':float(np.mean([float(r['original_target_CE']) for r in selected])),
                    'literal_target_CE':float(np.mean([float(r['literal_target_CE']) for r in selected])),
                    'min_actual_task_accuracy':float(minimum['literal_target_accuracy'] if treatment=='native_translated' else minimum['original_target_accuracy']),
                    'minimum_r':minimum['r'],'minimum_H':minimum['H'],'minimum_S':minimum['S'],
                    'minimum_translated_S':minimum['translated_S'],'minimum_d':minimum['d'],
                    'minimum_query_type':minimum['query_type'],'minimum_n':int(minimum['n'])})
    csv_write(OUT/'native_all_strata.csv',strata);csv_write(OUT/'native_seed_summary.csv',summaries)

def translation_mappings():
    rows=[]
    for cohort in ('B1','B_reserved'):
        for seed in SEEDS:
            root=OUT/cohort/f'seed_{seed}'
            for d in (0,10):
                for s in SUFFIXES:
                    saved=dict(np.load(root/f'translation_d{d}_S{s}.npz'))
                    # All queries reconstruct both maps. Full-map results do not
                    # turn incomplete query masks into purported complete maps.
                    for a,b in PAIRS:
                        target=saved['Q_original'][b];literal=saved['Q_literal'][b]
                        for treatment in ('original','translated'):
                            pred=saved[f'patch_{treatment}_mapping_predictions'][a,b]
                            fwd,rev=pred[:,0],pred[:,1]
                            bij=np.all(np.sort(fwd,axis=-1)==np.arange(7),axis=-1)
                            revbij=np.all(np.sort(rev,axis=-1)==np.arange(7),axis=-1)
                            mutual=bij & revbij & np.all(np.take_along_axis(rev,fwd,axis=-1)==np.arange(7),axis=-1)
                            donor=saved['Q_original' if treatment=='original' else 'Q_literal'][a]
                            candidates={'original':target,'literal':literal,'donor_rule':donor}
                            distinct=np.any(target!=literal,axis=(-1,-2))
                            for subset,mask in [('all',np.ones(len(pred),bool)),('original_literal_distinct',distinct)]:
                                n=int(mask.sum())
                                rows.append({'cohort':cohort,'seed':seed,'donor':a,'receiver':b,'cross':int(CONDITIONS[a][0]!=CONDITIONS[b][0]),
                                  'd':d,'S':s,'treatment':treatment,'subset':subset,'n_mappings':n,'coverage':n/len(pred),
                                  'item_bijection':float(bij[mask].mean()) if n else '',
                                  'position_bijection':float(revbij[mask].mean()) if n else '',
                                  'mutually_inverse':float(mutual[mask].mean()) if n else '',
                                  **{name+'_full_mapping_agree':float(np.all(pred==ans,axis=(-1,-2))[mask].mean()) if n else '' for name,ans in candidates.items()}})
    csv_write(OUT/'translation_full_mapping_diagnostics.csv',rows)

if __name__=='__main__':
    candidates();native_summary();translation_mappings()
