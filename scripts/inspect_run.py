import json

with open('runs/20261117_162714_c1-full-v2/scalability.json') as f:
    s = json.load(f)
print('Scalability keys:', list(s.keys()))
print('timings_seconds keys:', list(s.get('timings_seconds', {}).keys()))
print('model_names:', s.get('model_names'))
print('memory_warning:', s.get('memory_warning'))

with open('runs/20261117_162714_c1-full-v2/evaluation.json') as f:
    e = json.load(f)
print('\nEvaluation splits:')
splits = e.get('splits', {})
for split_name in ['time', 'group_dest', 'random']:
    models = splits.get(split_name, {})
    for m, metrics in models.items():
        ap = metrics.get('average_precision', 0.0)
        roc = metrics.get('roc_auc', 0.0)
        conf = metrics.get('at_threshold', {}).get('confusion', {})
        tp = conf.get('tp', 0)
        fn = conf.get('fn', 0)
        fp = conf.get('fp', 0)
        tn = conf.get('tn', 0)
        total = tp + fn + fp + tn
        fraud = tp + fn
        thresh = metrics.get('threshold', 0.0)
        print(f'{split_name:10} | {m:18} | AP: {ap:.4f} | ROC-AUC: {roc:.4f} | tp+fn: {fraud:5d} | total: {total:7d} | thresh: {thresh:.4f}')
