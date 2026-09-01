# Test all dashboard imports
import matplotlib
matplotlib.use('Agg')
import torch, torchvision, numpy, pandas, PIL, pathlib, json, pickle, io, base64, datetime, warnings
print('Core imports: OK')

# Test model paths exist
from pathlib import Path
IMPL = Path(r'e:/Sharvayu data/Malware/Symbiosis Nagpur SIT/7th SEM/Forest Fire task/Implementation')
tab_dir = IMPL / 'models' / 'tabular'
files = ['classifier.pkl', 'regressor.pkl', 'scaler.pkl', 'le_month.pkl', 'le_day.pkl', 'metadata.json']
for f in files:
    p = tab_dir / f
    status = "EXISTS" if p.exists() else "MISSING"
    print(f'  {f}: {status}')
print()

# Test tabular model loads
import pickle
clf = pickle.load(open(tab_dir / 'classifier.pkl', 'rb'))
reg = pickle.load(open(tab_dir / 'regressor.pkl', 'rb'))
sc  = pickle.load(open(tab_dir / 'scaler.pkl', 'rb'))
le_m = pickle.load(open(tab_dir / 'le_month.pkl', 'rb'))
le_d = pickle.load(open(tab_dir / 'le_day.pkl', 'rb'))
print(f'Classifier loaded: {type(clf).__name__}')
print(f'Regressor loaded:  {type(reg).__name__}')
print(f'Months: {list(le_m.classes_)}')
print(f'Days:   {list(le_d.classes_)}')

# Test a prediction
import numpy as np
row = {'X':7,'Y':5,'month':'aug','day':'fri','FFMC':92.3,'DMC':85.3,'DC':488.0,'ISI':14.7,'temp':22.2,'RH':29,'wind':5.4,'rain':0.0}
import json
with open(tab_dir / 'metadata.json') as f:
    tmeta = json.load(f)
num_feats = tmeta['numerical_features']
month_val = 'aug'
day_val = 'fri'
num_vals = [float(row.get(ff, 0.0)) for ff in num_feats]
month_enc = le_m.transform([month_val])[0]
day_enc   = le_d.transform([day_val])[0]
X = np.array(num_vals + [month_enc, day_enc]).reshape(1, -1)
fire_prob = clf.predict_proba(X)[0, 1]
area_log  = reg.predict(X)[0]
area_pred = max(0, float(np.expm1(area_log)))
print(f'Test prediction: P(fire)={fire_prob:.3f}, area={area_pred:.2f}ha')
print()
print('All checks PASSED')
