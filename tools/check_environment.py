"""Offline checks only. Does not call Kakao or change data/model files."""
import sys,json,importlib.metadata as metadata,hashlib,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
parser=argparse.ArgumentParser();parser.add_argument('--full-hash',action='store_true');args=parser.parse_args()
required=['legacy/app.js','legacy/core.js','legacy/data.js','legacy/index.html','legacy/style.css','legacy/server.js','legacy/live_route_features.py','source/links.geojson','data/link_scores.json','models/route_lightgbm.joblib','training/predict_routes.py','training/predict_live_routes.py','training/processed/features.json']
errors=[]
if sys.version_info[:2]!=(3,12):errors.append('Use Python 3.12.')
for name in required:
 if not (ROOT/name).is_file():errors.append('Missing: '+name)
for name in ['numpy','pandas','geopandas','pyproj','shapely','joblib','scikit-learn','lightgbm','pyarrow','pyogrio']:
 try:print(name+' '+metadata.version(name))
 except metadata.PackageNotFoundError:errors.append('Missing package: '+name)
if not errors:
 import joblib
 model=joblib.load(ROOT/'models/route_lightgbm.joblib')
 names=json.loads((ROOT/'training/processed/features.json').read_text(encoding='utf-8'))
 if len(names)!=11 or model.feature_name_!=names:errors.append('Model feature names/order mismatch.')
if args.full_hash:
 manifest=json.loads((ROOT/'bundle_manifest.json').read_text(encoding='utf-8'))
 for entry in manifest['files']:
  path=ROOT/entry['path']
  if not path.is_file():errors.append('Missing manifest file: '+entry['path']);continue
  if hashlib.sha256(path.read_bytes()).hexdigest()!=entry['sha256']:errors.append('Hash mismatch: '+entry['path'])
for error in errors:print('FAIL: '+error)
print('PASS: offline environment and model checks' if not errors else 'Environment check failed')
sys.exit(1 if errors else 0)
