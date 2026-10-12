import os
from pathlib import Path
os.environ['KAGGLE_API_TOKEN'] = (Path.home() / '.kaggle' / 'access_token').read_text().strip()
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()
print("AUTH OK", flush=True)
seen = set()
for q in ['bone tumor classification', 'bone fracture x-ray', 'musculoskeletal xray']:
    try:
        res = api.dataset_list(search=q, max_size=8) or []
        print(f"Q={q} n={len(res)}", flush=True)
        for d in res:
            if d.ref not in seen:
                seen.add(d.ref)
                print('  ', d.ref, '|', str(d.title)[:65], flush=True)
    except Exception as e:
        print(q, 'ERR', str(e)[:150], flush=True)
print("DONE", flush=True)
