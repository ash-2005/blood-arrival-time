import numpy as np
from pathlib import Path

r = np.load('data/fc_bias_results.npz')
print('Keys:', list(r.files))
for k in r.files:
    v = r[k]
    shp = v.shape if hasattr(v, 'shape') else '?'
    print(f'  {k}: shape={shp} dtype={v.dtype}')
print()
print('mean_abs_delta:', float(r['mean_abs_delta']))
print('matrix_corr:', float(r['matrix_corr']))
print('spearman_rho:', float(r['spearman_rho']))
print('spearman_p:', float(r['spearman_p']))
print('max_pair_delta:', float(r['max_pair_delta']))
print()
print('fc_legacy range:', r['fc_legacy'].min().round(4), r['fc_legacy'].max().round(4))
print('delta_fc range:', r['delta_fc'].min().round(4), r['delta_fc'].max().round(4))
print()
for f in ['figures/phase2_fc_matrices.png', 'figures/phase2_bias_scatter.png', 'report.md']:
    p = Path(f)
    print(f'{f}  exists={p.exists()}  {p.stat().st_size} bytes')
