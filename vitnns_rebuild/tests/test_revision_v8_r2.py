import importlib.util
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('v8r2',ROOT/'scripts/run_revision_v8_r2_analysis.py')
v8r2=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(v8r2)


def test_r2_preserves_frozen_thresholds_and_types_empty_cells():
    source=(ROOT/'scripts/run_revision_v8_analysis.py').read_text()
    recovery=(ROOT/'scripts/run_revision_v8_r2_analysis.py').read_text()
    assert "np.quantile(data,np.asarray(base.MASS_QUANTILES)/100.0,method='linear')" in recovery
    assert 'NA_EMPTY_EARLY_OR_LATE_MASS_QUARTILE' in recovery
    assert 'ENTROPY_QUARTILE_EMPTY' in source


def test_r2_uses_fresh_roots_and_binds_predecessor():
    assert v8r2.base.ROOT.name=='nns_vit_endpoint_revision_v8_r2_20260923'
    assert v8r2.base.REGISTRY_ROOT.name=='revision_v8_r2_20260923'
    assert v8r2.R1.name=='nns_vit_endpoint_revision_v8_r1_20260923'
