#!/usr/bin/env python3
"""Run the independent v8 validator against the r2 recovery root."""
import csv
import importlib.util
import json
from pathlib import Path

HERE=Path(__file__).resolve();REPO=HERE.parents[1]
SPEC=importlib.util.spec_from_file_location('v8_validator',REPO/'scripts/validate_revision_v8_analysis.py')
validator=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(validator)
validator.ROOT=REPO/'reports/nns_vit_endpoint_revision_v8_r2_20260923'
validator.REG=REPO/'manifests/nns_vit_endpoint/revision_v8_r2_20260923'


def main():
    code=validator.main()
    with (validator.ROOT/'entropy_mass_image_values.csv').open() as stream:
        image_rows=list(csv.DictReader(stream))
    with (validator.ROOT/'entropy_mass_sensitivity.csv').open() as stream:
        summary=list(csv.DictReader(stream))
    na_cells=sum(row[f'q{quartile}_na_reason']=='NA_EMPTY_EARLY_OR_LATE_MASS_QUARTILE' for row in image_rows for quartile in range(1,5))
    if na_cells!=8:raise ValueError('R2_TYPED_NA_COUNT')
    if sum(int(row['n_na']) for row in summary if row['metric'].startswith('q'))!=8:raise ValueError('R2_SUMMARY_NA_COUNT')
    report=json.loads((validator.ROOT/'validation_report.json').read_text())
    report['r2_typed_empty_quartile_cells']=na_cells
    report['checks'].append('eight empty image-quartile cells retained as typed NA')
    (validator.ROOT/'validation_report.json').write_text(json.dumps(report,indent=2)+'\n')
    return code


if __name__=='__main__':raise SystemExit(main())
