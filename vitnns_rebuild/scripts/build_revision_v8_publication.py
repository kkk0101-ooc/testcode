#!/usr/bin/env python3
"""Build revision-v8 publications without changing the sealed v7 builder."""
from __future__ import annotations
import importlib.util
import argparse
import json
from pathlib import Path

HERE=Path(__file__).resolve();REPO=HERE.parents[1]
SPEC=importlib.util.spec_from_file_location('v7_builder',REPO/'scripts/build_revision_v7_publication.py')
builder=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(builder)
builder.REPORT=REPO/'reports/nns_vit_endpoint_revision_v8_r2_20260923'
builder.__file__=str(HERE)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--pandoc',type=Path,required=True);parser.add_argument('--tectonic',type=Path,required=True);args=parser.parse_args()
    builder.build(args.pandoc.resolve(),args.tectonic.resolve())
    for language in ('zh','en'):
        for kind in ('manuscript','supplement'):
            tex=builder.PUB/language/f'{kind}.tex'
            text=tex.read_text().replace('Generated from Markdown by scripts/build_revision_v7_publication.py','Generated from Markdown by scripts/build_revision_v8_publication.py',1)
            tex.write_text(text)
    report_path=builder.REPORT/'latex_build_report.json';report=json.loads(report_path.read_text())
    report['generator']=builder.rec(HERE)
    report['outputs']=[builder.rec(builder.PUB/language/f'{kind}.{suffix}') for language in ('zh','en') for kind in ('manuscript','supplement') for suffix in ('tex','pdf')]
    report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    return 0


if __name__=='__main__':raise SystemExit(main())
