# NNS-ViT manuscript

This directory contains the first bilingual manuscript draft based on the R10-R4 outline and frozen 35-figure package. The English and Chinese versions use the same section structure, figure numbers, numerical results, and evidence boundaries.

Ready-to-read deliverables:

- NNS_ViT_论文初稿_中文版.md
- output/NNS_ViT_论文初稿_中文版.pdf
- output/NNS_ViT_manuscript_draft_en.pdf

## Reading order

1. `notes/scientific_story.md`
2. `notes/evidence_matrix.md`
3. `zh/` or `en/` in section order
4. `references/literature_notes.md`
5. `qc/drafting_status.md`

`scripts/build_publication_files.py` assembles the standalone Chinese Markdown file and both PDFs. It reads all 35 figures from the frozen R10-R4 package without copying or modifying them.

PDF build dependencies are listed in `requirements-pdf.txt`. A CJK-capable font can be supplied through `NNS_PDF_FONT`; otherwise the script uses ReportLab's built-in Chinese CID font. Example:

```bash
python -m pip install -r manuscript/requirements-pdf.txt
NNS_PDF_FONT=/path/to/NotoSansSC-Regular.ttf \
  python manuscript/scripts/build_publication_files.py
python manuscript/scripts/validate_publication_files.py
```

The current version is a scientific manuscript draft, not a submission-ready package. Author names, affiliations, target journal formatting, final bibliography metadata, and a small number of explicitly marked reference gaps remain to be supplied or verified. No new experiment, statistic, sample selection, or figure rendering was performed during drafting.
