# exadata-validator

`exadata-validator` validates data-model folders using DuckDB.

## Install with pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install exadata-validator
```

Validate a data model folder:

```bash
exadata-validator validate-data-model /path/to/data_model_folder
```

Run as a Python module:

```bash
python -m validator validate-data-model /path/to/data_model_folder
```

Useful options:

```bash
exadata-validator validate-data-model /path/to/data_model_folder --fail-fast
exadata-validator validate-data-model /path/to/data_model_folder --format html --output report.html
```

By default the command collects and reports all validation errors in text format. If `--format html` is used without `--output`, the report is written under `/tmp`.

Use `exadata-validator validate-data-model --help` for reporting, output, and threading options.

## Folder Layout

```text
/path/to/data_model_folder/
  CDEsMetadata.json
  dataset1.csv
  dataset2.csv
```

## Validation Notes

- CSV validation queries files directly with DuckDB and uses fused aggregate checks to reduce scan overhead.
- Folder-level dataset uniqueness is enforced across all CSV files via SQL using normalized codes (`trim + lower`).
