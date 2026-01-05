# NetCDF (.nc) → CSV converter (Python)

This app watches the `Files/` folder for new `.nc` files and converts each one into a `.csv` file in `Converted files/`.

## Setup

Create/activate your environment (this workspace already has `.venv/` configured), then install dependencies:

```bash
./.venv/bin/python -m pip install -r requirements.txt
```

## Run

### Desktop app (select files and convert)

```bash
./.venv/bin/python app.py --gui
```

### Download page (paste links, auto-open with delay)

```bash
./.venv/bin/python app.py --download-page
```

Paste one link or multiple links (space/newline/comma separated). The app will open them automatically with a 10 second delay between each.

Select one or more `.nc` files from `Files/`, then press **Convert**. Output CSVs will be written to `Converted files/`.

### Watch mode (recommended)

```bash
./.venv/bin/python app.py --watch
```

Now drop any `.nc` file into `Files/` and it will produce `Converted files/<same-name>.csv`.

### One-time conversion

```bash
./.venv/bin/python app.py --once
```

## Optional: export only specific variables

Some NetCDFs contain many variables and can produce very large CSVs.

```bash
./.venv/bin/python app.py --once --variables chlor_a
```

(Use a comma-separated list for multiple variables.)

## CSV formats

For 2D grids (like `lat` × `lon`) you can choose:

- `--format long-sparse` (default): writes only non-missing cells as `lat,lon,value`
- `--format long-full`: writes every cell (can be extremely large)
- `--format matrix`: writes one row per latitude with longitude columns

For 3D grids (like `time` × `lat` × `lon`), `long-sparse` / `long-full` are supported and will write rows as `time,lat,lon,value`.

If you need to re-run a conversion, use:

```bash
./.venv/bin/python app.py --once --force
```
