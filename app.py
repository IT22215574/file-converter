from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import xarray as xr
from netCDF4 import Dataset


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = ROOT / "Files"
DEFAULT_OUTPUT_DIR = ROOT / "Converted files"
DEFAULT_STATE_PATH = ROOT / ".converted_state.json"


@dataclass(frozen=True)
class ConvertResult:
    input_path: Path
    output_path: Path
    rows: int


def _format_cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (np.floating, float)):
        if np.isnan(value):
            return ""
        return f"{float(value):g}"
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    return str(value)


def _load_state(state_path: Path) -> dict:
    if not state_path.exists():
        return {"files": {}}
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return {"files": {}}


def _save_state(state_path: Path, state: dict) -> None:
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def _fingerprint(path: Path) -> dict:
    st = path.stat()
    return {"size": st.st_size, "mtime": int(st.st_mtime)}


def wait_for_stable_file(path: Path, *, checks: int = 3, interval_s: float = 0.5) -> None:
    """Wait until file size+mtime stop changing.

    This helps when a file is still being copied into the folder.
    """
    last = None
    stable = 0
    while stable < checks:
        if not path.exists():
            stable = 0
            time.sleep(interval_s)
            continue
        current = _fingerprint(path)
        if current == last:
            stable += 1
        else:
            stable = 0
        last = current
        time.sleep(interval_s)


def convert_nc_to_csv(
    nc_path: Path,
    output_dir: Path,
    *,
    variables: list[str] | None = None,
    engine: str | None = "netcdf4",
    output_format: str = "long-sparse",
) -> ConvertResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not nc_path.exists():
        raise FileNotFoundError(nc_path)

    # Use xarray for cheap metadata inspection (no DataFrame conversion!)
    xr_ds = xr.open_dataset(nc_path, engine=engine)
    try:
        if variables:
            missing = [v for v in variables if v not in xr_ds.data_vars]
            if missing:
                raise ValueError(f"Variables not found in dataset: {missing}")
            selected = variables
        else:
            # Prefer lat/lon gridded variables if present
            preferred = [
                v
                for v in xr_ds.data_vars
                if {"lat", "lon"}.issubset(set(xr_ds[v].dims))
            ]
            selected = preferred or list(xr_ds.data_vars)
    finally:
        xr_ds.close()

    # Stream using netCDF4 to avoid loading huge tables into memory
    ds = Dataset(str(nc_path), mode="r")
    try:
        total_rows = 0
        multiple = len(selected) != 1

        for var_name in selected:
            if var_name not in ds.variables:
                continue
            var = ds.variables[var_name]
            dims = list(var.dimensions)

            safe_var = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in var_name)
            out_path = (
                output_dir / f"{nc_path.stem}__{safe_var}.csv"
                if multiple
                else output_dir / f"{nc_path.stem}.csv"
            )

            with out_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)

                if len(dims) == 1:
                    d0 = dims[0]
                    coord0 = ds.variables[d0][:] if d0 in ds.variables else np.arange(var.shape[0])
                    writer.writerow([d0, var_name])
                    for i in range(var.shape[0]):
                        value = var[i]
                        if np.ma.isMaskedArray(value):
                            value = value.filled(np.nan)
                        writer.writerow([_format_cell(coord0[i]), _format_cell(value)])
                        total_rows += 1

                elif len(dims) == 2:
                    d0, d1 = dims
                    coord0 = ds.variables[d0][:] if d0 in ds.variables else np.arange(var.shape[0])
                    coord1 = ds.variables[d1][:] if d1 in ds.variables else np.arange(var.shape[1])

                    if output_format == "matrix":
                        writer.writerow([d0] + [_format_cell(x) for x in coord1])
                        for i in range(var.shape[0]):
                            row = var[i, :]
                            if np.ma.isMaskedArray(row):
                                row = row.filled(np.nan)
                            writer.writerow([_format_cell(coord0[i])] + [_format_cell(x) for x in row])
                            total_rows += 1

                    elif output_format in ("long-sparse", "long-full"):
                        writer.writerow([d0, d1, var_name])
                        for i in range(var.shape[0]):
                            row = var[i, :]
                            if np.ma.isMaskedArray(row):
                                row = row.filled(np.nan)

                            if output_format == "long-full":
                                for j in range(var.shape[1]):
                                    writer.writerow(
                                        [_format_cell(coord0[i]), _format_cell(coord1[j]), _format_cell(row[j])]
                                    )
                                    total_rows += 1
                                continue

                            # long-sparse: write only non-NaN values (much smaller for sparse grids)
                            if row.dtype.kind in ("f", "c"):
                                mask = ~np.isnan(row)
                            else:
                                mask = np.ones(row.shape, dtype=bool)

                            idx = np.nonzero(mask)[0]
                            c0 = _format_cell(coord0[i])
                            for j in idx:
                                writer.writerow([c0, _format_cell(coord1[j]), _format_cell(row[j])])
                                total_rows += 1

                    else:
                        raise ValueError(
                            f"Unknown --format '{output_format}'. Use: long-sparse, long-full, matrix."
                        )

                else:
                    raise ValueError(
                        f"Variable '{var_name}' has {len(dims)} dims ({dims}). "
                        "This app currently supports 1D and 2D variables only. "
                        "Try specifying a 2D lat/lon variable via --variables."
                    )

        # Report the last output file if only one variable was written; otherwise point to the folder.
        result_path = out_path if len(selected) == 1 else output_dir
        return ConvertResult(input_path=nc_path, output_path=result_path, rows=total_rows)
    finally:
        ds.close()


def iter_nc_files(input_dir: Path) -> Iterable[Path]:
    if not input_dir.exists():
        return []
    return sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() == ".nc"])


def process_existing(
    input_dir: Path,
    output_dir: Path,
    state_path: Path,
    *,
    variables: list[str] | None,
    force: bool = False,
) -> int:
    state = _load_state(state_path)
    changed = 0

    for nc_path in iter_nc_files(input_dir):
        fp = _fingerprint(nc_path)
        prev = state.get("files", {}).get(str(nc_path))
        if (not force) and prev == fp:
            continue

        wait_for_stable_file(nc_path)
        result = convert_nc_to_csv(
            nc_path,
            output_dir,
            variables=variables,
            output_format=state.get("format", "long-sparse"),
        )
        out_label = (
            result.output_path.name if result.output_path.is_file() else str(result.output_path)
        )
        print(f"Converted: {result.input_path.name} -> {out_label} ({result.rows} row(s))")

        state.setdefault("files", {})[str(nc_path)] = fp
        changed += 1

    _save_state(state_path, state)
    return changed


def watch_folder(
    input_dir: Path,
    output_dir: Path,
    state_path: Path,
    *,
    variables: list[str] | None,
) -> None:
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer

    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    class Handler(FileSystemEventHandler):
        def on_created(self, event):  # type: ignore[override]
            if getattr(event, "is_directory", False):
                return
            path = Path(getattr(event, "src_path"))
            if path.suffix.lower() != ".nc":
                return
            try:
                process_existing(input_dir, output_dir, state_path, variables=variables)
            except Exception as exc:
                print(f"Failed converting {path.name}: {exc}")

        def on_moved(self, event):  # type: ignore[override]
            # Some copy operations create a temp file then rename
            self.on_created(event)

        def on_modified(self, event):  # type: ignore[override]
            # In case file is written in place, we re-scan
            self.on_created(event)

    observer = Observer()
    observer.schedule(Handler(), str(input_dir), recursive=False)
    observer.start()

    print(f"Watching: {input_dir}")
    print(f"Output:   {output_dir}")
    print("Drop .nc files into the input folder to convert them.")

    try:
        # Also process anything already there
        process_existing(input_dir, output_dir, state_path, variables=variables)
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        observer.stop()
        observer.join()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Watch the Files/ folder for NetCDF (.nc) files and convert them to CSV."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help='Input folder (default: "Files/")',
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help='Output folder (default: "Converted files/")',
    )
    parser.add_argument(
        "--state",
        type=Path,
        default=DEFAULT_STATE_PATH,
        help="Where to store conversion state to avoid re-processing",
    )
    parser.add_argument(
        "--variables",
        type=str,
        default=None,
        help="Comma-separated list of variables to export (optional)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="long-sparse",
        choices=["long-sparse", "long-full", "matrix"],
        help="CSV output format for 2D variables (default: long-sparse)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-convert even if the file was already processed",
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--watch",
        action="store_true",
        help="Watch input folder continuously (default)",
    )
    mode.add_argument(
        "--once",
        action="store_true",
        help="Convert any unprocessed .nc files once, then exit",
    )

    args = parser.parse_args()
    if not args.watch and not args.once:
        args.watch = True

    if args.variables:
        args.variables = [v.strip() for v in args.variables.split(",") if v.strip()]
    else:
        args.variables = None

    return args


def main() -> None:
    args = parse_args()

    # Persist the chosen format so watch mode uses it too.
    state = _load_state(args.state)
    state["format"] = args.format
    _save_state(args.state, state)

    if args.once:
        changed = process_existing(
            args.input,
            args.output,
            args.state,
            variables=args.variables,
            force=args.force,
        )
        print(f"Done. Converted {changed} file(s).")
        return

    watch_folder(args.input, args.output, args.state, variables=args.variables)


if __name__ == "__main__":
    main()
