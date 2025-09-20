from __future__ import annotations
from pathlib import Path
from typing import Union, Any, Dict
import pandas as pd
from types import SimpleNamespace
from .utils import sanitize

class Dataset:
    def __init__(self, df: pd.DataFrame, source: Union[str, Path] | None = None, sheet: Any = None):
        self.df = df
        self.source = str(source) if source is not None else None
        self.sheet = sheet
        self.columns = list(df.columns)
        self.col = SimpleNamespace(**{ sanitize(c): df[c] for c in df.columns })
    def summary(self) -> pd.DataFrame:
        dtypes = self.df.dtypes.astype(str)
        return pd.DataFrame({
            "dtype": dtypes,
            "non_null": self.df.notna().sum(),
            "nulls": self.df.isna().sum(),
            "unique": self.df.nunique(dropna=True),
        })
    def head(self, n: int = 5): return self.df.head(n)

def load_any(path: Union[str, Path], *, all_sheets: bool = False) -> Union[Dataset, Dict[str, Dataset]]:
    path = Path(path); suf = path.suffix.lower()
    if suf in {".csv", ".tsv", ".txt"}:
        df = pd.read_csv(path, sep=None, engine="python", header=0)
    elif suf in {".xlsx", ".xls"}:
        if all_sheets:
            sheets = pd.read_excel(path, sheet_name=None, header=0)
            return { name: Dataset(df.convert_dtypes(), source=path, sheet=name)
                     for name, df in sheets.items() }
        df = pd.read_excel(path, sheet_name=0, header=0)
    elif suf == ".parquet": df = pd.read_parquet(path)
    elif suf == ".feather": df = pd.read_feather(path)
    elif suf == ".json":
        try: df = pd.read_json(path, orient="records")
        except ValueError: df = pd.read_json(path, orient="table")
    elif suf == ".pkl": df = pd.read_pickle(path)
    else: raise ValueError(f"Unsupported file type: {suf}")
    return Dataset(df.convert_dtypes(), source=path)
