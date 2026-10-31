"""Read raw inputs without modifying them."""
import logging
from pathlib import Path

import pandas as pd

from .config import POSITIVE_PATH, UNLABELED_PATH

LOGGER = logging.getLogger(__name__)
ENCODINGS = ("utf-8-sig", "utf-8", "gb18030")
IDENTIFIER_COLUMNS = {
    "leads_company_id", "mid_company_clean_id", "mid_company_id", "raw_data_id",
    "credit_code", "sample_type", "observed_positive",
}


class DataFileMissingError(FileNotFoundError):
    """Raised when an expected raw CSV is absent."""


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise DataFileMissingError(str(path))
    errors: list[str] = []
    for encoding in ENCODINGS:
        try:
            # IDs must never pass through float parsing: U leads_company_id may
            # already be scientific notation in its source CSV and cannot be
            # used as an entity key or an is_in_leads proxy.
            header = pd.read_csv(path, encoding=encoding, nrows=0).columns
            dtype = {column: "string" for column in IDENTIFIER_COLUMNS if column in header}
            frame = pd.read_csv(path, encoding=encoding, low_memory=False, dtype=dtype)
            LOGGER.info("已读取 %s，编码=%s，shape=%s", path.name, encoding, frame.shape)
            LOGGER.info("字段：%s", list(frame.columns))
            for column in ("leads_company_id", "credit_code", "company_name"):
                if column in frame:
                    count = int(frame[column].dropna().duplicated().sum())
                    LOGGER.info("%s 重复值数量：%d", column, count)
            return frame
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
    raise UnicodeError(f"无法读取 {path}；已尝试 {ENCODINGS}。详情：{' | '.join(errors)}")


def load_positive(path: Path | str = POSITIVE_PATH) -> pd.DataFrame:
    return _read_csv(Path(path))


def load_unlabeled(path: Path | str = UNLABELED_PATH) -> pd.DataFrame:
    return _read_csv(Path(path))
