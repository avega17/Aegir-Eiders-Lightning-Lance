from typing import Iterable, Optional
import pyarrow as pa

# Skeleton LanceDB IO helpers


def connect_uri(root: str = "./data/lancedb") -> str:
    return root


def ensure_table(uri: str, name: str, schema: Optional[pa.Schema] = None, mode: str = "append") -> None:
    """Create or open a LanceDB table. If schema is provided and table missing, create it."""
    return None


def append_records(uri: str, name: str, table: pa.Table) -> int:
    """Append Arrow records to LanceDB table; return rows written."""
    return 0

