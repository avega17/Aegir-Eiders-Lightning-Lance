from typing import Optional
import pyarrow as pa

# Skeleton: write Arrow tables into LanceDB tables (create/append semantics)


def lance_uri(root: str = "./data/lancedb") -> str:
    return root


def write_training_data(records: pa.Table, uri: str = lance_uri(), table: str = "training_data", mode: str = "append") -> str:
    """Create or append to a LanceDB table and return table name.
    mode: "append" or "overwrite"
    """
    return table

