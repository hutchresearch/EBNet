import random
from astropy.table import Table

def drop_random_columns(table: Table, cols_to_consider=None, n_drop=1, seed=None) -> Table:
    """
    Return a copy of the table with n_drop random columns removed.
    
    Parameters
    ----------
    table : Table
        The input astropy Table.
    cols_to_consider : list[str] or None
        If provided, restricts which columns are candidates for dropping.
        Otherwise, all columns are candidates.
    n_drop : int
        Number of columns to drop.
    seed : int or None
        Random seed for reproducibility.
    """
    rng = random.Random(seed)
    if cols_to_consider is None:
        candidates = list(table.colnames)
    else:
        candidates = [c for c in cols_to_consider if c in table.colnames]

    if n_drop > len(candidates):
        raise ValueError("n_drop is larger than available candidate columns.")

    drop_cols = rng.sample(candidates, n_drop)
    new_table = table.copy()
    new_table.remove_columns(drop_cols)

    print(f"Dropped columns: {drop_cols}")
    return new_table
