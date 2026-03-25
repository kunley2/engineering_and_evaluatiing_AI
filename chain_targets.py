import pandas as pd

from config import Config


def build_chained_targets(df: pd.DataFrame) -> pd.DataFrame:
    chained_df = df.copy()
    labels = {}

    for col in Config.TYPE_COLS:
        labels[col] = (
            chained_df[col]
            .fillna(Config.MISSING_LABEL)
            .astype(str)
            .str.strip()
            .replace('', Config.MISSING_LABEL)
        )

    chained_df[Config.CHAIN_TARGET_COLUMNS['Type 2']] = labels['y2']
    chained_df[Config.CHAIN_TARGET_COLUMNS['Type 2 + Type 3']] = (
        labels['y2'] + Config.CHAIN_SEPARATOR + labels['y3']
    )
    chained_df[Config.CHAIN_TARGET_COLUMNS['Type 2 + Type 3 + Type 4']] = (
        labels['y2']
        + Config.CHAIN_SEPARATOR
        + labels['y3']
        + Config.CHAIN_SEPARATOR
        + labels['y4']
    )
    return chained_df
