import numpy as np
import pandas as pd
from typing import Dict, List

from chain_targets import build_chained_targets
from config import Config
from data_loader import Data
from data_loader import ChainedData
from pipeline import Pipeline


def make_group_rows(group_name: str) -> List[Dict[str, object]]:
    rows = []
    for i in range(3):
        rows.append(
            {
                "y1": group_name,
                "y2": "Suggestion",
                "y3": "Payment",
                "y4": "Subscription cancellation",
                Config.TICKET_SUMMARY: f"{group_name} summary payment {i}",
                Config.INTERACTION_CONTENT: f"{group_name} content payment {i}",
            }
        )
    for i in range(3):
        rows.append(
            {
                "y1": group_name,
                "y2": "Suggestion",
                "y3": "Refund",
                "y4": "Within 14 days",
                Config.TICKET_SUMMARY: f"{group_name} summary refund {i}",
                Config.INTERACTION_CONTENT: f"{group_name} content refund {i}",
            }
        )
    for i in range(3):
        rows.append(
            {
                "y1": group_name,
                "y2": "Problem/Fault",
                "y3": "Payment issue",
                "y4": "Risk Control",
                Config.TICKET_SUMMARY: f"{group_name} summary fault {i}",
                Config.INTERACTION_CONTENT: f"{group_name} content fault {i}",
            }
        )
    return rows


def make_dataset() -> pd.DataFrame:
    rows = make_group_rows("GroupA")
    rows.extend(make_group_rows("GroupB"))
    return pd.DataFrame(rows)


def test_build_chained_targets_uses_missing_placeholder():
    df = pd.DataFrame(
        [
            {
                "y1": "GroupA",
                "y2": "Suggestion",
                "y3": np.nan,
                "y4": "",
                Config.TICKET_SUMMARY: "summary",
                Config.INTERACTION_CONTENT: "content",
            }
        ]
    )

    chained_df = build_chained_targets(df)

    assert chained_df.loc[0, "chain_y2"] == "Suggestion"
    assert chained_df.loc[0, "chain_y2_y3"] == "Suggestion || <missing>"
    assert chained_df.loc[0, "chain_y2_y3_y4"] == "Suggestion || <missing> || <missing>"


def test_chained_data_creates_all_three_target_levels():
    df = build_chained_targets(pd.DataFrame(make_group_rows("GroupA")))
    X = np.arange(len(df) * 4).reshape(len(df), 4)

    data = ChainedData(X, df, Config.CHAIN_TARGET_COLUMNS)

    assert data.get_target_names() == [
        "Type 2",
        "Type 2 + Type 3",
        "Type 2 + Type 3 + Type 4",
    ]

    for target_name in data.get_target_names():
        data.set_active_target(target_name)
        assert data.get_active_target() == target_name
        assert data.get_X_train().shape[0] > 0
        assert data.get_X_test().shape[0] > 0
        assert len(data.get_type_y_train()) > 0
        assert len(data.get_type_y_test()) > 0


def test_pipeline_runs_all_chained_targets_with_uniform_model_interface(monkeypatch):
    df = make_dataset()
    calls = []

    class StubModel:
        def __init__(self, model_name, embeddings, y):
            self.model_name = model_name
            self.embeddings = embeddings
            self.y = y

        def train(self, data):
            calls.append(("train", data.get_active_target(), tuple(sorted(set(data.get_type())))))

        def predict(self, X_test):
            calls.append(("predict", X_test.shape[0]))

        def print_results(self, data):
            calls.append(("print_results", data.get_active_target()))

    def fake_load_data(self):
        return df.copy()

    def fake_preprocess_data(self, input_df):
        return input_df.copy()

    def fake_get_embeddings(self, input_df):
        X = np.arange(len(input_df) * 3).reshape(len(input_df), 3)
        return X, input_df

    monkeypatch.setattr(Pipeline, "load_data", fake_load_data)
    monkeypatch.setattr(Pipeline, "preprocess_data", fake_preprocess_data)
    monkeypatch.setattr(Pipeline, "get_embeddings", fake_get_embeddings)

    pipeline = Pipeline(mode="chained")
    pipeline.model_classes = [("StubModel", StubModel)]

    pipeline.run()

    trained_targets = [call[1] for call in calls if call[0] == "train"]
    reported_targets = [call[1] for call in calls if call[0] == "print_results"]

    assert trained_targets == [
        "Type 2",
        "Type 2 + Type 3",
        "Type 2 + Type 3 + Type 4",
        "Type 2",
        "Type 2 + Type 3",
        "Type 2 + Type 3 + Type 4",
    ]
    assert reported_targets == trained_targets


def test_pipeline_single_label_mode_uses_original_data_object():
    df = pd.DataFrame(make_group_rows("GroupA"))
    df["y"] = df["y2"]
    X = np.arange(len(df) * 3).reshape(len(df), 3)

    pipeline = Pipeline(mode="single_label")
    data = pipeline.get_data_object(X, df)

    assert isinstance(data, Data)
