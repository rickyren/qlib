# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from unittest.mock import patch

import pandas as pd

from qlib.contrib.report.analysis_position import risk_analysis as risk_analysis_module


def _monthly_analysis_without_annualized_return() -> pd.DataFrame:
    metrics = ["max_drawdown", "information_ratio", "std"]
    index = pd.MultiIndex.from_product([["excess_return_without_cost", "excess_return_with_cost"], metrics])
    analysis = pd.DataFrame(
        {
            "risk": [value / 100 for value in range(1, len(index) + 1)],
            "date": pd.Timestamp("2024-01-31"),
        },
        index=index,
    )
    return analysis


def test_risk_analysis_graph_skips_missing_monthly_metric() -> None:
    report = pd.DataFrame(index=pd.DatetimeIndex([], name="date"))
    monthly_analysis = _monthly_analysis_without_annualized_return()

    with patch.object(
        risk_analysis_module,
        "_get_monthly_risk_analysis_with_report",
        return_value=monthly_analysis,
    ):
        figures = risk_analysis_module.risk_analysis_graph(
            analysis_df=None,
            report_normal_df=report,
            show_notebook=False,
        )

    assert [figure.layout.title.text for figure in figures] == ["max_drawdown", "information_ratio", "std"]


def test_risk_analysis_graph_accepts_report_without_complete_month() -> None:
    report = pd.DataFrame(
        {
            "return": [0.01, -0.005],
            "bench": [0.002, -0.001],
            "cost": [0.0005, 0.0005],
            "turnover": [0.1, 0.1],
        },
        index=pd.DatetimeIndex(["2024-01-02", "2024-01-03"], name="date"),
    )

    figures = risk_analysis_module.risk_analysis_graph(
        analysis_df=None,
        report_normal_df=report,
        show_notebook=False,
    )

    assert figures == []


def test_risk_analysis_graph_preserves_complete_monthly_metrics() -> None:
    dates = pd.date_range("2024-01-02", periods=40, freq="D")
    report = pd.DataFrame(
        {
            "return": [((day % 7) - 3) / 1000 for day in range(len(dates))],
            "bench": [((day % 5) - 2) / 2000 for day in range(len(dates))],
            "cost": 0.0001,
            "turnover": 0.1,
        },
        index=pd.DatetimeIndex(dates, name="date"),
    )

    figures = risk_analysis_module.risk_analysis_graph(
        analysis_df=None,
        report_normal_df=report,
        show_notebook=False,
    )

    assert [figure.layout.title.text for figure in figures] == [
        "annualized_return",
        "max_drawdown",
        "information_ratio",
        "std",
    ]
