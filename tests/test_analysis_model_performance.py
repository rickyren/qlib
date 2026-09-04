# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import pytest

from qlib.contrib.report.analysis_model import analysis_model_performance


def _pred_label(start_year: int, end_year: int) -> pd.DataFrame:
    dates = []
    for year in range(start_year, end_year + 1):
        dates.extend(pd.to_datetime([f"{year}-01-03", f"{year}-06-03", f"{year}-12-29"]))
    instruments = [f"S{instrument:02d}" for instrument in range(10)]
    index = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])
    rng = np.random.default_rng(7)
    score = rng.normal(size=len(index))
    label = 0.3 * score + rng.normal(size=len(index))
    return pd.DataFrame({"score": score, "label": label}, index=index)


@pytest.mark.parametrize("start_year,end_year", [(2024, 2024), (2023, 2024)])
def test_full_year_month_index(start_year: int, end_year: int) -> None:
    actual = analysis_model_performance._full_year_month_index(start_year, end_year)
    expected = pd.MultiIndex.from_tuples(
        [(str(year), f"{month:02d}") for year in range(start_year, end_year + 1) for month in range(1, 13)],
        names=["year", "month"],
    )

    pd.testing.assert_index_equal(actual, expected, exact=True)


@pytest.mark.parametrize("start_year,end_year", [(2024, 2024), (2023, 2024)])
def test_pred_ic_fills_every_month_without_offset_alias(start_year: int, end_year: int) -> None:
    figures = analysis_model_performance._pred_ic(_pred_label(start_year, end_year))

    assert len(figures) == 3
    heatmap = figures[1].data[0]
    assert list(heatmap.x) == [f"{month:02d}" for month in range(1, 13)]
    assert list(heatmap.y) == [str(year) for year in range(start_year, end_year + 1)]
    assert np.asarray(heatmap.z).shape == (end_year - start_year + 1, 12)


def test_model_performance_graph_executes_pred_ic_path() -> None:
    figures = analysis_model_performance.model_performance_graph(
        _pred_label(2023, 2024), graph_names=["pred_ic"], show_notebook=False
    )

    assert len(figures) == 3
    assert figures[1].layout.title.text == "Monthly IC"
