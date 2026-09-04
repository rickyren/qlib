import io
import os
import pickle
import struct

import pandas as pd
import pytest

from qlib.data.dataset.handler import DataHandlerLP
from qlib.utils.pickle_utils import (
    RestrictedUnpickler,
    add_safe_class,
    restricted_pickle_loads,
)


class _ReducePayload:
    def __init__(self, target, args):
        self.target = target
        self.args = args

    def __reduce__(self):
        return self.target, self.args


class _SystemPayload:
    def __reduce__(self):
        return os.system, ("echo restricted-unpickler-must-not-run",)


class _ArrowBufferPayload:
    def __init__(self, target, value):
        self.target = target
        self.value = value

    def __reduce__(self):
        return self.target, (self.value,)


def _pickle_reduce(target, *args):
    return pickle.dumps(_ReducePayload(target, args), protocol=4)


def _valid_large_string_state(pa):
    return (
        pa.large_string(),
        1,
        0,
        0,
        [None, pa.py_buffer(struct.pack("<qq", 0, 1)), pa.py_buffer(b"a")],
        [],
        None,
    )


def _supported_arrow_string_array(values):
    try:
        array = pd.array(values, dtype="string[pyarrow]")
    except (ImportError, TypeError, ValueError) as error:
        pytest.skip(f"This pandas/PyArrow combination has no Arrow string dtype: {error}")
    if getattr(array.dtype, "storage", None) != "pyarrow":
        pytest.skip("This pandas version did not create an Arrow-backed string array")
    return array


def test_restricted_unpickler_rejects_code_execution():
    with pytest.raises(pickle.UnpicklingError, match="Forbidden class"):
        restricted_pickle_loads(pickle.dumps(_SystemPayload(), protocol=4))


def test_restricted_unpickler_rejects_arbitrary_pyarrow_global():
    pytest.importorskip("pyarrow")

    with pytest.raises(pickle.UnpicklingError, match="Forbidden class"):
        RestrictedUnpickler(io.BytesIO()).find_class("pyarrow.lib", "_restore_table")


def test_exact_pyarrow_restorer_cannot_be_bypassed_by_generic_allowlist():
    pytest.importorskip("pyarrow")
    key = ("pyarrow.lib", "_restore_array")
    from qlib.utils import pickle_utils

    was_safe = key in pickle_utils.SAFE_PICKLE_CLASSES
    try:
        add_safe_class(*key)
        restorer = RestrictedUnpickler(io.BytesIO()).find_class(*key)
        assert restorer.__module__ == "qlib.utils.pickle_utils"
        assert restorer.__name__ == "_restore_pyarrow_array"
    finally:
        if not was_safe:
            pickle_utils.SAFE_PICKLE_CLASSES.remove(key)


def test_pandas_arrow_string_data_handler_and_multiindex_round_trip():
    pytest.importorskip("pyarrow")
    instruments = _supported_arrow_string_array(["SH600000", None, "SH600001"])
    index = pd.MultiIndex.from_arrays(
        [pd.date_range("2024-01-01", periods=3), instruments],
        names=["datetime", "instrument"],
    )
    frame = pd.DataFrame(
        {
            "feature": _supported_arrow_string_array(["alpha", None, "beta"]),
            "value": [1.0, float("nan"), 3.0],
        },
        index=index,
    )

    handler = DataHandlerLP.from_df(frame)
    handler.config(dump_all=True)
    restored = restricted_pickle_loads(pickle.dumps(handler, protocol=4))

    pd.testing.assert_frame_equal(restored._data, frame)
    assert restored._infer is restored._data
    assert restored._learn is restored._data
    assert "_data" not in restored.data_loader.__dict__
    assert getattr(restored._data["feature"].dtype, "storage", None) == "pyarrow"


def test_mutable_arrow_buffer_is_copied_before_array_validation():
    pa = pytest.importorskip("pyarrow")
    offsets = bytearray(struct.pack("<qq", 0, 1))
    state = (
        pa.large_string(),
        1,
        0,
        0,
        [None, _ArrowBufferPayload(pa.lib.py_buffer, offsets), _ArrowBufferPayload(pa.lib.py_buffer, bytearray(b"a"))],
        [],
        None,
    )
    payload = (_ReducePayload(pa.lib._restore_array, (state,)), offsets)

    restored, restored_offsets = restricted_pickle_loads(pickle.dumps(payload, protocol=5))
    restored_offsets[8:] = struct.pack("<q", 999)

    restored.validate(full=True)
    assert restored.to_pylist() == ["a"]


@pytest.mark.parametrize("type_factory", ["string", "large_string"])
def test_sliced_pyarrow_string_array_with_unknown_null_count_round_trip(type_factory):
    pa = pytest.importorskip("pyarrow")
    array = pa.array(["skip", "alpha", None, "beta"], type=getattr(pa, type_factory)()).slice(1, 3)
    assert array.__reduce__()[1][0][2] == -1

    restored = restricted_pickle_loads(pickle.dumps(array, protocol=4))

    assert restored.equals(array)
    assert restored.offset == array.offset


@pytest.mark.parametrize(
    "mutate_state",
    [
        lambda state: list(state),
        lambda state: state[:-1],
        lambda state: (state[0], True, *state[2:]),
        lambda state: (state[0], state[1], True, *state[3:]),
        lambda state: (state[0], state[1], 2, *state[3:]),
        lambda state: (state[0], state[1], -2, *state[3:]),
        lambda state: (*state[:3], True, *state[4:]),
        lambda state: (*state[:3], -1, *state[4:]),
        lambda state: (*state[:4], tuple(state[4]), *state[5:]),
        lambda state: (*state[:4], [None, state[4][1]], *state[5:]),
        lambda state: (*state[:4], [None, state[4][1], b"a"], *state[5:]),
        lambda state: (*state[:5], [state[0]], state[6]),
        lambda state: (*state[:6], state[0]),
        lambda state: (
            *state[:4],
            [None, state[4][1], state[4][2].slice(0, 0)],
            *state[5:],
        ),
        lambda state: (
            *state[:4],
            [None, state[4][1].slice(0, 8), state[4][2]],
            *state[5:],
        ),
        lambda state: (
            state[0],
            state[1],
            1,
            state[3],
            [None, *state[4][1:]],
            *state[5:],
        ),
    ],
    ids=[
        "outer-list",
        "wrong-tuple-length",
        "boolean-length",
        "boolean-null-count",
        "null-count-out-of-bounds",
        "invalid-unknown-null-count",
        "boolean-offset",
        "negative-offset",
        "buffer-tuple",
        "missing-buffer",
        "raw-bytes-buffer",
        "children",
        "dictionary",
        "offset-exceeds-data",
        "offset-buffer-too-short",
        "null-count-without-validity",
    ],
)
def test_malformed_arrow_array_state_fails_closed(mutate_state):
    pa = pytest.importorskip("pyarrow")
    state = mutate_state(_valid_large_string_state(pa))

    with pytest.raises(pickle.UnpicklingError):
        restricted_pickle_loads(_pickle_reduce(pa.lib._restore_array, state))


@pytest.mark.parametrize(
    ("target_name", "argument"),
    [("type_for_alias", 1), ("type_for_alias", "int64"), ("py_buffer", "not-bytes")],
)
def test_pyarrow_helper_arguments_fail_closed(target_name, argument):
    pa = pytest.importorskip("pyarrow")
    target = getattr(pa.lib, target_name)

    with pytest.raises(pickle.UnpicklingError):
        restricted_pickle_loads(_pickle_reduce(target, argument))
