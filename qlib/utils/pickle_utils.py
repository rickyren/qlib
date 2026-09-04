# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Secure pickle utilities to prevent arbitrary code execution through deserialization.

This module provides a secure alternative to pickle.load() and pickle.loads()
that restricts deserialization to a whitelist of safe classes.
"""

import io
import pickle
from typing import Any, BinaryIO, Set, Tuple

_PYARROW_STRING_TYPE_ALIASES = {"string", "large_string"}
# Serialized Arrow state is untrusted.  Exact type checks intentionally reject
# bool-as-int coercion and subclasses with custom behavior.
# pylint: disable=unidiomatic-typecheck


def _restore_pyarrow_type_for_alias(alias: str):
    """Restore only the Arrow string types needed by pandas string arrays."""
    if type(alias) is not str or alias not in _PYARROW_STRING_TYPE_ALIASES:
        raise pickle.UnpicklingError("Only exact Arrow string type aliases are allowed")

    try:
        import pyarrow as pa  # pylint: disable=import-outside-toplevel

        data_type = pa.type_for_alias(alias)
        if data_type not in (pa.string(), pa.large_string()):
            raise ValueError(f"Unexpected Arrow type for alias {alias!r}: {data_type}")
        return data_type
    except pickle.UnpicklingError:
        raise
    except Exception as error:
        raise pickle.UnpicklingError("Failed to restore an Arrow string type") from error


def _restore_pyarrow_buffer(value):
    """Restore a CPU Arrow buffer without accepting arbitrary buffer providers."""
    if type(value) not in (bytes, bytearray):
        raise pickle.UnpicklingError("Arrow buffers must be restored from exact bytes or bytearray values")

    try:
        import pyarrow as pa  # pylint: disable=import-outside-toplevel

        # ``pa.py_buffer(bytearray)`` is zero-copy.  Copy mutable input so a
        # separately memoized bytearray in the pickle cannot invalidate a
        # previously validated Arrow array after this function returns.
        return pa.py_buffer(bytes(value))
    except Exception as error:
        raise pickle.UnpicklingError("Failed to restore an Arrow buffer") from error


def _count_set_bits(buffer, offset: int, length: int) -> int:
    """Count a bounded bitmap range without allocating an array per bit."""
    import numpy as np  # pylint: disable=import-outside-toplevel

    if length == 0:
        return 0

    values = np.frombuffer(buffer, dtype=np.uint8)
    first_byte = offset // 8
    last_byte = (offset + length - 1) // 8
    first_shift = offset % 8
    end_shift = (offset + length) % 8
    popcount = np.fromiter((bin(value).count("1") for value in range(256)), dtype=np.uint8, count=256)

    if first_byte == last_byte:
        mask = ((1 << length) - 1) << first_shift
        return int(popcount[int(values[first_byte]) & mask])

    first_mask = (0xFF << first_shift) & 0xFF
    last_mask = (1 << end_shift) - 1 if end_shift else 0xFF
    total = int(popcount[int(values[first_byte]) & first_mask])
    total += int(popcount[int(values[last_byte]) & last_mask])
    if last_byte > first_byte + 1:
        histogram = np.bincount(values[first_byte + 1 : last_byte], minlength=256)
        total += int(histogram.dot(popcount))
    return total


def _validate_pyarrow_string_buffers(data_type, length: int, null_count: int, offset: int, buffers) -> int:
    """Validate Arrow string buffers before passing them to Arrow's native restorer."""
    import numpy as np  # pylint: disable=import-outside-toplevel
    import pyarrow as pa  # pylint: disable=import-outside-toplevel

    # Arrow string and large-string arrays have exactly three buffers:
    # validity, offsets, and UTF-8 data.  Do not rely on newer Arrow metadata
    # attributes here because Qlib still supports older optional PyArrow builds.
    if type(buffers) is not list or len(buffers) != 3:
        raise pickle.UnpicklingError("Arrow string buffers must be an exact, complete list")
    if any(buffer is not None and type(buffer) is not pa.Buffer for buffer in buffers):
        raise pickle.UnpicklingError("Arrow string buffers must contain only Arrow buffers or None")

    validity_buffer, offsets_buffer, data_buffer = buffers
    required_validity_bits = offset + length
    if validity_buffer is None:
        actual_null_count = 0
        if null_count not in (-1, 0):
            raise pickle.UnpicklingError("Arrow null_count requires a validity buffer")
    else:
        if validity_buffer.size * 8 < required_validity_bits:
            raise pickle.UnpicklingError("Arrow validity buffer is too short")
        actual_null_count = length - _count_set_bits(validity_buffer, offset, length)
        if null_count not in (-1, actual_null_count):
            raise pickle.UnpicklingError("Arrow validity buffer does not match null_count")

    if offsets_buffer is None or data_buffer is None:
        raise pickle.UnpicklingError("Arrow string offset and data buffers are required")

    offset_dtype = np.int32 if data_type == pa.string() else np.int64
    required_offsets = offset + length + 1
    if offsets_buffer.size < required_offsets * np.dtype(offset_dtype).itemsize:
        raise pickle.UnpicklingError("Arrow string offsets buffer is too short")
    offsets = np.frombuffer(offsets_buffer, dtype=offset_dtype, count=required_offsets)
    selected_offsets = offsets[offset:required_offsets]
    if np.any(selected_offsets < 0) or np.any(selected_offsets[1:] < selected_offsets[:-1]):
        raise pickle.UnpicklingError("Arrow string offsets must be non-negative and monotonic")
    if int(selected_offsets[-1]) > data_buffer.size:
        raise pickle.UnpicklingError("Arrow string offsets exceed the data buffer")
    return actual_null_count


def _restore_pyarrow_array(state):
    """Restore a validated Arrow string array without exposing raw Arrow globals."""
    try:
        import pyarrow as pa  # pylint: disable=import-outside-toplevel

        if type(state) is not tuple or len(state) != 7:
            raise pickle.UnpicklingError("Arrow array state must be an exact seven-item tuple")

        data_type, length, null_count, offset, buffers, children, dictionary = state
        if type(data_type) is not pa.DataType or data_type not in (
            pa.string(),
            pa.large_string(),
        ):
            raise pickle.UnpicklingError("Only exact Arrow string data types are allowed")
        if type(length) is not int or length < 0:
            raise pickle.UnpicklingError("Arrow array length must be a non-negative exact integer")
        if type(null_count) is not int or null_count < -1 or null_count > length:
            raise pickle.UnpicklingError("Arrow array null_count must be -1 or an exact integer within array bounds")
        if type(offset) is not int or offset < 0:
            raise pickle.UnpicklingError("Arrow array offset must be a non-negative exact integer")
        if type(children) is not list or children:
            raise pickle.UnpicklingError("Arrow string arrays must have an empty exact children list")
        if dictionary is not None:
            raise pickle.UnpicklingError("Arrow string arrays cannot contain a dictionary")

        actual_null_count = _validate_pyarrow_string_buffers(data_type, length, null_count, offset, buffers)
        array = pa.lib._restore_array(state)  # pylint: disable=c-extension-no-member
        if not isinstance(array, pa.Array):
            raise pickle.UnpicklingError("Arrow restorer did not return an array")
        if (
            array.type != data_type
            or len(array) != length
            or array.null_count != actual_null_count
            or array.offset != offset
        ):
            raise pickle.UnpicklingError("Restored Arrow array metadata does not match its serialized state")
        array.validate(full=True)
        return array
    except pickle.UnpicklingError:
        raise
    except Exception as error:
        raise pickle.UnpicklingError("Failed to restore a validated Arrow string array") from error


_PYARROW_SAFE_RESTORERS = {
    ("pyarrow.lib", "_restore_array"): _restore_pyarrow_array,
    ("pyarrow.lib", "type_for_alias"): _restore_pyarrow_type_for_alias,
    ("pyarrow.lib", "py_buffer"): _restore_pyarrow_buffer,
}

# Whitelist of safe classes that are allowed to be unpickled
# These are common data types used in qlib that should be safe to deserialize
SAFE_PICKLE_CLASSES: Set[Tuple[str, str]] = {
    # python builtins
    ("builtins", "slice"),
    ("builtins", "range"),
    ("builtins", "dict"),
    ("builtins", "list"),
    ("builtins", "tuple"),
    ("builtins", "set"),
    ("builtins", "frozenset"),
    ("builtins", "bytearray"),
    ("builtins", "bytes"),
    ("builtins", "str"),
    ("builtins", "int"),
    ("builtins", "float"),
    ("builtins", "bool"),
    ("builtins", "complex"),
    ("builtins", "type"),
    ("builtins", "property"),
    # common utility classes
    ("datetime", "datetime"),
    ("datetime", "date"),
    ("datetime", "time"),
    ("datetime", "timedelta"),
    ("datetime", "timezone"),
    ("decimal", "Decimal"),
    ("collections", "OrderedDict"),
    ("collections", "defaultdict"),
    ("collections", "Counter"),
    ("collections", "namedtuple"),
    ("enum", "Enum"),
    ("pathlib", "Path"),
    ("pathlib", "PosixPath"),
    ("pathlib", "WindowsPath"),
    ("qlib.data.dataset.handler", "DataHandler"),
    ("qlib.data.dataset.handler", "DataHandlerLP"),
    ("qlib.data.dataset.loader", "StaticDataLoader"),
}


TRUSTED_MODULE_PREFIXES = (
    "pandas",
    "numpy",
)


class RestrictedUnpickler(pickle.Unpickler):
    """Custom unpickler that only allows safe classes to be deserialized.

    This prevents arbitrary code execution through malicious pickle files by
    restricting deserialization to a whitelist of safe classes.

    Example:
        >>> with open("data.pkl", "rb") as f:
        ...     data = RestrictedUnpickler(f).load()
    """

    def find_class(self, module: str, name: str):
        """Override find_class to restrict allowed classes.

        Args:
            module: Module name of the class
            name: Class name

        Returns:
            The class object if it's in the whitelist

        Raises:
            pickle.UnpicklingError: If the class is not in the whitelist
        """
        # These exact Arrow globals must always resolve to Qlib-owned validating
        # wrappers, even if a caller later adds them to the generic allowlist.
        pyarrow_restorer = _PYARROW_SAFE_RESTORERS.get((module, name))
        if pyarrow_restorer is not None:
            return pyarrow_restorer

        if module.startswith(TRUSTED_MODULE_PREFIXES):
            return super().find_class(module, name)

        # 2. explicit whitelist (qlib internal)
        if (module, name) in SAFE_PICKLE_CLASSES:
            return super().find_class(module, name)

        raise pickle.UnpicklingError(
            f"Forbidden class: {module}.{name}. "
            f"Only whitelisted classes are allowed for security reasons. "
            f"This is to prevent arbitrary code execution through pickle deserialization."
        )


def restricted_pickle_load(file: BinaryIO) -> Any:
    """Safely load a pickle file with restricted classes.

    This is a drop-in replacement for pickle.load() that prevents
    arbitrary code execution by only allowing whitelisted classes.

    Args:
        file: An opened file object in binary mode

    Returns:
        The unpickled Python object

    Raises:
        pickle.UnpicklingError: If the pickle contains forbidden classes

    Example:
        >>> with open("data.pkl", "rb") as f:
        ...     data = restricted_pickle_load(f)
    """
    return RestrictedUnpickler(file).load()


def restricted_pickle_loads(data: bytes) -> Any:
    """Safely load a pickle from bytes with restricted classes.

    This is a drop-in replacement for pickle.loads() that prevents
    arbitrary code execution by only allowing whitelisted classes.

    Args:
        data: Bytes object containing pickled data

    Returns:
        The unpickled Python object

    Raises:
        pickle.UnpicklingError: If the pickle contains forbidden classes

    Example:
        >>> data = b'\\x80\\x04\\x95...'
        >>> obj = restricted_pickle_loads(data)
    """
    file_like = io.BytesIO(data)
    return RestrictedUnpickler(file_like).load()


def add_safe_class(module: str, name: str) -> None:
    """Add a class to the whitelist of safe classes for unpickling.

    Use this function to extend the whitelist if your code needs to deserialize
    additional classes. However, be very careful when adding classes, as this
    could potentially introduce security vulnerabilities.

    Args:
        module: Module name of the class (e.g., 'my_package.my_module')
        name: Class name (e.g., 'MyClass')

    Warning:
        Only add classes that you fully control and trust. Adding arbitrary
        classes from external packages could introduce security risks.

    Example:
        >>> add_safe_class('my_package.models', 'CustomModel')
    """
    SAFE_PICKLE_CLASSES.add((module, name))


def get_safe_classes() -> Set[Tuple[str, str]]:
    """Get a copy of the current whitelist of safe classes.

    Returns:
        A set of (module, name) tuples representing allowed classes
    """
    return SAFE_PICKLE_CLASSES.copy()
