# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import unittest
import platform
import mlflow
import os
import time
from pathlib import Path
import shutil
import tempfile
from unittest.mock import patch

from sqlalchemy.pool import NullPool

from qlib.config import get_default_mlflow_uri


class MLflowTest(unittest.TestCase):
    def setUp(self) -> None:
        self._poolclass = patch.dict(os.environ, {"MLFLOW_SQLALCHEMYSTORE_POOLCLASS": "NullPool"})
        self._poolclass.start()
        self.addCleanup(self._poolclass.stop)
        self.tmp_path = Path(tempfile.mkdtemp(prefix="qlib-mlruns-client-"))
        self.addCleanup(shutil.rmtree, self.tmp_path)
        self.uri = get_default_mlflow_uri(self.tmp_path / "mlflow.db")
        # Exclude one-time database initialization from the client creation benchmark.
        client = mlflow.tracking.MlflowClient(tracking_uri=self.uri)
        self.assertIsInstance(client._tracking_client.store.engine.pool, NullPool)

    def test_creating_client(self):
        """
        Please refer to qlib/workflow/expm.py:MLflowExpManager._client
        we don't cache _client (this is helpful to reduce maintainance work when MLflowExpManager's uri is chagned)

        This implementation is based on the assumption creating a client is fast
        """
        start = time.time()
        for i in range(10):
            _ = mlflow.tracking.MlflowClient(tracking_uri=self.uri)
        end = time.time()
        elapsed = end - start
        if platform.system() == "Linux":
            self.assertLess(elapsed, 1e-2)  # it can be done in less than 10ms
        else:
            self.assertLess(elapsed, 2e-2)
        print(elapsed)


if __name__ == "__main__":
    unittest.main()
