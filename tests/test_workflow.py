# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import unittest
import os
from pathlib import Path
import shutil
import tempfile
from unittest.mock import patch

import mlflow
from sqlalchemy.pool import NullPool

from qlib.workflow import R
from qlib.config import get_default_mlflow_uri
from qlib.tests import TestAutoData


class WorkflowTest(TestAutoData):
    def setUp(self) -> None:
        self._poolclass = patch.dict(os.environ, {"MLFLOW_SQLALCHEMYSTORE_POOLCLASS": "NullPool"})
        self._poolclass.start()
        self.addCleanup(self._poolclass.stop)
        self.tmp_path = Path(tempfile.mkdtemp(prefix="qlib-mlruns-workflow-"))
        self.addCleanup(shutil.rmtree, self.tmp_path)
        self.uri = get_default_mlflow_uri(self.tmp_path / "mlflow.db")

    def test_get_local_dir(self):
        """ """
        with R.start(uri=self.uri):
            R.save_objects(local_dir_smoke="ok")

        with R.uri_context(uri=self.uri):
            resume_recorder = R.get_recorder()
            local_dir = Path(resume_recorder.get_local_dir())
            self.assertIn(self.tmp_path.resolve(), local_dir.parents)

        self.assertIsNone(mlflow.active_run())
        client = mlflow.tracking.MlflowClient(tracking_uri=self.uri)
        self.assertIsInstance(client._tracking_client.store.engine.pool, NullPool)


if __name__ == "__main__":
    unittest.main()
