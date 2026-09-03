# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import unittest
from pathlib import Path
import shutil

from qlib.workflow import R
from qlib.config import get_default_mlflow_uri
from qlib.tests import TestAutoData


class WorkflowTest(TestAutoData):
    TMP_PATH = Path("./.mlruns_workflow_tmp")
    URI = get_default_mlflow_uri(TMP_PATH / "mlflow.db")

    def tearDown(self) -> None:
        if self.TMP_PATH.exists():
            shutil.rmtree(self.TMP_PATH)

    def test_get_local_dir(self):
        """ """
        self.TMP_PATH.mkdir(parents=True, exist_ok=True)

        with R.start(uri=self.URI):
            R.save_objects(local_dir_smoke="ok")

        with R.uri_context(uri=self.URI):
            resume_recorder = R.get_recorder()
            local_dir = Path(resume_recorder.get_local_dir())
            self.assertIn(self.TMP_PATH.resolve(), local_dir.parents)


if __name__ == "__main__":
    unittest.main()
