"""Manual acceptance tests against disposable Gravitino and Delta storage.

The fixture owner must provide RAY_GRAVITINO_DELTA_TEST_ settings URI, METALAKE,
TABLE, LOCATION, STORAGE_OPTIONS, ENDPOINT, TABLE_ROOT, ACCESS, SECRET,
LIMITED_ACCESS, LIMITED_SECRET, DENIED_ACCESS and DENIED_SECRET. Clean up the
provisioned resources after the run. The registered table must contain at least
four data files. The limited identity can read the Delta log but not data files;
the denied identity cannot read the Delta log.
"""

import base64
import json
import os

import pyarrow.fs as pafs
import pytest
from deltalake import DeltaTable
from deltalake.exceptions import DeltaError

import ray
from ray.data.catalog import GravitinoCatalog
from ray.data.tests.conftest import *  # noqa: F401,F403
from ray.exceptions import RayTaskError

_PREFIX = "RAY_GRAVITINO_DELTA_TEST_"


def _required(name):
    value = os.environ.get(_PREFIX + name)
    if not value:
        pytest.fail(f"Set {_PREFIX + name} for the Gravitino Delta integration tests.")
    return value


def _headers(user):
    value = base64.b64encode(f"{user}:".encode()).decode()
    return {"Authorization": f"Basic {value}"}


def _catalog(user="adminUser"):
    return GravitinoCatalog(
        uri=_required("URI"),
        metalake=_required("METALAKE"),
        main_api_headers=_headers(user),
    )


def _storage():
    return json.loads(_required("STORAGE_OPTIONS"))


def _filesystem(limited=False):
    access = _required("LIMITED_ACCESS" if limited else "ACCESS")
    secret = _required("LIMITED_SECRET" if limited else "SECRET")
    fs = pafs.S3FileSystem(
        access_key=access,
        secret_key=secret,
        endpoint_override=_required("ENDPOINT"),
        scheme="http",
        region="us-east-1",
    )
    return pafs.SubTreeFileSystem(_required("TABLE_ROOT"), fs)


def test_delta_multitask_read(ray_start_regular_shared):
    expected = DeltaTable(
        _required("LOCATION"), storage_options=_storage()
    ).to_pyarrow_table(filesystem=_filesystem())
    ds = ray.data.read_delta(
        _required("TABLE"),
        catalog=_catalog(),
        storage_options=_storage(),
        filesystem=_filesystem(),
        override_num_blocks=4,
    ).materialize()
    expected_ids = sorted(expected.column("id").to_pylist())
    assert expected_ids == [0, 1, 2, 3]
    assert sorted(row["id"] for row in ds.take_all()) == expected_ids
    assert ds._raw_stats().extra_metrics["num_tasks_finished"] >= 2


def test_delta_metadata_access_denied(ray_start_regular_shared):
    with pytest.raises(PermissionError):
        ray.data.read_delta(_required("TABLE"), catalog=_catalog("read_denied"))


def test_delta_missing_table(ray_start_regular_shared):
    identifier = _required("TABLE").rsplit(".", 1)[0] + ".missing_table"
    with pytest.raises(FileNotFoundError):
        ray.data.read_delta(identifier, catalog=_catalog())


def test_delta_log_access_denied(ray_start_regular_shared):
    options = _storage()
    options["AWS_ACCESS_KEY_ID"] = _required("DENIED_ACCESS")
    options["AWS_SECRET_ACCESS_KEY"] = _required("DENIED_SECRET")
    with pytest.raises(
        (DeltaError, OSError), match="(?i)(access.?denied|forbidden|403)"
    ):
        ray.data.read_delta(
            _required("TABLE"), catalog=_catalog(), storage_options=options
        )


def test_delta_worker_storage_denied(ray_start_regular_shared):
    with pytest.raises(RayTaskError, match="(?i)(access.?denied|forbidden|403)"):
        ds = ray.data.read_delta(
            _required("TABLE"),
            catalog=_catalog(),
            storage_options=_storage(),
            filesystem=_filesystem(limited=True),
            override_num_blocks=4,
        )
        ds.materialize()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
