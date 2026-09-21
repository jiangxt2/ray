import pyarrow as pa
import pytest

import ray
import ray.data.read_api as read_api
from ray.data._internal.datasource.hive_hs2_client import HiveConnectionOptions
from ray.data._internal.datasource_v2.hive_datasource import HiveDatasourceV2
from ray.data._internal.datasource_v2.logical_optimizers import (
    SupportsColumnPruning,
    SupportsFilterPushdown,
    SupportsLimitPushdown,
)
from ray.data._internal.datasource_v2.readers.hive_reader import (
    DEFAULT_HIVE_TARGET_BATCH_BYTES,
    DEFAULT_HIVE_TASK_MEMORY_BYTES,
)
from ray.data._internal.logical.operators import ListFiles, ReadFiles
from ray.data.context import DataContext


@pytest.fixture(autouse=True)
def disable_auto_init_for_planning_tests(monkeypatch):
    # These tests inspect the plan and never execute a Ray task. Calling the
    # undecorated planner also keeps them independent of host Ray startup hooks.
    wrapped = read_api._read_datasource_v2
    monkeypatch.setattr(read_api, "_read_datasource_v2", wrapped.__wrapped__)


@pytest.fixture
def restore_context():
    ctx = DataContext.get_current()
    original = ctx.use_datasource_v2
    try:
        yield ctx
    finally:
        ctx.use_datasource_v2 = original


def test_hive_datasource_v2_declares_database_reader_contract(monkeypatch):
    schema = pa.schema([("id", pa.int64()), ("name", pa.string())])
    monkeypatch.setattr(
        HiveDatasourceV2,
        "_resolve_table_schema",
        lambda self: schema,
    )
    datasource = HiveDatasourceV2(
        host="hive-server",
        port=10000,
        connection_options=HiveConnectionOptions(
            host="hive-server", port=10000, auth_mechanism="NOSASL"
        ),
        table="default.users",
    )

    assert datasource.category.value == "database"
    assert datasource.filesystem is None
    assert datasource.paths == ["hive://planned-read"]
    assert datasource.schema_needs_file_sample is False
    assert datasource.infer_schema(None) == schema

    scanner = datasource.create_scanner(schema, filesystem=None)
    assert isinstance(scanner, SupportsColumnPruning)
    assert isinstance(scanner, SupportsFilterPushdown)
    assert isinstance(scanner, SupportsLimitPushdown)


def test_read_hive_builds_single_v2_work_unit_and_disables_retries(
    monkeypatch, restore_context
):
    schema = pa.schema([("id", pa.int64()), ("name", pa.string())])
    monkeypatch.setattr(
        HiveDatasourceV2,
        "_resolve_table_schema",
        lambda self: schema,
    )
    restore_context.use_datasource_v2 = True

    ds = ray.data.read_hive("default.users", host="hive-server")

    read_op = ds._logical_plan.dag
    assert isinstance(read_op, ReadFiles)
    assert isinstance(read_op.input_dependencies[0], ListFiles)
    assert read_op.input_dependencies[0].paths == ["hive://planned-read"]
    assert read_op.ray_remote_args["max_retries"] == 0
    assert read_op.ray_remote_args["memory"] >= DEFAULT_HIVE_TASK_MEMORY_BYTES
    assert ds.schema().names == ["id", "name"]
    assert ds.context.target_max_block_size == DEFAULT_HIVE_TARGET_BATCH_BYTES


def test_raw_sql_requires_explicit_schema_and_does_not_probe_hs2(monkeypatch):
    def fail_if_connected(**kwargs):
        raise AssertionError("raw SQL schema must not be probed with HS2")

    monkeypatch.setattr(
        "ray.data._internal.datasource.hive_hs2_client.connect_hiveserver2",
        fail_if_connected,
    )
    schema = pa.schema([("value", pa.int32())])

    ds = ray.data.read_hive(
        host="hive-server",
        query="SELECT 1 AS value",
        schema=schema,
    )

    assert ds.schema().names == ["value"]
    assert isinstance(ds._logical_plan.dag, ReadFiles)

    with pytest.raises(ValueError, match="explicit pyarrow.Schema"):
        ray.data.read_hive(host="hive-server", query="SELECT 1 AS value")
