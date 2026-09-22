import asyncio
import collections
import inspect
import logging
from dataclasses import dataclass
from threading import Event, Thread
from types import GeneratorType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    TypeVar,
)

if TYPE_CHECKING:
    from ray.data.expressions import _CallableClassSpec

import numpy as np
import pandas as pd
import pyarrow as pa

import ray
from ray._common.utils import env_integer, get_or_create_event_loop
from ray.data._internal.compute import ActorPoolStrategy, ComputeStrategy, get_compute
from ray.data._internal.execution.bundle_queue import ExactMultipleSize, RebundleQueue
from ray.data._internal.execution.interfaces import PhysicalOperator
from ray.data._internal.execution.interfaces.task_context import TaskContext
from ray.data._internal.execution.operators.map_operator import MapOperator
from ray.data._internal.execution.operators.map_transformer import (
    BatchMapTransformFn,
    BlockMapTransformFn,
    MapTransformCallable,
    MapTransformer,
    Row,
    RowMapTransformFn,
)
from ray.data._internal.execution.util import make_callable_class_single_threaded
from ray.data._internal.logical.operators import (
    AbstractUDFMap,
    Filter,
    FlatMap,
    MapBatches,
    MapRows,
    Project,
    StreamingRepartition,
)
from ray.data._internal.numpy_support import _is_valid_column_values
from ray.data._internal.output_buffer import OutputBlockSizeOption
from ray.data._internal.util import _truncated_repr
from ray.data.block import (
    Block,
    BlockAccessor,
    CallableClass,
    DataBatch,
    UserDefinedFunction,
    _is_cudf_dataframe,
)
from ray.data.context import DataContext
from ray.data.exceptions import UserCodeException
from ray.util.rpdb import _is_ray_debugger_post_mortem_enabled

logger = logging.getLogger(__name__)


# Controls default max-concurrency setting for async row-based UDFs
DEFAULT_ASYNC_ROW_UDF_MAX_CONCURRENCY = env_integer(
    "RAY_DATA_DEFAULT_ASYNC_ROW_UDF_MAX_CONCURRENCY", 16
)

# Controls default max-concurrency setting for async batch-based UDFs
DEFAULT_ASYNC_BATCH_UDF_MAX_CONCURRENCY = env_integer(
    "RAY_DATA_DEFAULT_ASYNC_BATCH_UDF_MAX_CONCURRENCY", 4
)


@dataclass
class UDFSpec:
    """Specification for a callable class UDF to be instantiated in an actor.

    Attributes:
        spec: The callable class specification (contains class and constructor args)
        instantiation_class: The class to instantiate (may be wrapped, e.g., for concurrency)
    """

    spec: "_CallableClassSpec"
    instantiation_class: type


class _MapActorContext:
    def __init__(
        self,
        is_async: bool = False,
        udf_instances: Optional[Dict[int, Any]] = None,
    ):
        """Initialize the map actor context.

        Args:
            is_async: Whether any UDF is async
            udf_instances: Dict mapping UDF class ID to instantiated instance
        """
        self.is_async = is_async
        self.udf_map_asyncio_loop = None
        self.udf_map_asyncio_thread = None
        self.udf_instances = udf_instances or {}

        if is_async:
            self._init_async()

    def _init_async(self):
        # Only used for callable class with async generator `__call__` method.
        loop = get_or_create_event_loop()

        def run_loop():
            asyncio.set_event_loop(loop)
            loop.run_forever()

        thread = Thread(target=run_loop, daemon=True)
        thread.start()
        self.udf_map_asyncio_loop = loop
        self.udf_map_asyncio_thread = thread


def plan_project_op(
    op: Project,
    physical_children: List[PhysicalOperator],
    data_context: DataContext,
) -> MapOperator:
    assert len(physical_children) == 1
    input_physical_dag = physical_children[0]

    # Extract expressions before defining the closure to prevent cloudpickle from
    # serializing the entire op object (which may contain references to non-serializable
    # datasources with weak references, e.g., PyIceberg tables)
    projection_exprs = op.exprs
    common_sub_exprs = op.get_common_sub_exprs()

    compute = get_compute(op.compute)

    # Create init_fn to initialize all callable class UDFs at actor startup
    from ray.data.util.expression_utils import (
        _create_callable_class_udf_init_fn,
    )

    init_fn = _create_callable_class_udf_init_fn(op.get_all_exprs())

    def _project_block(block: Block) -> Block:
        try:
            from ray.data._internal.planner.plan_expression.expression_evaluator import (
                eval_projection,
            )

            return eval_projection(
                projection_exprs,
                block,
                common_sub_exprs=common_sub_exprs,
            )
        except Exception as e:
            _try_wrap_udf_exception(e)

    map_transformer = MapTransformer(
        [
            BlockMapTransformFn(
                _generate_transform_fn_for_map_block(_project_block),
                disable_block_shaping=(len(op.exprs) == 0),
            )
        ],
        init_fn=init_fn,
    )
    return MapOperator.create(
        map_transformer,
        input_physical_dag,
        data_context,
        name=op.name,
        compute_strategy=compute,
        ray_remote_args=op.ray_remote_args,
        ray_remote_args_fn=op.ray_remote_args_fn,
    )


def plan_streaming_repartition_op(
    op: StreamingRepartition,
    physical_children: List[PhysicalOperator],
    data_context: DataContext,
) -> MapOperator:
    assert len(physical_children) == 1
    input_physical_dag = physical_children[0]
    compute = get_compute(op.compute)
    transform_fn = BlockMapTransformFn(
        lambda blocks, ctx: blocks,
        output_block_size_option=OutputBlockSizeOption.of(
            target_num_rows_per_block=op.target_num_rows_per_block,  # To split n*target_max_block_size row into n blocks
        ),
    )
    map_transformer = MapTransformer([transform_fn])

    if op.strict:
        ref_bundler = RebundleQueue(ExactMultipleSize(op.target_num_rows_per_block))
    else:
        ref_bundler = None

    operator = MapOperator.create(
        map_transformer,
        input_physical_dag,
        data_context,
        name=op.name,
        compute_strategy=compute,
        ref_bundler=ref_bundler,
        ray_remote_args=op.ray_remote_args,
        ray_remote_args_fn=op.ray_remote_args_fn,
    )

    return operator


def plan_filter_op(
    op: Filter,
    physical_children: List[PhysicalOperator],
    data_context: DataContext,
) -> MapOperator:
    assert len(physical_children) == 1
    input_physical_dag = physical_children[0]

    output_block_size_option = OutputBlockSizeOption.of(
        target_max_block_size=data_context.target_max_block_size,
    )

    predicate_expr = op.predicate_expr
    compute = get_compute(op.compute)
    if predicate_expr is not None:

        def filter_block_fn(
            blocks: Iterable[Block], ctx: TaskContext
        ) -> Iterable[Block]:
            for block in blocks:
                block_accessor = BlockAccessor.for_block(block)
                filtered_block = block_accessor.filter(predicate_expr)
                yield filtered_block

        init_fn = None
        transform_fn = BlockMapTransformFn(
            filter_block_fn,
            output_block_size_option=output_block_size_option,
        )
    else:
        udf_is_callable_class = isinstance(op.fn, CallableClass)
        filter_fn, init_fn = _get_udf(
            op.fn,
            op.fn_args,
            op.fn_kwargs,
            op.fn_constructor_args if udf_is_callable_class else None,
            op.fn_constructor_kwargs if udf_is_callable_class else None,
            compute=compute,
        )

        transform_fn = RowMapTransformFn(
            _generate_transform_fn_for_filter(filter_fn),
            output_block_size_option=output_block_size_option,
        )

    map_transformer = MapTransformer([transform_fn], init_fn=init_fn)

    return MapOperator.create(
        map_transformer,
        input_physical_dag,
        data_context,
        name=op.name,
        compute_strategy=compute,
        ray_remote_args=op.ray_remote_args,
        ray_remote_args_fn=op.ray_remote_args_fn,
    )


def plan_udf_map_op(
    op: AbstractUDFMap,
    physical_children: List[PhysicalOperator],
    data_context: DataContext,
) -> MapOperator:
    """Get the corresponding physical operators DAG for AbstractUDFMap operators.

    Note this method only converts the given `op`, but not its input dependencies.
    See Planner.plan() for more details.
    """
    assert len(physical_children) == 1
    input_physical_dag = physical_children[0]

    output_block_size_option = OutputBlockSizeOption.of(
        target_max_block_size=data_context.target_max_block_size,
    )

    compute = get_compute(op.compute)
    udf_is_callable_class = isinstance(op.fn, CallableClass)
    fn, init_fn = _get_udf(
        op.fn,
        op.fn_args,
        op.fn_kwargs,
        op.fn_constructor_args if udf_is_callable_class else None,
        op.fn_constructor_kwargs if udf_is_callable_class else None,
        compute=compute,
    )

    if isinstance(op, MapBatches):
        transform_fn = BatchMapTransformFn(
            _generate_transform_fn_for_map_batches(fn),
            batch_size=op.batch_size,
            batch_format=op.batch_format,
            zero_copy_batch=op.zero_copy_batch,
            output_block_size_option=output_block_size_option,
        )

    else:
        if isinstance(op, MapRows):
            udf_fn = _generate_transform_fn_for_map_rows(fn)
        elif isinstance(op, FlatMap):
            udf_fn = _generate_transform_fn_for_flat_map(fn)
        else:
            raise ValueError(f"Found unknown logical operator during planning: {op}")

        transform_fn = RowMapTransformFn(
            udf_fn,
            output_block_size_option=output_block_size_option,
        )

    map_transformer = MapTransformer([transform_fn], init_fn=init_fn)

    return MapOperator.create(
        map_transformer,
        input_physical_dag,
        data_context,
        name=op.name,
        compute_strategy=compute,
        min_rows_per_bundle=op.min_rows_per_bundled_input,
        ray_remote_args_fn=op.ray_remote_args_fn,
        ray_remote_args=op.ray_remote_args,
        per_block_limit=op.per_block_limit,
    )


def _get_udf(
    op_fn: Callable,
    op_fn_args: Tuple[Any, ...],
    op_fn_kwargs: Dict[str, Any],
    op_fn_constructor_args: Optional[Tuple[Any, ...]],
    op_fn_constructor_kwargs: Optional[Dict[str, Any]],
    compute: Optional[ComputeStrategy],
):
    # Note, it's important to define these standalone variables.
    # So the parsed functions won't need to capture the entire operator, which may not
    # be serializable.
    udf = op_fn
    fn_args = op_fn_args or ()
    fn_kwargs = op_fn_kwargs or {}

    if isinstance(udf, CallableClass):
        from ray.data.expressions import _CallableClassSpec

        fn_constructor_args = op_fn_constructor_args or ()
        fn_constructor_kwargs = op_fn_constructor_kwargs or {}

        is_async_udf = _is_async_udf(udf.__call__)

        # Capture original class BEFORE wrapping for use as dict key
        original_udf_class = udf

        if (
            not is_async_udf
            and isinstance(compute, ActorPoolStrategy)
            and not compute.enable_true_multi_threading
        ):
            # NOTE: By default Actor-based UDFs are restricted to run within a
            # single-thread (when enable_true_multi_threading=False).
            #
            # Historically, this has been done to allow block-fetching, batching, etc to
            # be overlapped with the actual UDF invocation, while avoiding the
            # pitfalls of concurrent GPU access (like OOMs, etc) when specifying
            # max_concurrency > 1.
            udf = make_callable_class_single_threaded(udf)

        # Create the callable class spec for this UDF
        callable_class_spec = _CallableClassSpec(
            cls=original_udf_class,
            args=fn_constructor_args,
            kwargs=fn_constructor_kwargs,
        )

        # Use the shared init function creator (handles both map_batches and expressions)
        init_fn = create_actor_context_init_fn(
            udf_specs=[UDFSpec(spec=callable_class_spec, instantiation_class=udf)]
        )

        # Capture the spec for lookup on the actor
        captured_spec = callable_class_spec

        if inspect.iscoroutinefunction(udf.__call__):
            # Async coroutine UDF: wrapper must be async to work with async transform machinery
            async def _wrapped_udf_map_fn(item: Any) -> Any:
                assert ray.data._map_actor_context is not None
                assert ray.data._map_actor_context.is_async

                try:
                    # Use spec's key for lookup
                    udf_key = captured_spec.make_key()
                    udf_instance = ray.data._map_actor_context.udf_instances[udf_key]
                    # Direct await - already in async context
                    return await udf_instance(
                        item,
                        *fn_args,
                        **fn_kwargs,
                    )
                except Exception as e:
                    _try_wrap_udf_exception(e)

        elif inspect.isasyncgenfunction(udf.__call__):

            async def _wrapped_udf_map_fn(item: Any) -> Any:
                assert ray.data._map_actor_context is not None
                assert ray.data._map_actor_context.is_async

                try:
                    # Use spec's key for lookup
                    udf_key = captured_spec.make_key()
                    udf_instance = ray.data._map_actor_context.udf_instances[udf_key]
                    gen = udf_instance(
                        item,
                        *fn_args,
                        **fn_kwargs,
                    )

                    async for res in gen:
                        yield res
                except Exception as e:
                    _try_wrap_udf_exception(e, item)

        else:
            assert isinstance(
                udf.__call__, Callable
            ), f"Expected Callable, got {udf.__call__} ({type(udf.__call__)})"

            def _wrapped_udf_map_fn(item: Any) -> Any:
                assert ray.data._map_actor_context is not None
                assert not ray.data._map_actor_context.is_async
                try:
                    # Use spec's key for lookup
                    udf_key = captured_spec.make_key()
                    udf_instance = ray.data._map_actor_context.udf_instances[udf_key]
                    return udf_instance(
                        item,
                        *fn_args,
                        **fn_kwargs,
                    )
                except Exception as e:
                    _try_wrap_udf_exception(e)

    else:

        def _wrapped_udf_map_fn(item: Any) -> Any:
            try:
                return udf(item, *fn_args, **fn_kwargs)
            except Exception as e:
                _try_wrap_udf_exception(e)

        def init_fn():
            pass

    return _wrapped_udf_map_fn, init_fn


def _try_wrap_udf_exception(e: Exception, item: Any = None):
    """If the Ray Debugger is enabled, keep the full stack trace unmodified
    so that the debugger can stop at the initial unhandled exception.
    Otherwise, clear the stack trace to omit noisy internal code path."""
    ctx = ray.data.DataContext.get_current()
    if _is_ray_debugger_post_mortem_enabled() or ctx.raise_original_map_exception:
        raise e
    else:
        raise UserCodeException("UDF failed to process a data block.") from e


# Following are util functions for converting UDFs to `MapTransformCallable`s.


def _validate_batch_output(batch: Block) -> None:
    allowed = isinstance(
        batch,
        (
            list,
            pa.Table,
            np.ndarray,
            collections.abc.Mapping,
            pd.core.frame.DataFrame,
            dict,
        ),
    ) or _is_cudf_dataframe(batch)
    if not allowed:
        raise ValueError(
            "The `fn` you passed to `map_batches` returned a value of type "
            f"{type(batch)}. This isn't allowed -- `map_batches` expects "
            "`fn` to return a `pandas.DataFrame`, `pyarrow.Table`, "
            "`cudf.DataFrame`, `numpy.ndarray`, `list`, or "
            "`dict[str, numpy.ndarray]`."
        )

    if isinstance(batch, list):
        raise ValueError(
            f"Error validating {_truncated_repr(batch)}: "
            "Returning a list of objects from `map_batches` is not "
            "allowed in Ray 2.5. To return Python objects, "
            "wrap them in a named dict field, e.g., "
            "return `{'results': objects}` instead of just `objects`."
        )

    # Handle cudf.DataFrame before the Mapping check, since cudf.DataFrame
    # implements the Mapping protocol. Mirrors the order in batch_to_block.
    if _is_cudf_dataframe(batch):
        return

    if isinstance(batch, collections.abc.Mapping):
        for key, value in list(batch.items()):
            if not _is_valid_column_values(value):
                raise ValueError(
                    f"Error validating {_truncated_repr(batch)}: "
                    "The `fn` you passed to `map_batches` returned a "
                    f"`dict`. `map_batches` expects all `dict` values "
                    f"to be `list` or `np.ndarray` type, but the value "
                    f"corresponding to key {key!r} is of type "
                    f"{type(value)}. To fix this issue, convert "
                    f"the {type(value)} to a `np.ndarray`."
                )


class _TransformingBatchIterator(Iterator[DataBatch]):
    """Iterator that applies a UDF to batches.

    Unlike a generator, local variables in __next__ go out of scope when the method
    returns, avoiding holding references to yielded values.

    Uses a deque with popleft() to actually release references when items are consumed,
    rather than keeping them in an iterator.
    """

    def __init__(self, batches: Iterable[DataBatch], fn: UserDefinedFunction):
        self._input_iter = iter(batches)
        self._fn = fn
        self._cur_output_iter: Optional[Iterator[DataBatch]] = None

    def __iter__(self) -> "_TransformingBatchIterator":
        return self

    def __next__(self) -> DataBatch:
        while True:
            # Check if there's pending output iter we'd continue fetching
            # from
            if self._cur_output_iter is not None:
                try:
                    out_batch = next(self._cur_output_iter)
                except StopIteration:
                    pass
                else:
                    _validate_batch_output(out_batch)
                    return out_batch

            # Fetch the next batch from upstream
            input_batch = next(self._input_iter)

            if (
                not isinstance(input_batch, collections.abc.Mapping)
                and not _is_cudf_dataframe(input_batch)
                and BlockAccessor.for_block(input_batch).num_rows() == 0
            ):
                # For empty input blocks, we directly output them without
                # calling the UDF.
                # TODO(hchen): This workaround is because some all-to-all
                # operators output empty blocks with no schema.
                self._cur_output_iter = _ReleasingIterator(
                    collections.deque([input_batch])
                )
            else:
                try:
                    res = self._fn(input_batch)

                    if not isinstance(res, GeneratorType):
                        # NOTE: It's critical that we're utilizing *releasing* iterator
                        #       to avoid capturing intermediate objects along the whole
                        #       iterator chain
                        self._cur_output_iter = _ReleasingIterator(
                            collections.deque([res])
                        )
                    else:
                        # In cases when UDF returns a generator we iterate over it
                        # as is (given that we can't release intermediate state from
                        # UDF anyway)
                        self._cur_output_iter = res
                except ValueError as e:
                    read_only_msgs = [
                        "assignment destination is read-only",
                        "buffer source array is read-only",
                    ]
                    err_msg = str(e)
                    if any(msg in err_msg for msg in read_only_msgs):
                        raise ValueError(
                            f"Batch mapper function {self._fn.__name__} tried to mutate a "
                            "zero-copy read-only batch. To be able to mutate the "
                            "batch, pass zero_copy_batch=False to map_batches(); "
                            "this will create a writable copy of the batch before "
                            "giving it to fn. To elide this copy, modify your mapper "
                            "function so it doesn't try to mutate its input."
                        ) from e
                    else:
                        raise e from None


def _generate_transform_fn_for_map_batches(
    fn: UserDefinedFunction,
) -> MapTransformCallable[DataBatch, DataBatch]:

    if _is_async_udf(fn):
        transform_fn = _generate_transform_fn_for_async_map(
            fn,
            _validate_batch_output,
            max_concurrency=DEFAULT_ASYNC_BATCH_UDF_MAX_CONCURRENCY,
        )

    else:

        def transform_fn(
            batches: Iterable[DataBatch], _: TaskContext
        ) -> Iterable[DataBatch]:
            return _TransformingBatchIterator(batches, fn)

    return transform_fn


def _is_async_udf(fn: UserDefinedFunction) -> bool:
    return inspect.iscoroutinefunction(fn) or inspect.isasyncgenfunction(fn)


def create_actor_context_init_fn(
    udf_specs: List[UDFSpec],
):
    """Create an init function for registering callable class UDFs in actor context.

    This is the shared core logic between map_batches (single UDF) and expressions (multiple UDFs).

    Args:
        udf_specs: List of UDF specifications

    Returns:
        An init function that sets up all UDFs in the actor context
    """

    def init_fn():
        import ray

        if ray.data._map_actor_context is None:
            # Check if any UDF is async
            has_async_udf = any(
                _is_async_udf(spec.instantiation_class.__call__) for spec in udf_specs
            )

            # Create instances for all callable class UDFs
            udf_instances = {}
            for spec in udf_specs:
                # Use the spec's key for deduplication and lookup
                udf_key = spec.spec.make_key()
                if udf_key not in udf_instances:
                    # Instantiate using the wrapped/processed class
                    udf_instances[udf_key] = spec.instantiation_class(
                        *spec.spec.args, **spec.spec.kwargs
                    )

            # Single unified context for all UDFs
            ray.data._map_actor_context = _MapActorContext(
                is_async=has_async_udf,
                udf_instances=udf_instances,
            )

    return init_fn


def _validate_row_output(item):
    if not isinstance(item, collections.abc.Mapping):
        raise ValueError(
            f"Error validating {_truncated_repr(item)}: "
            "Standalone Python objects are not "
            "allowed in Ray >= 2.5. To return Python objects from map(), "
            "wrap them in a dict, e.g., "
            "return `{'item': item}` instead of just `item`."
        )


def _generate_transform_fn_for_map_rows(
    fn: UserDefinedFunction,
) -> MapTransformCallable[Row, Row]:

    if _is_async_udf(fn):
        transform_fn = _generate_transform_fn_for_async_map(
            fn,
            _validate_row_output,
            # NOTE: UDF concurrency is limited
            max_concurrency=DEFAULT_ASYNC_ROW_UDF_MAX_CONCURRENCY,
        )

    else:

        def transform_fn(rows: Iterable[Row], _: TaskContext) -> Iterable[Row]:
            for row in rows:
                out_row = fn(row)
                _validate_row_output(out_row)
                yield out_row

    return transform_fn


def _generate_transform_fn_for_flat_map(
    fn: UserDefinedFunction,
) -> MapTransformCallable[Row, Iterable[Row]]:
    if _is_async_udf(fn):
        # UDF is a callable class with async generator `__call__` method.
        transform_fn = _generate_transform_fn_for_async_map(
            fn,
            _validate_row_output,
            max_concurrency=DEFAULT_ASYNC_ROW_UDF_MAX_CONCURRENCY,
            is_flat_map=True,
        )

    else:

        def transform_fn(rows: Iterable[Row], _: TaskContext) -> Iterable[Row]:
            for row in rows:
                for out_row in fn(row):
                    _validate_row_output(out_row)
                    yield out_row

    return transform_fn


def _generate_transform_fn_for_filter(
    fn: UserDefinedFunction,
) -> MapTransformCallable[Row, Row]:
    def transform_fn(rows: Iterable[Row], _: TaskContext) -> Iterable[Row]:
        for row in rows:
            if fn(row):
                yield row

    return transform_fn


def _generate_transform_fn_for_map_block(
    fn: UserDefinedFunction,
) -> MapTransformCallable[Block, Block]:
    def transform_fn(blocks: Iterable[Block], _: TaskContext) -> Iterable[Block]:
        for block in blocks:
            out_block = fn(block)
            yield out_block

    return transform_fn


_SENTINEL = object()


@dataclass
class _AsyncUDFOutputError:
    exception: BaseException


T = TypeVar("T")
U = TypeVar("U")


def _generate_transform_fn_for_async_map(
    fn: UserDefinedFunction,
    validate_fn: Callable,
    *,
    max_concurrency: int,
    is_flat_map: bool = False,
) -> MapTransformCallable:
    assert max_concurrency > 0, "Max concurrency must be positive"

    if inspect.isasyncgenfunction(fn):

        async def _apply_udf(
            item: T,
            output_queue: asyncio.Queue,
            cancellation_requested: asyncio.Event,
        ) -> None:
            try:
                async for output in fn(item):
                    # A per-input queue bounds yielded objects while keeping each UDF
                    # driver active. The consumer drains these queues in input order.
                    await output_queue.put(output)
                    # Release the producer's reference before requesting the next item.
                    del output
            except asyncio.CancelledError as e:
                if cancellation_requested.is_set():
                    raise
                error = RuntimeError("Async UDF task was cancelled unexpectedly")
                error.__cause__ = e
                await output_queue.put(_AsyncUDFOutputError(error))
            except BaseException as e:
                await output_queue.put(_AsyncUDFOutputError(e))
            else:
                await output_queue.put(_SENTINEL)

    elif inspect.iscoroutinefunction(fn):

        async def _apply_udf(
            item: T,
            output_queue: asyncio.Queue,
            cancellation_requested: asyncio.Event,
        ) -> None:
            try:
                result = await fn(item)
                # Keep the result intact so synchronous flat_map iterables are
                # consumed on the consumer thread, as they were previously.
                await output_queue.put(result)
            except asyncio.CancelledError as e:
                if cancellation_requested.is_set():
                    raise
                error = RuntimeError("Async UDF task was cancelled unexpectedly")
                error.__cause__ = e
                await output_queue.put(_AsyncUDFOutputError(error))
            except BaseException as e:
                await output_queue.put(_AsyncUDFOutputError(e))
            else:
                await output_queue.put(_SENTINEL)

    else:
        raise ValueError(f"Expected a coroutine function, got {fn}")

    async def _execute_transform(
        it: Iterator[T],
        output_queue_holder: List[asyncio.Queue],
        output_queue_ready: Event,
        execution_finished: Event,
    ) -> None:
        loop = asyncio.get_running_loop()
        # This one-slot queue bridges the async loop to the synchronous consumer.
        # Create it on the loop thread for Python versions where asyncio.Queue binds to
        # the running loop at construction time.
        output_queue = asyncio.Queue(maxsize=1)
        output_queue_holder.append(output_queue)
        output_queue_ready.set()

        # Keep a bounded producer window. Each input has its own one-item channel,
        # so later inputs cannot fill a shared queue and starve the next output index.
        active: Dict[int, Tuple[asyncio.Task, asyncio.Queue, asyncio.Event]] = {}
        consumed = False
        enumerated_it = enumerate(it)
        next_output_idx = 0

        try:
            while True:
                while len(active) < max_concurrency and not consumed:
                    try:
                        idx, item = next(enumerated_it)
                    except StopIteration:
                        consumed = True
                        break

                    per_input_queue = asyncio.Queue(maxsize=1)
                    cancellation_requested = asyncio.Event()
                    task = loop.create_task(
                        _apply_udf(item, per_input_queue, cancellation_requested)
                    )
                    active[idx] = (task, per_input_queue, cancellation_requested)

                if next_output_idx not in active:
                    if consumed and not active:
                        break
                    continue

                task, per_input_queue, _ = active[next_output_idx]
                output = await per_input_queue.get()

                if output is _SENTINEL:
                    await task
                    del active[next_output_idx]
                    next_output_idx += 1
                elif isinstance(output, _AsyncUDFOutputError):
                    raise output.exception
                else:
                    await output_queue.put(output)
                    del output

        except asyncio.CancelledError:
            tasks = [task for task, _, _ in active.values()]
            for task, _, cancellation_requested in active.values():
                if not task.done():
                    cancellation_requested.set()
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        except BaseException as e:
            tasks = [task for task, _, _ in active.values()]
            for task, _, cancellation_requested in active.values():
                if not task.done():
                    cancellation_requested.set()
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await output_queue.put(_AsyncUDFOutputError(e))
        else:
            await output_queue.put(_SENTINEL)
        finally:
            execution_finished.set()

    def _transform(batch_iter: Iterable[T], task_context: TaskContext) -> Iterable[U]:
        output_queue_holder: List[asyncio.Queue] = []
        output_queue_ready = Event()
        execution_finished = Event()

        loop = ray.data._map_actor_context.udf_map_asyncio_loop
        execution_future = asyncio.run_coroutine_threadsafe(
            _execute_transform(
                iter(batch_iter),
                output_queue_holder,
                output_queue_ready,
                execution_finished,
            ),
            loop,
        )
        output_queue_ready.wait()
        output_queue = output_queue_holder[0]

        try:
            while True:
                output_future = asyncio.run_coroutine_threadsafe(
                    output_queue.get(), loop
                )
                output = output_future.result()

                if output is _SENTINEL:
                    execution_future.result()
                    break
                elif isinstance(output, _AsyncUDFOutputError):
                    raise output.exception
                else:
                    if is_flat_map and inspect.iscoroutinefunction(fn):
                        for item in output:
                            validate_fn(item)
                            yield item
                    else:
                        validate_fn(output)
                        yield output
                    del output
        finally:
            if not execution_future.done():
                execution_future.cancel()
            execution_finished.wait()

    return _transform


class _ReleasingIterator(Iterator[T]):
    def __init__(self, d: collections.deque):
        self._d = d

    def __iter__(self):
        return self

    def __next__(self):
        if not self._d:
            raise StopIteration

        return self._d.popleft()
