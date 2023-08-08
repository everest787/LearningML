import tensorflow as tf
contents = 0
primals = 0
tangents = 0
x = 0
y = 0

tf.audio.decode_wav(
	contents, desired_channels=-1, 
    desired_samples=-1, name=None
)

tf.autodiff.ForwardAccumulator(
    primals, tangents
)

tf.bitwise.bitwise_xor(
    x, y, name=None
)

tf.compat.as_str_any(
    value, encoding='utf-8'
)

tf.data.Dataset(
    variant_tensor
)

tf.debugging.assert_equal(
    x, y, message=None, summarize=None, name=None
)
tf.distribute.MirroredStrategy(
    devices=None, cross_device_ops=None
)
tf.dtypes.as_dtype(
    type_value
)
tf.graph_util.import_graph_def(
    graph_def,
    input_map=None,
    return_elements=None,
    name=None,
    producer_op_list=None
)
tf.io.decode_gif(
    contents, name=None
)
tf.linalg.eigvals(
    tensor, name=None
)
tf.lite.TFLiteConverter(
    funcs, trackable_obj=None
)
tf.lookup.KeyValueTensorInitializer(
    keys, values, key_dtype=None, value_dtype=None, name=None
)
tf.math.argmax(
    input,
    axis=None,
    output_type=tf.dtypes.int64,
    name=None
)
tf.nest.is_nested(
    seq
)
tf.quantization.quantize(
    input,
    min_range,
    max_range,
    T,
    mode='MIN_COMBINED',
    round_mode='HALF_AWAY_FROM_ZERO',
    name=None,
    narrow_range=False,
    axis=None,
    ensure_minimum_range=0.01
)
tf.queue.FIFOQueue(
    capacity,
    dtypes,
    shapes=None,
    names=None,
    shared_name=None,
    name='fifo_queue'
)
tf.ragged.constant(
    pylist,
    dtype=None,
    ragged_rank=None,
    inner_shape=None,
    name=None,
    row_splits_dtype=tf.dtypes.int64
)
tf.random.normal(
    shape,
    mean=0.0,
    stddev=1.0,
    dtype=tf.dtypes.float32,
    seed=None,
    name=None
)
tf.saved_model.load(
    export_dir, tags=None, options=None
)
tf.sets.union(
    a, b, validate_indices=True
)
tf.signal.fft(
    input, name=None
)
tf.sparse.add(
    a, b, threshold=0
)
tf.strings.join(
    inputs, separator='', name=None
)
tf.summary.histogram(
    name, data, step=None, buckets=None, description=None
)
tf.sysconfig.get_build_info()
tf.test.create_local_cluster(
    num_workers,
    num_ps,
    protocol='grpc',
    worker_config=None,
    ps_config=None
)
tf.tpu.XLAOptions(
    use_spmd_for_xla_partitioning=True, enable_xla_dynamic_padder=True
)
tf.train.latest_checkpoint(
    checkpoint_dir, latest_filename=None
)
