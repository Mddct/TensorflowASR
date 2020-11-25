def _bytes_feature(value):
  if isinstance(value, type(tf.constant(0))): # if value ist tensor
    value = value.numpy() # get value of tensor
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _serialize_example(audio, transcript):

  # Create a dictionary mapping the feature name to the tf.Example-compatible
  # data type.
  
  audio_bytes = tf.io.serialize_tensor( audio)
  transcript_bytes = tf.io.serialize_tensor(transcript)
  feature = {
      'audio': _bytes_feature(audio),
      'transcript': _bytes_feature(transcript),
  }

  # Create a Features message using tf.train.Example.

  example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
  return example_proto.SerializeToString()


def eager_serialize_example(audio, transcript, sample_rate):
  tf_string = tf.py_function(
      _serialize_example,
      (audio, transcript ),
      tf.string
  )
  return tf.reshape(tf_string, ()) 

@tf.function
def _parse_line(line):
  ss = tf.strings.split(line, "\t")
  raw_audio_path, transcript = ss[0], ss[1]
  
  raw_audio = tf.io.read_file(raw_audio_path)

  return raw_audio, transcript, samples_rate

dataset = tf.data.TextLineDataset("test.txt")


dataset = dataset.map(_parse_line).map(eager_serialize_example)

NUM_SHARDS = 12

def reduce_func(key, dataset):
  filename = tf.strings.join([PATH_PREFIX, tf.strings.as_string(key)])
  writer = tf.data.experimental.TFRecordWriter(filename)
  writer.write(dataset.map(lambda _, x: x))

  return tf.data.Dataset.from_tensors(filename)

dataset = dataset.enumerate()
dataset = dataset.apply(tf.data.experimental.group_by_window(
  lambda i, _: i % NUM_SHARDS, reduce_func, tf.int64.max
))


def _parse_example(serialized_example):
  features_map = {
    "audio" : tf.io.FixedLenFeature([], dtype=tf.string),
    "transcript" : tf.io.FixedLenFeature([], dtype=tf.string)
  }
  
  example = tf.io.parse_single_example(serialized_example,  features_map)
  audio = tf.io.parse_tensor(example["audio"])
  transcript = tf.io.parse_tensor(example["transcript"])

  return audio, example
