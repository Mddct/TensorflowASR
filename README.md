# dataset 设计原则
1 与模型分离，
2 尽量原生tf api
3 文本预处理放到dataset 之前
4 词典生成放到dataset 之前
3 tfrecord 中tf.train.Example 只包含 {audio:..., transcript:....,labels:} 
