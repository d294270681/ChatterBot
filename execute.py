import tensorflow as tf
from model import Encoder, Decoder
import datetime

print(tf.__version__)

source_path = './ids/source.txt'
target_path = './ids/target.txt'
dict_path = './ids/all_dict.txt'

# %% 一 训练前的数据准备工作
# 1.  ID向量
# 1.1  构建DataSet（数据集）
source_ds = tf.data.TextLineDataset(source_path)
target_ds = tf.data.TextLineDataset(target_path)

CONST = {'_BOS': 0, '_EOS': 1, '_UNK': 2, '_PAD': 3}
# 1.2 加载词典（加载维哈希表）
table = tf.lookup.StaticHashTable(
    initializer=tf.lookup.TextFileInitializer(
        filename=dict_path,
        key_dtype=tf.string,
        key_index=tf.lookup.TextFileIndex.WHOLE_LINE,
        value_dtype=tf.int64,
        value_index=tf.lookup.TextFileIndex.LINE_NUMBER,
    ),
    default_value=CONST['_UNK'] - len(CONST),  # 默认值（不出现在字典的值）
)
# 1.3 转换
# s = '这件 衣服 有货 吗'
# table.lookup(tf.strings.split(s))
source_ds = source_ds.map(lambda s: table.lookup(tf.strings.split(s, sep=' ')) + len(CONST))
target_ds = target_ds.map(lambda s: table.lookup(tf.strings.split(s, sep=' ')) + len(CONST))

source_ds = source_ds.map(lambda s2: tf.concat(values=[[CONST['_BOS']], s2, [CONST['_EOS']]], axis=0))
target_ds = target_ds.map(lambda s2: tf.concat(values=[[CONST['_BOS']], s2, [CONST['_EOS']]], axis=0))

# 2.  数据填充
max_seq_len = 30
batch_size = 10
# tf.keras.preprocessing.sequence.pad_sequences()
source_ds = source_ds.map(lambda x: tf.cast(x, tf.int32))
target_ds = target_ds.map(lambda x: tf.cast(x, tf.int32))
dataset = tf.data.Dataset.zip((source_ds, target_ds))

# dataset = dataset.cache()
dataset = dataset.shuffle(5)
dataset_ds = dataset.padded_batch(batch_size, padded_shapes=(max_seq_len, max_seq_len),
                                  padding_values=(CONST['_PAD'], CONST['_PAD']), drop_remainder=False)

# 3.  提高训练速度
dataset_ds = dataset_ds.prefetch(tf.data.experimental.AUTOTUNE)

# %% 二 模型搭建
embedding_size = 128  # 词嵌入维度
enc_units = 128  # encoder端隐层神经元个数
dec_units = 128  # decoder端隐层神经元个数
# 2.1 实例化Encoder端
encoder = Encoder(vocab_size=table.size() + len(CONST), embedding_dim=embedding_size, enc_units=enc_units)
# 2.2 实例化Decoder端（Attention机制）
decoder = Decoder(vocab_size=table.size() + len(CONST), embedding_dim=embedding_size, dec_units=dec_units)
# 2.3 定义loss function , optimization
optimizer = tf.keras.optimizers.Adam()  # 实例化优化器
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)  # 实例化损失函数


def loss_function(loss_object, y_real, y_pred):
    loss_ = loss_object(y_real, y_pred)
    mask = tf.math.logical_not(tf.math.equal(y_real, CONST['_PAD']))
    mask = tf.cast(mask, loss_.dtype)
    return tf.reduce_mean(loss_ * mask)


# 2.4 checkpoint路径参数设置
checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)

# %% 三 模型训练
# 3.3循环
epochs = 501
ckpt_path = './model/chatbot.ckpt'
print('开始训练')
for j in range(epochs):
    total_loss = 0
    for i, (src, tgt) in enumerate(dataset_ds):
        # print(src.shape, tgt.shape)
        # 3.1 前向传播
        with tf.GradientTape() as tape:  # 定义定义梯度的磁带
            enc_output, enc_hidden = encoder(src)
            _, tgt_length = tgt.shape
            loss = 0
            for t in range(tgt_length-1):
                dec_input = tf.expand_dims(tgt[:, t], 1)
                prediction, dec_hidden, _ = decoder(dec_input, enc_hidden, enc_output)
                loss += loss_function(loss_object, tgt[:, t+1], prediction)
        batch_loss = loss/tgt_length
        total_loss += batch_loss

        # 3.2 反向更新
        variables = encoder.trainable_variables + decoder.trainable_variables
        grads = tape.gradient(loss, variables)  # 计算梯度
        optimizer.apply_gradients(zip(grads, variables))

    if j%100 == 0:
        print(f'{datetime.datetime.now()} Epochs: {j} loss: {total_loss:.4f}')
        checkpoint.save(ckpt_path)
