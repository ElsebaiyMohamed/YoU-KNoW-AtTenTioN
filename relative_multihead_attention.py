'''
NOT completed emplementation
'''

import tensorflow as tf


class Linear(tf.keras.layers.Layer):
    def __init__(self, in_feature, out_feature, name=None):
        super().__init__()
        
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.name = name if name is not None else 'untitled'
        
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(initial_value=w_init(shape=(self.in_feature, self.out_feature),
                                                  dtype='float32'), trainable=True)
        
    def call(self, x):
        x = tf.matmul(x, self.w)
        return x


class RelativeMultiHeadAttenion(tf.keras.layers.Layer):
    def __init__(self, heads, d_model, max_len=500, **kwargs):
        super().__init__(**kwargs)
        assert d_model % heads ==0, "Model dim should be divisable by num of heads"
        
        self.max_len = max_len
        self.heads   = heads
        self.d_model = d_model
        self.d     = self.d_model // self.heads
        
        self.WQ = Linear(self.d_model, self.d_model)
        self.WK = Linear(self.d_model, self.d_model)
        self.WV = Linear(self.d_model, self.d_model)
        w_init = tf.random_normal_initializer()
        self.Er = tf.Variable(initial_value=w_init(shape=(self.max_len, self.d), dtype='float32'), trainable=True)
        self.WMerge = Linear(self.d_model, self.d_model)
        

    def call(self, query, key, value, mask=False, training=False):
        '''mask is 4d rank tensor of shape [1, n_heads, time_step, time_step]'''
        BATCH_SIZE, seq_len, _ = query.shape
        
        query = self.WQ(query)     
        key = self.WK(key) 
        value = self.WV(value) 
        
        query = query.reshape(BATCH_SIZE, self.heads, -1, self.d)
        k_t = key.reshape(BATCH_SIZE, self.heads, self.d, -1)
        value = value.reshape(BATCH_SIZE, self.heads, -1, self.d)
        
        start = self.max_len - seq_len
        Er_t = tf.transpose(self.Er[start:, :], perm=[1, 0] )
        print(Er_t.shape)
        QEr = tf.linalg.matmul(query, Er_t)
        # QEr.shape = (batch_size, num_heads, seq_len, seq_len)
        Srel = self.skew(QEr)
        # Srel.shape = (batch_size, num_heads, seq_len, seq_len)
        
        QK_t = tf.linalg.matmul(query, k_t)
        
        attn = (QK_t + Srel) / tf.math.sqrt(query.shape[-1])
        if mask:
            mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), 0, -1) == 0
            mask = tf.expand_dim(mask, axis=0)
            mask = tf.expand_dim(mask, axis=0)
            # mask.shape = (1, 1, seq_len, seq_len)
            attn = mask * attn
            # attn.shape = (batch_size, num_heads, seq_len, seq_len)
        attn = tf.nn.softmax(attn, dim=-1)
        out = tf.linalg.matmul(attn, value)
        # out.shape = (batch_size, num_heads, seq_len, d_head)
        out = tf.transpose(out, perm=[0, 2, 1, 3] )
        # out.shape == (batch_size, seq_len, num_heads, d_head)
        out = out.reshape(BATCH_SIZE, -1, self.d_model)
        # out.shape == (batch_size, seq_len, d_model)
        out = self.WMerge(out)
        if training:
            return tf.nn.dropout(out, rate =0.2)
        return out
    
    def skew(self, QEr):
        # QEr.shape = (batch_size, num_heads, seq_len, seq_len)
        padded = tf.pad(QEr, (1, 0))
        # padded.shape = (batch_size, num_heads, seq_len, 1 + seq_len)
        batch_size, num_heads, num_rows, num_cols = padded.shape
        reshaped = padded.reshape(batch_size, num_heads, num_cols, num_rows)
        # reshaped.size = (batch_size, num_heads, 1 + seq_len, seq_len)
        Srel = reshaped[:, :, 1:, :]
        # Srel.shape = (batch_size, num_heads, seq_len, seq_len)
        return Srel
