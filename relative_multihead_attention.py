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
    def __init__(self, heads, d_model, k=2, **kwargs):
        super().__init__(**kwargs)
        assert d_model % heads ==0, "Model dim should be divisable by num of heads"
        
        self.heads   = heads
        self.d_model = d_model
        self.d     = self.d_model // self.heads
        
        self.WQ = Linear(self.d_model, self.d_model)
        self.WK = Linear(self.d_model, self.d_model)
        self.WV = Linear(self.d_model, self.d_model)
        
        self.WMerge = Linear(self.d_model, self.d_model)
        

    def call(self, query, key, value, mask=None):
        '''mask is 4d rank tensor of shape [1, n_heads, time_step, time_step]'''
        BATCH_SIZE, _, _ = query.shape
        
        query = self.WQ(query)     
        key = self.WK(key) 
        value = self.WV(value) 
        
        query = query.reshape(BATCH_SIZE, self.heads, -1, self.d)
        key = key.reshape(BATCH_SIZE, self.heads, -1, self.d)
        value = value.reshape(BATCH_SIZE, self.heads, -1, self.d)
        
        out = self.scaled_relative_dot_product(query, key, value, mask=mask); del query, key, value
        
        out = out.reshape(BATCH_SIZE, -1, self.d_model)
        
        out = self.WMerge(out)
        return out
        
        
        
    def scaled_relative_dot_product(self, query, key, value, mask=None, return_score=False):
        
        scores = tf.matmul(query, tf.transpose(key, perm=[0, 1, 3, 2])); del query, key
        scores = scores / tf.sqrt(self.d)
        if mask is not None:
            scores = tf.where(mask, 1e-9, scores)

        scores = tf.nn.softmax(scores)
        output = tf.matmul(scores, value)
        if return_score:
            return output, scores
        return output
    
    
        
    
    


    
    
def create_mask(self, x):
    tf.linalg.band_part(x, -1, 0)