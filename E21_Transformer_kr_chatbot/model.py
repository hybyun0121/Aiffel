import tensorflow as tf

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, d_model, name="multi_head_attention"):
        super(MultiHeadAttention, self).__init__(name=name)
        self.num_heads = num_heads
        self.d_model = d_model

        assert self.d_model % self.num_heads == 0

        self.depth = self.d_model // self.num_heads

        self.query_dense = tf.keras.layers.Dense(units=self.d_model)
        self.key_dense = tf.keras.layers.Dense(units=self.d_model)
        self.value_dense = tf.keras.layers.Dense(units=self.d_model)

        self.dense = tf.keras.layers.Dense(units=self.d_model)

    def split_heads(self, inputs, batch_size):
        inputs = tf.reshape(
            inputs, shape=(batch_size, -1, self.num_heads, self.depth)
        )
        return tf.transpose(inputs, perm=[0,2,1,3])

    def call(self, inputs):
        query, key, value, mask = inputs['query'], inputs['key'], inputs['value'], inputs['mask']
        batch_size = tf.shape(query)[0]

        # liner layers
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        # 병렬 연산을 위한 머리를 여러 개 만든다.
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        #스케일드 닷-프로덕트 어텐션
        scaled_attention = self.scaled_dot_product_attention(query, key, value, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0,2,1,3])

        #어텐션 연산 후에 각 결과를 다시 연결(concatenate).
        concat_attention = tf.reshape(scaled_attention,
                                           (batch_size, -1, self.d_model))

        # final linear layer
        outputs = self.dense(concat_attention)

        return outputs

    def scaled_dot_product_attention(self, query, key, value, mask):
        """어텐션 가중치를 계산"""
        matmul_qk = tf.matmul(query, key, transpose_b = True)

        # scale matmul_qk
        depth = tf.cast(tf.shape(key)[-1], tf.float32)
        logits = matmul_qk / tf.math.sqrt(depth)

        if mask is not None:
            logits += (mask * -1e9)

        # softmax is normalized on the last axis (seq_len_k)
        attention_weights = tf.nn.softmax(logits, axis=-1)

        output = tf.matmul(attention_weights, value)

        return output

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)
    
    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model)
        # 배열의 짝수 인덱스에는 sin 함수 적용
        sines = tf.math.sin(angle_rads[:, 0::2])
        
        # 배열의 홀수 인덱스에는 cosine 함수 적용
        cosines = tf.math.cos(angle_rads[:, 1::2])

        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

class Transformer():
    def __int__(self):
        pass

    def transformer(self,
                    vocab_size,
                    num_layers,
                    units,
                    d_model,
                    num_heads,
                    dropout,
                    name="transformer",):

        inputs = tf.keras.Input(shape=(None,), name="inputs")
        dec_inputs = tf.keras.Input(shape=(None,), name="dec_inputs")

        #인코더에서 패딩을 위한 마스크
        enc_padding_mask = tf.keras.layers.Lambda(
            self.create_padding_mask, output_shape=(1,1, None),
            name='enc_padding_mask')(inputs)

        #디코더에서 미래의 토큰을 마스크하기위해서 사용한다.
        #내부적으로 패딩 마스크도 포함되어져 있다.
        look_ahead_mask = tf.keras.layers.Lambda(
            self.create_look_ahead_mask,
            output_shape=(1, None, None),
            name='look_ahead_mask')(dec_inputs)

        #두번째 어텐션 블록에서 인코더의 벡터들을 마스킹
        #디코더에서 패딩을 위한 마스크
        dec_padding_mask = tf.keras.layers.Lambda(
            self.create_padding_mask, output_shape=(1, 1, None),
            name='dec_padding_mask')(inputs)

        #인코더
        enc_outputs = self.encoder(vocab_size=vocab_size,
                                   num_layers=num_layers,
                                   units=units,
                                   d_model=d_model,
                                   num_heads=num_heads,
                                   dropout=dropout,)(inputs=[inputs, enc_padding_mask])
        #디코더
        dec_outputs = self.decoder(vocab_size=vocab_size,
                                   num_layers=num_layers,
                                   units=units,
                                   d_model=d_model,
                                   num_heads=num_heads,
                                   dropout=dropout,)(inputs=[dec_inputs, enc_outputs,
                                                            look_ahead_mask, dec_padding_mask])

        outputs = tf.keras.layers.Dense(units=vocab_size, name="outputs")(dec_outputs)

        return tf.keras.Model(inputs=[inputs, dec_inputs], outputs=outputs, name=name)

    def create_padding_mask(self, x):
        mask = tf.cast(tf.math.equal(x,0), tf.float32)
        # (batch_size, 1, 1, sequence length)
        return mask[:, tf.newaxis, tf.newaxis, :]

    def create_look_ahead_mask(self, x):
        seq_len = tf.shape(x)[1]
        look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)),-1,0)
        padding_mask = self.create_padding_mask(x)
        return tf.maximum(look_ahead_mask, padding_mask)

    def encoder_layer(self, units, d_model, num_heads, dropout, name='encoder_layers'):
        inputs = tf.keras.Input(shape=(None, d_model), name="inputs")

        #패딩 마스크 사용
        padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

        #첫번째 서브 레이어 : 멀티 헤드 어텐션 수행
        attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads, name="attention")({
            'query': inputs,
            'key' : inputs,
            'value' : inputs,
            'mask' : padding_mask
        })

        attention = tf.keras.layers.Dropout(rate=dropout)(attention)
        attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs + attention)

        # 두번째 서브 레이어 : 2개의 Dense
        outputs = tf.keras.layers.Dense(units=units, activation='relu')(attention)
        outputs = tf.keras.layers.Dense(units=d_model)(outputs)

        outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
        outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention + outputs)

        return tf.keras.Model(
            inputs=[inputs, padding_mask], outputs=outputs, name=name)

    def encoder(self, vocab_size,
                num_layers,
                units,
                d_model,
                num_heads,
                dropout,
                name="encoder"):
        inputs = tf.keras.Input(shape=(None,), name='inputs')

        #패딩 마스크 사용
        padding_mask = tf.keras.Input(shape=(1,1,None), name="padding_mask")

        # 임베딩 레이어
        embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
        embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))

        # 포지셔널 인코딩
        embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)
        outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

        for i in range(num_layers):
            output = self.encoder_layer(
                units=units,
                d_model=d_model,
                num_heads=num_heads,
                dropout=dropout,
                name="encoder_layer_{}".format(i),
            )([outputs, padding_mask])

        return tf.keras.Model(
            inputs=[inputs, padding_mask], outputs=outputs, name=name
        )
    def decoder_layer(self, units, d_model, num_heads, dropout, name="decoder_layer"):
        inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
        enc_outputs = tf.keras.Input(shape=(None, d_model), name="encoder_outputs")
        look_ahead_mask = tf.keras.Input(shape=(1, None, None), name="look_ahead_mask")
        padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

        attention1 = MultiHeadAttention(d_model=d_model, num_heads=num_heads, name="attention_1")(inputs={
            'query':inputs,
            'key':inputs,
            'value':inputs,
            'mask':look_ahead_mask
        })

        attention1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention1+inputs)

        attention2 = MultiHeadAttention(d_model=d_model, num_heads=num_heads, name="attention_2")(inputs={
            'query': attention1,
            'key': enc_outputs,
            'value': enc_outputs,
            'mask': padding_mask
        })

        # 마스크드 멀티 헤드 어텐션의 결과는
        # dropout과 layerNormalization이라는 훈련을 돕는 테크닉을 수행
        attention2 = tf.keras.layers.Dropout(rate=dropout)(attention2)
        attention2 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6)(attention2 + attention1)

        # 세번째 서브 레이어 : 2개의 완전연결층
        outputs = tf.keras.layers.Dense(units=units, activation='relu')(attention2)
        outputs = tf.keras.layers.Dense(units=d_model)(outputs)

        # 완전연결층의 결과는 Dropout과 LayerNormalization 수행
        outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
        outputs = tf.keras.layers.LayerNormalization(
            epsilon=1e-6)(outputs + attention2)

        return tf.keras.Model(
            inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
            outputs=outputs,
            name=name)

    def decoder(self,
                vocab_size,
                num_layers,
                units,
                d_model,
                num_heads,
                dropout,
                name='decoder'):

        inputs = tf.keras.Input(shape=(None,), name='inputs')
        enc_outputs = tf.keras.Input(shape=(None, d_model), name='encoder_outputs')
        look_ahead_mask = tf.keras.Input(
            shape=(1, None, None), name='look_ahead_mask')

        # 패딩 마스크
        padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

        # 임베딩 레이어
        embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
        embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))

        # 포지셔널 인코딩
        embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)

        # Dropout이라는 훈련을 돕는 테크닉을 수행
        outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

        for i in range(num_layers):
            outputs = self.decoder_layer(
                units=units,
                d_model=d_model,
                num_heads=num_heads,
                dropout=dropout,
                name='decoder_layer_{}'.format(i),
            )(inputs=[outputs, enc_outputs, look_ahead_mask, padding_mask])

        return tf.keras.Model(
            inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
            outputs=outputs,
            name=name)