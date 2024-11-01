import keras
import tensorflow as tf
from keras import layers
from keras.applications import efficientnet
import keras.utils
from typing import Tuple


def get_cnn_model(image_size:Tuple):
    base_model = efficientnet.EfficientNetB0(
        input_shape=(*image_size,3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False
    base_model_out = base_model.output
    base_model_out = layers.Reshape((-1,base_model_out.shape[-1]))(base_model_out)
    cnn_model = keras.models.Model(base_model.input,base_model_out)
    return cnn_model

class TransformerEncoderBlock(layers.Layer):
    def __init__(self,embed_dim,dense_dim,num_heads,**kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention_1 = layers.MultiHeadAttention(
            num_heads=num_heads,key_dim=embed_dim,dropout=0.0
        )
        self.layer_norm_1 = layers.LayerNormalization()
        self.layer_norm_2 = layers.LayerNormalization()
        self.dense_1 = layers.Dense(embed_dim,activation="relu")

    def call(self,inputs,training,mask=None):
        inputs = self.layer_norm_1(inputs)
        inputs = self.dense_1(inputs)

        attention_output_1 = self.attention_1(
            query = inputs,
            value = inputs,
            key = inputs,
            attention_mask = None,
            training=training
        )
        out_1 = self.layer_norm_2(inputs + attention_output_1)
        return out_1
    
class PositionalEmbedding(layers.Layer):
    def __init__(self,sequence_length,vocab_size,embed_dim,**kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding(
            input_dim=vocab_size,output_dim=embed_dim
        )
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length,output_dim=embed_dim
        )
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embed_scale = tf.math.sqrt(tf.cast(embed_dim,tf.float32))

    def call(self,inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0,limit=length,delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_tokens = embedded_tokens * self.embed_scale
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions
    
    def compute_mask(self,inputs,mask = None):
        return tf.math.not_equal(inputs,0)
    

class TransformerDecoderBlock(layers.Layer):
    def __init__(self,embed_dim,ff_dim,num_heads,vocab_size,seq_length,**kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads
        self.attention_1 = layers.MultiHeadAttention(
            num_heads=num_heads,key_dim=embed_dim,dropout=0.1
        )
        self.attention_2 = layers.MultiHeadAttention(
            num_heads=num_heads,key_dim=embed_dim,dropout=0.1
        )
        self.ffn_layer_1 = layers.Dense(ff_dim,activation="relu")
        self.ffn_layer_2 = layers.Dense(embed_dim)

        self.layer_norm_1 = layers.LayerNormalization()
        self.layer_norm_2 = layers.LayerNormalization()
        self.layer_norm_3 = layers.LayerNormalization()

        self.embedding = PositionalEmbedding(
            embed_dim=embed_dim,
            sequence_length=seq_length,
            vocab_size=vocab_size
        )
        self.out = layers.Dense(vocab_size,activation="softmax")
        
        self.dropout_1 = layers.Dropout(0.3)
        self.dropout_2 = layers.Dropout(0.5)
        self.supports_masking = True

    def call(self,inputs,encoder_outputs,training,mask=None):
        inputs = self.embedding(inputs)
        causal_mask = self.get_causal_attention_mask(inputs)
        if mask is not None:
            padding_mask = tf.cast(mask[:,:,tf.newaxis],dtype=tf.int32)
            combined_mask = tf.cast(mask[:,tf.newaxis,:],dtype=tf.int32)
            combined_mask = tf.minimum(combined_mask,causal_mask)

        attention_output_1 = self.attention_1(
            query = inputs,
            value = inputs,
            key = inputs,
            attention_mask = combined_mask,
            training=training,
        )
        out_1 = self.layer_norm_1(inputs+ attention_output_1)

        attention_output_2 = self.attention_2(
            query=out_1,
            value=encoder_outputs,
            key=encoder_outputs,
            attention_mask=padding_mask,
            training=training,
        )
        out_2 = self.layer_norm_2(out_1 + attention_output_2)
        
        ffn_out = self.ffn_layer_1(out_2)
        ffn_out = self.dropout_1(ffn_out,training=training)
        ffn_out = self.ffn_layer_2(ffn_out)

        ffn_out = self.layer_norm_3(ffn_out + out_2,training=training)
        ffn_out = self.dropout_2(ffn_out,training=training)
        preds = self.out(ffn_out)
        return preds
    def get_causal_attention_mask(self,inputs):
        input_shape = tf.shape(inputs)
        batch_size,sequence_length = input_shape[0],input_shape[1]
        i = tf.range(sequence_length)[:,tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i>=j,dtype="int32")
        mask = tf.reshape(mask,(1,input_shape[1],input_shape[1]))
        mult = tf.concat(
            [
                tf.expand_dims(batch_size,-1),
                tf.constant([1,1],dtype=tf.int32)
            ],
            axis=0
        )
        return tf.tile(mask,mult)
    
class ImageCaptioningModel(keras.Model):
    def __init__(
            self,
            cnn_model,
            encoder,
            decoder,
            num_captions_per_image=5,
            image_aug=None,
    ):
        super().__init__()
        self.cnn_model = cnn_model
        self.encoder = encoder
        self.decoder = decoder
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.acc_tracker = keras.metrics.Mean(name="accuracy")
        self.num_caption_per_image = num_captions_per_image
        self.image_aug = image_aug

    def calculate_loss(self,y_true,y_pred,mask):
        loss = self.loss(y_true,y_pred)
        mask = tf.cast(mask,dtype=loss.dtype)
        loss *= mask
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)
    
    def calculate_accuracy(self,y_true,y_pred,mask):
        accuracy = tf.equal(y_true,tf.argmax(y_pred,axis=2))
        accuracy = tf.math.logical_and(mask,accuracy)
        accuracy = tf.cast(accuracy,dtype=tf.float32)
        mask = tf.cast(mask,dtype=tf.float32)
        return tf.reduce_sum(accuracy) / tf.reduce_sum(mask)
    
    def _compute_caption_loss_and_acc(self,img_embed,batch_seq,training=True):
        encoder_out = self.encoder(img_embed,training=training)
        batch_seq_inp = batch_seq[:,:-1]
        batch_seq_true = batch_seq[:,1:]
        mask = tf.math.not_equal(batch_seq_true,0)
        batch_seq_pred = self.decoder(
            batch_seq_inp,encoder_out,training=training,mask=mask
        )
        loss = self.calculate_loss(batch_seq_true,batch_seq_pred,mask)
        acc = self.calculate_accuracy(batch_seq_true,batch_seq_pred,mask)
        return loss,acc
    
    def train_step(self,batch_data):
        batch_img,batch_seq = batch_data
        batch_loss = 0
        batch_acc = 0

        if self.image_aug:
            batch_img = self.image_aug(batch_img)

        #1. Get image embeddings
        img_embed = self.cnn_model(batch_img)

        # 2. pass each of the five captions one by one to the decoder
        # along with the encoder outputs and compute the loss as well as accuracy for each caption.
        for i in range(self.num_caption_per_image):
            with tf.GradientTape() as tape:
                loss , acc = self._compute_caption_loss_and_acc(
                    img_embed,batch_seq[:,i,:],training=True
                ) 

                #3. update loss and accuracy
                batch_loss += loss
                batch_acc += acc
            
            # 4. get the list of all trainable weights
            train_vars = (
                self.encoder.trainable_variables + self.decoder.trainable_variables
            )

            # 5. get the gradients
            grads = tape.gradient(loss,train_vars)

            # 6. update the trainable weights
            self.optimizer.apply_gradients(zip(grads,train_vars))

        # 7. update the trackers
        batch_acc /= float(self.num_caption_per_image)
        self.loss_tracker.update_state(batch_loss)
        self.acc_tracker.update_state(batch_acc)

        # 8.return the loss and accuracy values

        return {
            "loss":self.loss_tracker.result(),
            "acc":self.acc_tracker.result(),
        }
    def test_step(self,batch_data):

        batch_img, batch_seq = batch_data
        batch_loss = 0
        batch_acc = 0

        img_embed = self.cnn_model(batch_img)

        for i in range(self.num_caption_per_image):
            loss,acc = self._compute_caption_loss_and_acc(
                img_embed,batch_seq[:,i,:],training=False
            )

            batch_loss += loss
            batch_acc += acc

        batch_acc /= float(self.num_caption_per_image)

        self.loss_tracker.update_state(batch_loss)
        self.acc_tracker.update_state(batch_acc)

        return {
            "loss":self.loss_tracker.result(),
            "acc":self.acc_tracker.result()
        }
    @property
    def metrics(self):
        return [self.loss_tracker,self.acc_tracker]
    



