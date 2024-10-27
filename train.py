from dataset.dataset_preparation import get_image_augmentation,get_complete_dataset
from model.modules import get_cnn_model,TransformerEncoderBlock,TransformerDecoderBlock,\
    ImageCaptioningModel,PositionalEmbedding


import tensorflow as tf
from keras import optimizers,callbacks,metrics,losses
from keras.optimizers import schedules
from keras.optimizers.schedules import learning_rate_schedule

import pickle
import yaml

def train(config):
    filename = config['filename']
    images_path = config['images_path']
    image_size = config['image_size']
    embed_dim = config['embed_dim']
    dense_dim_encoder = config['dense_dim_encoder']
    num_heads_encoder = config['num_heads_encoder']
    dense_dim_decoder = config['dense_dim_decoder']
    num_heads_decoder = config['num_heads_decoder']
    vocab_size = config['vocab_size']
    sequence_length = config['sequence_length']
    epochs = config['epochs']
    model_save_path = config['model_save_path']
    assert model_save_path.endswith("h5"),"model save path must ends with '.h5'"
    vectorizer_save_path = config['vectorizer_save_path']
    assert vectorizer_save_path.endswith(".pkl"),"vectorizer save path must ends with '.pkl'"

    image_aug = get_image_augmentation()
    train_dataset,validation_dataset,vectorizer = get_complete_dataset(filename=filename,images_path=images_path,config=config)

    cnn_model = get_cnn_model(image_size=image_size)
    encoder = TransformerEncoderBlock(embed_dim=embed_dim,dense_dim=dense_dim_encoder,num_heads=num_heads_encoder)
    decoder = TransformerDecoderBlock(embed_dim=embed_dim,ff_dim=dense_dim_decoder,num_heads=num_heads_decoder,vocab_size=vocab_size,seq_length=sequence_length)
    model = ImageCaptioningModel(
        cnn_model=cnn_model,
        decoder=decoder,
        encoder=encoder,
        image_aug=image_aug,
    )

    cross_entropy = losses.SparseCategoricalCrossentropy(
        from_logits=False,
        reduction="none"
    )
    early_stopping = callbacks.EarlyStopping(patience=5,restore_best_weights=True)

    class LRSchedule(learning_rate_schedule.LearningRateSchedule):
        def __init__(self,post_warmup_learning_rate,warmup_steps):
            super().__init__()
            self.post_warmup_learning_rate = post_warmup_learning_rate
            self.warmup_steps = warmup_steps

        def __call__(self, step):
            global_step = tf.cast(step,tf.float32)
            warmup_steps = tf.cast(self.warmup_steps,tf.float32)
            warmup_progress = global_step / warmup_steps
            warmup_learning_rate = self.post_warmup_learning_rate * warmup_progress
            return tf.cond(
                global_step < warmup_steps,
                lambda: warmup_learning_rate,
                lambda:self.post_warmup_learning_rate,
            )
    
    num_train_steps = len(train_dataset) * epochs
    num_warmup_steps = num_train_steps // 15
    lr_schedule = LRSchedule(post_warmup_learning_rate=1e-4,warmup_steps=num_warmup_steps)
    model.compile(optimizer=optimizers.Adam(lr_schedule),loss=cross_entropy)
    model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=validation_dataset,
        callbacks=[early_stopping],
    )
    model.save_weights(filepath=model_save_path)
    pickle.dump({"config":vectorizer.get_config(),"weights":vectorizer.get_weights()},open(vectorizer_save_path,"wb"))
    

def main():
    config_path = r"C:\Users\omera\Desktop\image captioning\tensorflow_image_captioning\config.yaml"
    with open(config_path,"r") as yaml_file:
        config = yaml.load(yaml_file,Loader=yaml.FullLoader)
    train(config=config)


if __name__ == "__main__":
    main()



