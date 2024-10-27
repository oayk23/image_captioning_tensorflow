import os
import numpy as np
import tensorflow as tf
import re
from keras.layers import TextVectorization
import keras
from keras import layers
import string
from keras.saving.save import generic_utils



@generic_utils.register_keras_serializable(package='custom_layers', name='TextVectorizer')
class TextVectorizer(layers.Layer):
    '''English - Spanish Text Vectorizer'''

    def __init__(self, max_tokens=None, output_mode='int', output_sequence_length=None, standardize='lower_and_strip_punctuation', vocabulary=None, config=None):
        super().__init__()
        if config:
            self.vectorization = layers.TextVectorization.from_config(config)

        else:
            self.max_tokens = max_tokens
            self.output_mode = output_mode
            self.output_sequence_length = output_sequence_length
            self.vocabulary = vocabulary
            if standardize != 'lower_and_strip_punctuation':
                self.vectorization = layers.TextVectorization(max_tokens=self.max_tokens,
                                                              output_mode=self.output_mode,
                                                              output_sequence_length=self.output_sequence_length,
                                                              vocabulary=self.vocabulary,
                                                              standardize=self.standardize)
            else:
                self.vectorization = layers.TextVectorization(max_tokens=self.max_tokens,
                                                              output_mode=self.output_mode,
                                                              output_sequence_length=self.output_sequence_length,
                                                              vocabulary=self.vocabulary)


    def standardize(self, input_string, preserve=['[', ']'], add=['¿']) -> str:
        strip_chars = string.punctuation
        for item in add:
            strip_chars += item
    
        for item in preserve:
            strip_chars = strip_chars.replace(item, '')

        lowercase = tf.strings.lower(input_string)
        output = tf.strings.regex_replace(lowercase, f'[{re.escape(strip_chars)}]', '')

        return output

    def __call__(self, *args, **kwargs):
        return self.vectorization.__call__(*args, **kwargs)

    def get_config(self):
        return {key: value if not callable(value) else None for key, value in self.vectorization.get_config().items()}

    def from_config(config):
        return TextVectorizer(config=config)

    def set_weights(self, weights):
        self.vectorization.set_weights(weights)

    def adapt(self, dataset):
        self.vectorization.adapt(dataset)

    def get_vocabulary(self):
        return self.vectorization.get_vocabulary()

def load_captions_data(filename,images_path,seq_length):

    with open(filename) as caption_file:
        caption_data = caption_file.readlines()
        caption_mapping = {}
        text_data = []
        images_to_skip = set()

        for line in caption_data:
            line = line.rstrip("\n")

            img_name , caption = line.split("\t")

            img_name = img_name.split("#")[0]
            img_name = os.path.join(images_path,img_name.strip())

            tokens = caption.strip().split()

            if len(tokens) < 5 or len(tokens) > seq_length:
                images_to_skip.add(img_name)
                continue

            if img_name.endswith("jpg") and img_name not in images_to_skip:

                caption = "<start> " + caption.strip() + " <end>"
                text_data.append(caption)

                if img_name in caption_mapping:
                    caption_mapping[img_name].append(caption)
                else:
                    caption_mapping[img_name] = [caption]
        for img_name in images_to_skip:
            if img_name in caption_mapping:
                del caption_mapping[img_name]
        
        return caption_mapping, text_data
    
def train_val_split(caption_data:tuple, train_size = 0.8,shuffle = True):

    all_images = list(caption_data.keys())

    if shuffle:
        np.random.shuffle(all_images)

    train_size = int(len(caption_data) * train_size)

    training_data = {
        img_name: caption_data[img_name] for img_name in all_images[:train_size]
    }
    validation_data = {
        img_name: caption_data[img_name] for img_name in all_images[train_size:]
    }

    return training_data,validation_data

def prepare_vectorizer(text_data,config):
    #strip_chars = "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"
    #strip_chars = strip_chars.replace("<","")
    #strip_chars = strip_chars.replace(">","")

    #def custom_standardization(input_string):
    #    lowercase = tf.strings.lower(input_string)
    #    return tf.strings.regex_replace(lowercase, "[%s]" % re.escape(strip_chars),"")
    
    vectorization = TextVectorizer(
        max_tokens=config['vocab_size'],
        standardize=None,
        output_mode="int",
        output_sequence_length=config['sequence_length']
    )

    vectorization.adapt(text_data)

    return vectorization

def get_image_augmentation():
    image_augmentation = keras.Sequential(
        [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomContrast(0.3),
        ]
    )
    return image_augmentation

def get_dataset(train_data,valid_data,vectorization,config):
    def decode_and_resize(img_path,config):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img,channels=3)
        img = tf.image.resize(img,config['image_size'])
        img = tf.image.convert_image_dtype(img,tf.float32)
        return img

    def process_input(img_path,captions):
        return decode_and_resize(img_path,config),vectorization(captions)
    AUTOTUNE = tf.data.AUTOTUNE
    def make_dataset(images,captions,config):
        dataset = tf.data.Dataset.from_tensor_slices((images,captions))
        dataset = dataset.shuffle(config['batch_size']*8)
        dataset = dataset.map(process_input,num_parallel_calls=AUTOTUNE)
        dataset = dataset.batch(config['batch_size']).prefetch(AUTOTUNE)

        return dataset

    train_dataset = make_dataset(list(train_data.keys()),list(train_data.values()),config)

    valid_dataset = make_dataset(list(valid_data.keys()),list(valid_data.values()),config)

    return train_dataset,valid_dataset


def get_complete_dataset(filename,images_path,config):
    caption_mapping,text_data = load_captions_data(filename=filename,images_path=images_path,seq_length=config['sequence_length'])
    training_data,validation_data = train_val_split(caption_data=caption_mapping,train_size=config['train_size'],shuffle=True)
    vectorizer = prepare_vectorizer(text_data=text_data,config=config)
    train_dataset,validation_dataset = get_dataset(train_data=training_data,valid_data=validation_data,vectorization=vectorizer,config=config)
    return train_dataset,validation_dataset,vectorizer
