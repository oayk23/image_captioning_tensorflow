from model.modules import ImageCaptioningModel,get_cnn_model,TransformerEncoderBlock,TransformerDecoderBlock
from dataset.dataset_preparation import TextVectorizer,get_image_augmentation
import pickle
import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import yaml


def decode_and_resize(img_path,config):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img,channels=3)
    img = tf.image.resize(img,config['image_size'])
    img = tf.image.convert_image_dtype(img,tf.float32)
    return img

def generate_caption(model,vectorizer,image_path,config):
    vocab = vectorizer.get_vocabulary()
    index_lookup = dict(zip(range(len(vocab)),vocab))
    image = decode_and_resize(image_path,config)
    img = np.array(image).clip(0,255).astype(np.uint8)
    plt.imshow(img)
    
    img = tf.expand_dims(image,0)
    img = model.cnn_model(img)

    encoded_img = model.encoder(img,training=False)

    decoded_caption = "<start> "
    for i in range(config['sequence_length']):
        tokenized_caption = vectorizer([decoded_caption])[:,:-1]
        mask = tf.math.not_equal(tokenized_caption,0)
        predictions = model.decoder(
            tokenized_caption,encoded_img,training=False,mask=mask
        )
        sampled_token_index = np.argmax(predictions[0,i,:])
        sampled_token = index_lookup[sampled_token_index]
        if sampled_token == "<end>":
            break
        decoded_caption += " " + sampled_token
    
    decoded_caption = decoded_caption.replace("<start>","")
    decoded_caption = decoded_caption.replace(" <end>","").strip()
    plt.title(decoded_caption)
    plt.show()
    plt.savefig("my_caption.png")
    print(decoded_caption)

def main(config):

    model_path = config['model_save_path']
    assert model_path.endswith(".h5"),"model path must endswith '.h5'"
    vectorizer_path = config['vectorizer_save_path']
    assert vectorizer_path.endswith(".pkl"),"vectorizer path must endsrith '.pkl'"
    image_aug = get_image_augmentation()
    encoder = TransformerEncoderBlock(embed_dim=config['embed_dim'],dense_dim=config['dense_dim_encoder'],num_heads=config['num_heads_encoder'])
    decoder = TransformerDecoderBlock(embed_dim=config['embed_dim'],ff_dim=config['dense_dim_decoder'],num_heads=config['num_heads_decoder'],vocab_size=config['vocab_size'],seq_length=config['sequence_length'])
    cnn_model = get_cnn_model(image_size=config['image_size'])
    model = ImageCaptioningModel(cnn_model=cnn_model,encoder=encoder,decoder=decoder,image_aug=image_aug)
    
    
    model.built = True
    


    vectorizer_config = pickle.load(open(vectorizer_path,"rb"))
    vectorizer = TextVectorizer.from_config(vectorizer_config['config'])
    vectorizer.set_weights(vectorizer_config['weights'])
    sample_image_path = config['sample_image_path']
    print(len(vectorizer.get_vocabulary()))
    generate_caption(model,vectorizer,sample_image_path,config)

if __name__ == "__main__":
    config_path = r"C:\Users\omera\Desktop\image captioning\tensorflow_image_captioning\config.yaml"
    with open(config_path,"r") as yaml_file:
        config = yaml.load(yaml_file,Loader=yaml.FullLoader)
    main(config)





