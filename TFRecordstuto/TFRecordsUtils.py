import tensorflow as tf
import numpy as np
from PIL import Image 
from tqdm import tqdm
import os
from random import shuffle
from operator import itemgetter
class TFRecordGenerator:
    '''
    La generation de TFrecords apartir de images dans des dossier
    '''
    
    def __init__(self):
        self.TFRecords = []
        
    def _get_files(self, path):
        '''
        retrouver toutes les images
        retrun liste images , classes
        '''
        print('*'*20 + ' Reading Images')
        dirs = [c for c in os.listdir(path) if not c.startswith('.')]
        images = []
        labels = []
        for i, d in tqdm(enumerate(dirs), desc= 'folders'):
            dir_path = path + d
            for f in tqdm(os.listdir(dir_path), desc='reading images in folder'):
                if(not f.startswith('.')):
                    file_path = dir_path + '/' + f
                    images.append(file_path)
                    labels.append(i)
        data = list(zip(images, labels))
        shuffle(data)
        img_getter = itemgetter(0)
        label_getter = itemgetter(1)
        
        images = list(map(img_getter,data))
        labels = list(map(label_getter,data))
        return images, labels

    def convert_image_folder(self, path_folder, tfrecord_file_name):
        '''
        Ecrit les TFRecords files apartir de
        PRAM path_folder
        '''
        img_paths, labels = self._get_files(path_folder)

        with tf.python_io.TFRecordWriter(tfrecord_file_name) as writer:
            for img_path, label in tqdm(zip(img_paths, labels), desc='writing images into TFRecord'):
                example = self._convert_image(img_path, label)
                writer.write(example.SerializeToString())
            self.TFRecords.append(tfrecord_file_name)

    def _convert_image(self, img_path, label):
        '''
        Convertir l'image to tf Example
        '''
        image_data = Image.open(img_path).convert('L')
        if image_data.mode == 'RGB':
            print('RBG here')
        image_data_resized = image_data.resize((300,300))
        image_arr = np.array(image_data_resized)
        image_arr = image_arr /255
        image_arr = image_arr.reshape((300,300,1))
        
        # Convert image to string data
        image_str = image_arr.tostring()
        # Store shape of image for reconstruction purposes
        img_shape = image_arr.shape
        if (image_arr.shape) == 3:
            print('RBG here ALSO')
        # Get filename 'image-XX.jpg'
        filename = os.path.basename(img_path)
        
        example = tf.train.Example(features = tf.train.Features(feature = {
            'filename': tf.train.Feature(bytes_list = tf.train.BytesList(value = [filename.encode('utf-8')])),
            'rows': tf.train.Feature(int64_list = tf.train.Int64List(value = [img_shape[0]])),
            'cols': tf.train.Feature(int64_list = tf.train.Int64List(value = [img_shape[1]])),
            'image': tf.train.Feature(bytes_list = tf.train.BytesList(value = [image_str])),
            'label': tf.train.Feature(int64_list = tf.train.Int64List(value = [label]))
        }))
        return example
