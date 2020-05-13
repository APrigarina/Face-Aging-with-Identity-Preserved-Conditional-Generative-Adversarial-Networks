import os.path
import os
os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import numpy as np
import tensorflow as tf
from datetime import datetime
from models import FaceAging
import sys
sys.path.append('./tools/')
from source_input import load_source_batch3
from utils import save_images, save_source
from data_generator import ImageDataGenerator
import math

    

flags = tf.app.flags
flags.DEFINE_float("learning_rate", 0.01, "Learning rate")

flags.DEFINE_integer("batch_size", 64, "The size of batch images")

flags.DEFINE_integer("image_size", 128, "the size of the generated image")

flags.DEFINE_integer("noise_dim", 256, "the length of the noise vector")

flags.DEFINE_integer("feature_size", 128, "image size after stride 2 conv")

flags.DEFINE_integer("age_groups", 5, "the number of different age groups")

flags.DEFINE_integer('max_steps', 200000, 'Number of batches to run')

flags.DEFINE_string("alexnet_pretrained_model", "/content/drive/My Drive/Diploma/alexnet/alexnet.model-292000",
                    "Directory name to save the checkpoints")

flags.DEFINE_string("age_pretrained_model", "/content/drive/My Drive/Diploma/age_classifier/age_classifier.model-300000",
                    "Directory name to save the checkpoints")

flags.DEFINE_integer('model_index', None, 'the index of trained model')

flags.DEFINE_float("gan_loss_weight", None, "gan_loss_weight")

flags.DEFINE_float("fea_loss_weight", None, "fea_loss_weight")

flags.DEFINE_float("age_loss_weight", None, "age_loss_weight")

flags.DEFINE_float("tv_loss_weight", None, "face_loss_weight")

flags.DEFINE_string("checkpoint_dir", None, "Directory name to save the checkpoints")

flags.DEFINE_string("source_checkpoint_dir", ' ', "Directory name to save the checkpoints")

flags.DEFINE_string("sample_dir", None, "Directory name to save the sample images")

flags.DEFINE_string("fea_layer_name", None, "which layer to use for fea_loss")

flags.DEFINE_string("source_file", '/content/drive/My Drive/Diploma/dataset/train.txt', "source file path")

flags.DEFINE_string("root_folder", '/content/drive/My Drive/Diploma/dataset/CACD2000/', "folder that contains images")

FLAGS = flags.FLAGS

# How often to run a batch through the validation model.
VAL_INTERVAL = 100

# How often to save a model checkpoint
SAVE_INTERVAL = 5000

d_iter = 1
g_iter = 1

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# Initalize the data generator seperately for the training and validation set
train_generator = ImageDataGenerator(batch_size=FLAGS.batch_size, height=FLAGS.feature_size, width=FLAGS.feature_size,
                                     z_dim=FLAGS.noise_dim, scale_size=(FLAGS.image_size, FLAGS.image_size), mode='train')
def my_train():
    with tf.Graph().as_default():
        sess = tf.Session(config=config)
        model = FaceAging(sess=sess, lr=FLAGS.learning_rate, keep_prob=1., model_num=FLAGS.model_index, batch_size=FLAGS.batch_size,
                        age_loss_weight=FLAGS.age_loss_weight, gan_loss_weight=FLAGS.gan_loss_weight,
                        fea_loss_weight=FLAGS.fea_loss_weight, tv_loss_weight=FLAGS.tv_loss_weight)

        # imgs = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 3])
        # age_label = tf.placeholder(tf.int32, [FLAGS.batch_size])

        # print('Shapes')
        # print('imgs', imgs)
        # print('age_label', age_label)

        source_img_227, _, age_label = load_source_batch3(FLAGS.source_file, FLAGS.root_folder, FLAGS.batch_size)
        print("after load source batch3")
        model.train_mobilenet(source_img_227, age_label)

        # Create a saver.
        model.mobilenet_saver = tf.train.Saver(model.mobilenet_vars)
        print("Model mobilenet", model.mobilenet_vars)

        age_error = model.age_loss

        # Start running operations on the Graph.
        sess.run(tf.global_variables_initializer())
        print("before start_queue_runners")
        tf.train.start_queue_runners(sess)
        print("after start_queue_runners")

        # print("age_lsgan before restore ",FLAGS.checkpoint_dir, model.saver )
        
        # print("alexnet_saver before restore ",FLAGS.alexnet_pretrained_model )
        # model.alexnet_saver.restore(sess, FLAGS.alexnet_pretrained_model)
        # print("age_saver before restore ",FLAGS.age_pretrained_model )
        # model.age_saver.restore(sess, FLAGS.age_pretrained_model)
        #FLAGS.checkpoint_dir, model.saver, 'acgan', 399999
        # print("==========load===========")
        # print("BEFORE")
        # for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
        #   print(v)
        # v1 = tf.get_variable('alexnet/conv5/weights', shape=[4])
        # if model.load(FLAGS.checkpoint_dir, model.saver, 'acgan', 399999):
        #     print(" [*] Load SUCCESS")
        # else:
        #     print(" [!] Load failed...")

        print("{} Start training...")

        # Loop over max_steps
        for step in range(FLAGS.max_steps):

            if step % 15 == 0:
                model.learning_rate = model.learning_rate * math.exp(-0.1)

            # images, _, _, _, age_labels = \
            #     train_generator.next_target_batch_transfer2()
            
            # print("images shape", images.shape)

            age_loss = sess.run([model.m_optim, age_error])

            format_str = ('%s: step %d, age_loss=%.3f')
            if step % 10 == 0: 
                print(format_str % (datetime.now(), step, age_loss))

            # Save the model checkpoint periodically.
            if step % SAVE_INTERVAL == SAVE_INTERVAL-1 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.checkpoint_dir)
                model.save(checkpoint_path, step, 'mobilenet')

            # if step % VAL_INTERVAL == VAL_INTERVAL-1:
            #     if not os.path.exists(FLAGS.sample_dir):
            #         os.makedirs(FLAGS.sample_dir)
            #     path = os.path.join(FLAGS.sample_dir, str(step))
            #     if not os.path.exists(path):
            #         os.makedirs(path)

            #     source = sess.run(source_img_128)
            #     save_source(source, [4, 8], os.path.join(path, 'source.jpg'))
            #     for j in range(train_generator.n_classes):
            #         true_label_fea = train_generator.label_features_128[j]
            #         dict = {
            #                 imgs: source,
            #                 true_label_features_128: true_label_fea
            #                 }
            #         samples = sess.run(ge_samples, feed_dict=dict)
            #         save_images(samples, [4, 8], '{}/test_{:01d}.jpg'.format(path, j))
            #         print("===========> img saved to", '{}/test_{:01d}.jpg'.format(path, j))

def main(argv=None):
    my_train()


if __name__ == '__main__':
        tf.app.run()
