import tensorflow as tf
import numpy as np
import os


DATA_DIR = "./"
TRAINING_SET_SIZE = 13718
BATCH_SIZE = 16
IMAGE_SIZE = 64
CHANNELS = 3


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

class _image_object:
    def __init__(self):
        self.image = tf.Variable([], dtype = tf.string)
        self.height = tf.Variable([], dtype = tf.int64)
        self.width = tf.Variable([], dtype = tf.int64)
        self.filename = tf.Variable([], dtype = tf.string)
        self.label = tf.Variable([], dtype = tf.int32)

def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, features = {
        "image/encoded": tf.FixedLenFeature([], tf.string),
        "image/height": tf.FixedLenFeature([], tf.int64),
        "image/width": tf.FixedLenFeature([], tf.int64),
        "image/filename": tf.FixedLenFeature([], tf.string),
        "image/class/label": tf.FixedLenFeature([], tf.int64),})
    image_encoded = features["image/encoded"]
    image_raw = tf.image.decode_jpeg(image_encoded, channels=CHANNELS)
    print(image_raw)
    image_object = _image_object()
    image_object.image = tf.image.resize_image_with_crop_or_pad(image_raw, IMAGE_SIZE, IMAGE_SIZE)
    image_object.height = features["image/height"]
    image_object.width = features["image/width"]
    image_object.filename = features["image/filename"]
    image_object.label = tf.cast(features["image/class/label"], tf.int64)
    return image_object

def net_input(if_random = True, if_training = True):
    if(if_training):
        filenames = [os.path.join(DATA_DIR, "train-0000%d-of-00002.tfrecord" % i) for i in range(0, 1)]
    else:
        filenames = [os.path.join(DATA_DIR, "validation-0000%d-of-00002.tfrecord" % i) for i in range(0, 1)]

    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError("Failed to find file: " + f)
    filename_queue = tf.train.string_input_producer(filenames)
    image_object = read_and_decode(filename_queue)
    image = tf.image.per_image_standardization(image_object.image)
    label = image_object.label
    filename = image_object.filename

    image_batch, label_batch, filename_batch = tf.train.batch([image, label, filename],batch_size = BATCH_SIZE,num_threads = 1)
    return image_batch, label_batch, filename_batch


def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def bias_variable(val,shape):
    if val == 0:
        return tf.Variable(tf.zeros(shape))
    else:
        return tf.Variable(tf.constant(val, shape=shape))

def conv2d(x, W, padding):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=padding)

def max_pool_2x2(x, kernel_size):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1,2,2,1], padding='VALID')

def net_inference(image_batch):
    batch_size = 16
    patch_size = 5
    kernel_size_1 = 5
    kernel_size_2 = 3
    depth1 = 32 #the depth of 1st convnet
    depth2 = 64 #the depth of 2nd convnet
    depth3 = 96 #the depth of 3rd convnet
    C5_units = 128
    F6_units = 64
    F7_units = 3

    x_image = tf.reshape(image_batch, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS])
    
    print(x_image.shape)

    # Start by defining all variables
    W_conv1 = weight_variable([kernel_size_1, kernel_size_1, CHANNELS, depth1])
    b_conv1 = bias_variable(0, [depth1])

    #S2 has no weights

    W_conv3 = weight_variable([kernel_size_2, kernel_size_2, depth1, depth2])
    b_conv3 = bias_variable(1.0, [depth2])
    
    #S4 has no weights

    W_conv5 = weight_variable([kernel_size_2, kernel_size_2, depth2, depth3])
    b_conv5 = bias_variable(1.0, [depth3])

    #S6 has no weights

    W_conv7 = weight_variable([8*8*depth3, C5_units])
    b_conv7 = bias_variable(1.0, [C5_units])
    
    W_fc8 = weight_variable([C5_units, F6_units])
    b_fc8 = bias_variable(1.0, [F6_units])

    W_fc9 = weight_variable([F6_units, F7_units])
    b_fc9 = bias_variable(1.0, [F7_units])


    #Convolutions and max pooling
    conv = conv2d(x_image, W_conv1, 'SAME')
    hidden = tf.nn.relu(conv + b_conv1)

    max_pool = max_pool_2x2(hidden, kernel_size_1)
    hidden = tf.nn.relu(max_pool)

    conv = conv2d(hidden, W_conv3, 'SAME')
    hidden = tf.nn.relu(conv + b_conv3)

    max_pool = max_pool_2x2(hidden, kernel_size_2)
    hidden = tf.nn.relu(max_pool)

    conv = conv2d(hidden, W_conv5, 'SAME')
    hidden = tf.nn.relu(conv + b_conv5)

    max_pool = max_pool_2x2(hidden, kernel_size_2)
    hidden = tf.nn.relu(max_pool)

    shape = hidden.get_shape().as_list()
    reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
    hidden = tf.nn.relu(tf.matmul(reshape, W_conv7) + b_conv7)

    #Fully connected layers leading to logits
    fc = tf.matmul(hidden,W_fc8)
    hidden = tf.nn.relu(fc + b_fc8)
        
    fc = tf.matmul(hidden,W_fc9)
    output = fc + b_fc9

    return output


def train():
    image_batch_out, label_batch_out, filename_batch = net_input(if_random = False, if_training = True)

    image_batch_placeholder = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS])
    image_batch = tf.reshape(image_batch_out, (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS))

    label_batch_placeholder = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 3])
    label_offset = -tf.ones([BATCH_SIZE], dtype=tf.int64, name="label_batch_offset")
    label_batch_one_hot = tf.one_hot(tf.add(label_batch_out, label_offset), depth=3, on_value=1.0, off_value=0.0)


    logits_out = net_inference(image_batch_placeholder)
    loss = tf.losses.mean_squared_error(labels=label_batch_placeholder, predictions=logits_out)
    train_step = tf.train.GradientDescentOptimizer(0.00005).minimize(loss)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        file_writer = tf.summary.FileWriter("./logs", sess.graph)

        sess.run(tf.global_variables_initializer())
        saver.restore(sess, "output/checkpoint-train")
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess = sess)

        for i in range(10000):
            image_out, label_out, label_batch_one_hot_out, filename_out = sess.run([image_batch, label_batch_out, label_batch_one_hot, filename_batch])
            _, infer_out, loss_out = sess.run([train_step, logits_out, loss], feed_dict={image_batch_placeholder: image_out, label_batch_placeholder: label_batch_one_hot_out})
          
            if(i%25 == 0):
                print(image_out.shape)
                print("label_out: ")
                print(filename_out)
                print(label_out)
                print(label_batch_one_hot_out)
                print("infer_out: ")
                print(infer_out)
                saver.save(sess, "output/checkpoint-train")
                print("loss: ")
                print(loss_out)

            if(loss_out < 0.02):
            	break
        coord.request_stop()
        coord.join(threads)
        sess.close()


def evaluate():
    image_batch_out, label_batch_out, filename_batch = net_input(if_random = False, if_training = False)

    image_batch_placeholder = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS])
    image_batch = tf.reshape(image_batch_out, (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS))

    label_tensor_placeholder = tf.placeholder(tf.int64, shape=[BATCH_SIZE])
    label_offset = -tf.ones([BATCH_SIZE], dtype=tf.int64, name="label_batch_offset")
    label_batch = tf.add(label_batch_out, label_offset)

    logits_out = tf.reshape(net_inference(image_batch_placeholder), [BATCH_SIZE, 3])
    logits_batch = tf.to_int64(tf.arg_max(logits_out, dimension = 1))

    correct_prediction = tf.equal(logits_batch, label_tensor_placeholder)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess,  "output/checkpoint-train")
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess = sess)

        accuracy_accu = 0

        for i in range(50):
            image_out, label_out, filename_out = sess.run([image_batch, label_batch, filename_batch])

            accuracy_out, logits_batch_out = sess.run([accuracy, logits_batch], feed_dict={image_batch_placeholder: image_out, label_tensor_placeholder: label_out})
            accuracy_accu += accuracy_out
            print(i)
            print(image_out.shape)
            print("label_out: ")
            print(filename_out)
            print(label_out)
            print(logits_batch_out)

        print("Accuracy: ")
        print(accuracy_accu / 50)

        coord.request_stop()
        coord.join(threads)
        sess.close()

#############################################
#### UNCOMMENT TO FURTHER TRAIN THE MODEL####
#############################################
#train()
#############################################
#### UNCOMMENT TO FURTHER TRAIN THE MODEL####
#############################################

evaluate()
