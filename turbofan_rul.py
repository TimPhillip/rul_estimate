import numpy as np
import os
import tensorflow as tf
import sklearn.decomposition
import argparse
import pickle
from rul_estimate.lstm_ed import LSTMED
from rul_estimate.health_index import TargetHI, TargetHIMatcher
from matplotlib import pyplot as plt

num_principal_components = 3
window_size = 20
batch_size = 250
num_epochs = 5000
num_lstm_units = 30

def load_file(filename, first_window_only = False):
    data = np.genfromtxt(filename.decode())

    # normalize the data
    sensor_mean = np.mean(data[:,2:], axis= 0)
    sensor_std = np.std(data[:,2:],axis=0)
    np.place(sensor_std, sensor_std == 0.0,vals= [1.0]) # remove constants from sensor readings
    data[:,2:] = (data[:,2:] - sensor_mean) / sensor_std

    # perform PCA analysis
    pca = sklearn.decomposition.PCA(n_components= num_principal_components)
    reduced_sensor_readings = pca.fit_transform(data[:,2:])
    data = np.concatenate((data[:,:2], reduced_sensor_readings), axis= 1)

    # Time windows
    def split_generator(data):
        start = 0
        current = 1
        for i in range(np.shape(data)[0]):
            if current == data[i,0]:
                continue
            else:
                yield data[start:i,:]
                current = data[i,0]
                start = i

    if first_window_only:
        windows = [block[:window_size] for block in split_generator(data)]
    else:
        windows = [block[l:l +window_size] for block in split_generator(data) for l in range(np.shape(block)[0] - window_size + 1)]

    # Save the PCA basis
    np.savez_compressed("rul_checkpoints/pca.npz",**pca.get_params())

    return np.asarray(windows,np.float32)

def tf_post_process(data):
    data.set_shape([window_size,2 + num_principal_components])
    return data[:,:2], data[:,2:]


def generate_dataset(filename, first_window_only, num_epochs, batch_size):

    dataset : tf.data.Dataset = tf.data.Dataset().from_tensor_slices([filename])
    dataset : tf.data.Dataset = dataset.map(lambda file: tuple(tf.py_func(load_file, [file, first_window_only], [tf.float32])))
    dataset : tf.data.Dataset = dataset.flat_map(lambda block: tf.data.Dataset().from_tensor_slices(block))
    dataset: tf.data.Dataset = dataset.cache()
    dataset: tf.data.Dataset = dataset.map(tf_post_process,num_parallel_calls= 8)
    dataset : tf.data.Dataset = dataset.shuffle(10000)
    dataset : tf.data.Dataset = dataset.repeat(num_epochs)
    dataset: tf.data.Dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)
    dataset: tf.data.Dataset = dataset.prefetch(1)

    return dataset


def training():

    current_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_path, "data/turbofan/train_FD001.txt")

    if not os.path.exists(data_path):
        print("NASA turbofan dataset not found.")
        exit(-1)

    _, next_element = generate_dataset(filename=data_path,
                                       first_window_only= True,
                                       num_epochs= num_epochs,
                                       batch_size= batch_size).make_one_shot_iterator().get_next()

    # init the model
    lstm_ed = LSTMED(num_units=num_lstm_units,
                     batch_size=batch_size,
                     input=next_element,
                     input_size=num_principal_components)

    # init the optimizer
    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.train.exponential_decay(1e-3, global_step, decay_steps=100, decay_rate=0.99)
    learning_rate = tf.maximum(learning_rate, 1e-5)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(lstm_ed.loss, global_step)

    tf.summary.scalar('Loss', lstm_ed.loss)
    tf.summary.scalar('learning_rate', learning_rate)


    with tf.train.MonitoredTrainingSession(checkpoint_dir='rul_checkpoints') as sess:

        print("LSTM Autoencoder Training started.")

        while not sess.should_stop():
            _, step, loss = sess.run([train_op, global_step, lstm_ed.loss])

            if step % 100 == 0:
                print("[%i] loss= %f" % (step, loss))


def compute_target_hi():

    # check if pickle for Target HI exists
    current_path = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(current_path, "../rul_checkpoints/target_hi.pickle")
    data_path = os.path.join(current_path, "data/turbofan/train_FD001.txt")

    if os.path.exists(out_path):
        target_hi = pickle.load(open(out_path))
    else:
        # define the tensorflow model
        next_indices, next_element = generate_dataset(filename= data_path,
                                                      first_window_only= False,
                                                      num_epochs= 1,
                                                      batch_size= batch_size).make_one_shot_iterator().get_next()

        lstm_ed = LSTMED(num_units= num_lstm_units,
                         batch_size= batch_size,
                         input= next_element,
                         input_size= num_principal_components)

        saver = tf.train.Saver()

        target_hi_matcher = TargetHIMatcher()

        # Restore latest TF checkpoint
        with tf.train.MonitoredSession() as sess:

            cp = tf.train.latest_checkpoint("rul_checkpoints")
            saver.restore(sess, cp)

            # iterate over data once
            while not sess.should_stop():

                indices, errors = sess.run([next_indices, lstm_ed.point_reconstruction_error])

                indices = np.reshape(indices, [-1,2]).astype(dtype= np.int32)
                errors = np.reshape(errors, [-1])

                # tell TargetHIMatcher
                target_hi_matcher.tell(indices,errors)

        # finalize matcher
        target_hi = target_hi_matcher.finalize()

        # pickle target functions
        #pickle.dump(target_hi, file= open(out_path, mode= 'wb'))

    # plot some target functions
    #for i, hi in enumerate(target_hi):
    #    print("[%i] %i" % (i, hi.get_lifetime()))

    visualize_everything(target_hi)


def visualize_everything(target_hi):
    h = target_hi[0].health()
    plt.plot(range(len(h)), h)
    plt.show()

    bins = np.zeros([100, len(target_hi)])
    for h_i, h in enumerate(target_hi):
        lifetime = h.get_lifetime()
        h_v = h.health()
        for v_i in range(lifetime):
            bins[int(v_i * 100 / lifetime), h_i] = h_v[v_i]

    bin_mean = np.mean(bins, axis=-1)
    bin_std = np.std(bins, axis=-1)

    f = plt.figure()
    plt.errorbar(x=range(100), y=bin_mean, yerr=bin_std, fmt='-', ecolor='g', capthick=2, capsize=2, errorevery=3)
    plt.xlabel('Lifetime in %')
    plt.ylabel('HI')
    plt.title('Reconstruction Error Based HI')
    plt.show()
    f.savefig("normalized_hi.pdf")




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_ae',action="store_true")
    parser.add_argument('--target_hi', action="store_true")

    args = parser.parse_args()

    if args.train_ae:
        training()

    tf.reset_default_graph()

    if args.target_hi:
        compute_target_hi()




