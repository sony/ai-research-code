"""
    The CIFAR-10 dataset
    http://www.cs.toronto.edu/~kriz/cifar.html

    Description
    -------------
    The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes,
    with 6000 images per class. There are 50000 training images and 10000 test
    images.


    Reference
    ---------
    ..  [1] Learning Multiple Layers of Features from Tiny Images, Alex
        Krizhevsky, 2009.
"""
from os.path import os
import numpy as np
import contextlib
import pickle


class DataIterator(object):

    def __init__(self, batch_size, mean_norm=True, var_norm=True, augmented_shift=False,
                 augmented_flip=False, shuffle=True, max_size=None, val=False, rng=None):
        """
        Parameters
        ----------
        batch_size : batch_size
        mean_norm : if True images are normalized to zero mean 
        var_norm : if True images are normalized to unit variance
        augmented_shift : if True, dataset is augmented by shifting existing images (dataset size x5)
        augmented_flip : if True, dataset is augmented by mirroring existing images (dataset size x2)
        shuffle : shuffle data to construct batches
        val : False for training data and True for validation data
        rng: random seed
        """
        if rng is None:
            rng = np.random.RandomState(313)
        if not isinstance(rng, np.random.RandomState):
            rng = np.random.RandomState(rng)
        self.rng = rng
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.val = val
        self.augmented_shift = augmented_shift
        self.augmented_flip = augmented_flip
        save_path = get_data_home()
        # Loading CIFAR dataset
        url = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
        dir_data = 'cifar-10-batches-py'
        path_tar = download_data(url, save_path=save_path)
        paths_train = os.path.join(save_path, dir_data, 'data_batch_{}')
        paths_train = [paths_train.format(i + 1) for i in range(5)]
        path_test = os.path.join(save_path, dir_data, 'test_batch')
        path_meta = os.path.join(save_path, dir_data, 'batches.meta')
        if not os.path.isfile(path_test):
            import tarfile

            t = tarfile.open(path_tar)
            t.extractall(save_path)
            t.close()

        n_feat = 3 * 32 * 32

        def _transpose_rasterized_images(data):
                return data.reshape(
                    (data.shape[0], 3, 32, 32)
                ).transpose((0, 2, 3, 1)).reshape(-1, n_feat)

        if self.val: # validation set
            dataset = pickle.load(open(path_test, 'rb'), encoding='bytes')
            data = _transpose_rasterized_images(dataset[b'data'])
            labels = np.array(dataset[b'labels'], dtype='uint8')
        else: # training set
            n_train_sample = 50000
            data = np.zeros((n_train_sample, n_feat), dtype='uint8')
            labels = np.zeros((n_train_sample,), dtype='uint8')
            idx = 0
            for path in paths_train:
                dataset = pickle.load(open(path, 'rb'), encoding='bytes')
                n_sample_one = dataset[b'data'].shape[0]
                data[idx:idx + n_sample_one, :] = \
                    _transpose_rasterized_images(dataset[b'data'])
                labels[idx:idx + n_sample_one] = dataset[b'labels']
                idx += n_sample_one
        meta = pickle.load(open(path_meta, 'rb'), encoding='bytes')
        label_names = meta[b'label_names']

        # Reshape
        data = data.reshape(-1, 32, 32, 3).transpose((0, 3, 1, 2)).astype(np.float32)

        """
        if mean_norm:
            mean = np.array([125.3, 123.0, 113.9], dtype=np.float32).reshape(1,3,1,1)
            data = data - mean

        if var_norm:
            std = np.array([63.0, 62.1, 66.7], dtype=np.float32).reshape(1,3,1,1)
            data = data / std
        """
        data = (data / 255.0 - 0.5) * 2.0

        self.labels = labels.reshape(-1, 1)
        self.images = data

        # if only smaller number of images is to be used
        if max_size:
            self._downsize(max_size)

        self._reset()

    def next(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        end = self.current + batch_size
        images = self.images[self.idxs[self.current:end]].copy()
        labels = self.labels[self.idxs[self.current:end]].copy()
        self.current = end
        if self.current + batch_size > self.idxs.size:
            self._reset()

        if self.augmented_shift:
            for i in range(batch_size):
                padded = np.pad(images[i], ((0, 0), (4, 4), (4, 4)), mode='constant')
                corner = (np.random.randint(9), np.random.randint(9))
                images[i] = padded[:, corner[0]:corner[0]+32, corner[1]:corner[1]+32]

        if self.augmented_flip:
            for i in range(batch_size):
                if np.random.rand() < 0.5:
                    images[i] = images[i, :, :, ::-1]

        return images, labels

    def _downsize(self, max_size):
        mode = 'validation' if self.val else 'training'
        idxs = self.rng.permutation(self.labels.size)
        self.labels = self.labels[idxs[:max_size]]
        self.images = self.images[idxs[:max_size]]

    def _reset(self):
        mode = 'validation' if self.val else 'training'
        #print("Initializing CIFAR10 epoch ({})...".format(mode))
        if self.shuffle:
            self.idxs = self.rng.permutation(self.labels.size)
        else:
            self.idxs = np.arange(self.labels.size)
        self.current = 0


def get_data_home():
    import os
    d = os.path.expanduser("~/work/.nnabla_data")
    if not os.path.isdir(d):
        os.makedirs(d)
    return d


@contextlib.contextmanager
def remove_file(*files):
    import os
    yield
    for f in files:
        try:
            os.remove(f)
        except:
            pass


@contextlib.contextmanager
def urlopen(url):
    from six.moves import urllib
    response = urllib.request.urlopen(url)
    yield response
    response.close()


def download_data(url, file_name=None, save_path=None, block_size=8192):
    import shutil
    import os
    from six.moves import urllib
    if save_path is None:
        save_path = get_data_home()
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    if file_name is None:
        file_name = urllib.parse.urlsplit(url)[2].split("/")[-1]
    data_file = os.path.join(save_path, file_name)
    if os.path.exists(data_file):
        return data_file
    data_file_tmp = data_file + '.downloading'

    with remove_file(data_file_tmp):
        with open(data_file_tmp, 'wb') as f, urlopen(url) as response:
            content_length = int(response.info().get("Content-Length"))
            prev_progress = 0
            print("Start download: {}.".format(url))
            for i in range(0, content_length, block_size):
                read_end = min(i + block_size, content_length)
                read_size = read_end - i
                buf = response.read(read_size)
                if not buf:
                    raise ValueError()
                f.write(buf)
                progress = int(i / (content_length / 10))
                if progress > prev_progress:
                    prev_progress = progress
                    print("{}% downloaded.".format(int(progress * 10)))
            print("[100%] downloaded.")
        shutil.move(data_file_tmp, data_file)
    return data_file
