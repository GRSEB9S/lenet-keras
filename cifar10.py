def unpickle(file):
	import pickle
	with open(file, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
	return dict

def prepare_pixel(im):
	import numpy as np
	assert len(im) == 3072
	r = im[0:1024]; r = np.reshape(r, [32, 32, 1])
	g = im[1024:2048]; g = np.reshape(g, [32, 32, 1])
	b = im[2048:3072]; b = np.reshape(b, [32, 32, 1])
	return np.concatenate([r, g, b], -1)

def prepare_data_batch(data_batch):
	import numpy as np
	assert data_batch.shape[1:] == (3072,)
	p_data_batch = np.zeros((data_batch.shape[0], 32, 32, 3))
	for i in range(data_batch.shape[0]):
		p_data_batch[i] = prepare_pixel(data_batch[i])
	return p_data_batch

def show_im(im):
	import matplotlib.pyplot as plt
	plt.imshow(im)
	plt.show()

def create_one_hot(values):
	import numpy as np
	return np.eye(values.max() + 1)[values]

# return (X_train, y_train), (X_test, y_test)
def cifar10_data_split(data_dir):
	import numpy as np
	print('Started Loading data')
	train_data_dict = []
	for i in range(5):
		train_data_dict.append(unpickle(data_dir + 'data_batch_%d' % (i + 1)))
	test_data_dict = unpickle(data_dir + 'test_batch')
	X_train, y_train = train_data_dict[0].get(b'data'), train_data_dict[0].get(b'labels')
	for i in range(1, 5):
		X_train = np.concatenate((X_train, train_data_dict[i].get(b'data')))
		y_train = np.concatenate((y_train, train_data_dict[i].get(b'labels')))
	assert X_train.shape[1:] == (3072,) and y_train.shape == (50000,)
	print('Finishied loading data')
	X_train = prepare_data_batch(X_train)
	y_train = create_one_hot(y_train)
	assert X_train.shape[1:] == (32, 32, 3,)
	X_test, y_test = test_data_dict.get(b'data'), test_data_dict.get(b'labels')
	X_test = prepare_data_batch(X_test)
	y_test = create_one_hot(np.array(y_test))
	return (X_train, y_train), (X_test, y_test)		
