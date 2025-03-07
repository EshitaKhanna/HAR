import numpy as np

num_samples = 1000  # num of samples
seq_len = 120       # sequence length
feature_num = 6     # num of features 

# data
dummy_data = np.random.rand(num_samples, seq_len, feature_num).astype(np.float32)

# labels
# shape: (num_samples, 3)
# 6 activity labels (0-5) and 10 users 
dummy_labels = np.zeros((num_samples, 3), dtype=np.int32)
dummy_labels[:, 0] = np.random.randint(0, 6, size=(num_samples,))   # activity labels
dummy_labels[:, 1] = np.random.randint(0, 10, size=(num_samples,))  # user labels
dummy_labels[:, 2] = 4 # domain label  

# eeshape to(num_samples, 1, 3)
dummy_labels = dummy_labels.reshape(num_samples, 1, 3)

np.save('dataset/dummy/data_20_120.npy', dummy_data)
np.save('dataset/dummy/label_20_120.npy', dummy_labels)

data = np.load('dataset/dummy/data_20_120.npy')
labels =  np.load('dataset/dummy/label_20_120.npy')
print("Data shape:", data.shape)
print("Labels shape:", labels.shape)

print("Dummy dataset has been created.")