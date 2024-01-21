from bnn_to_cnf import *
from bnn import *
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import larq as larq


file = 'C:\\Users\\freun\Desktop\WS2\Masterarbeit\\from_Scratch\\training\\Other_Test\\data3.txt'
with open(file, "r") as file:
	lines = file.readlines()

X_train_sequences = []
Y_train = []

for line in lines:
	 parts = line.strip().split()


	 left_data = [int(bit) for bit in parts[0]]
	 right_data = [int(bit) for bit in parts[1]]

	 X_train_sequences.append(left_data)
	 Y_train.append(right_data)


X_train_sequences = np.array(X_train_sequences, dtype=np.float32)
Y_train = np.array(Y_train, dtype=np.float32)


X_train_padded = pad_sequences(X_train_sequences, padding='post')

X_train, X_test, Y_train, Y_test = train_test_split(X_train_padded, Y_train, test_size=0.2, random_state=42)

print("X_train shape:", X_train.shape)
print("Y_train shape:", Y_train.shape)

opt=larq.optimizers.Bop(threshold=1e-08, gamma=0.0001, name="Bop")

model = BNN(num_neuron_in_hidden_dense_layer=X_train.shape[1], num_neuron_output_layer=Y_train.shape[1],input_dim=X_train.shape[1], output_dim=Y_train.shape[1])
trained_weights ="weights_after_training.h5"
model.load_weights(trained_weights)
print('Weights loaded successfully !!! ')

print('Encoding Process BNN----> CNF started')
dismacs_from_loaded_weights=encode_network(model)
