from tensorflow.keras.utils import to_categorical
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow.keras.callbacks import EarlyStopping
from larq.layers import QuantDense
from keras.callbacks import LearningRateScheduler
from sklearn.metrics import accuracy_score
from pysdd.sdd import SddManager
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Sequential
import larq as larq
from bnn import *
#from bnn_to_cnf import *
#from cnf_to_bdd import *
#from bnn_to_sdd import *
from  pysat.solvers import Glucose3
import itertools
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import  optimizers
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import OneHotEncoder

def read_and_print_file(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                print(line.strip())  # Strip to remove trailing newline characters
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
def get_num_variables_and_clauses_from_cnf(cnf_filename):
    num_variables, num_clauses = None, None
    
    with open(cnf_filename, 'r') as cnf_file:
        for line in cnf_file:
           
            if line.startswith('p cnf'):
               
                _, _, num_variables, num_clauses = line.split()
                num_variables, num_clauses = int(num_variables), int(num_clauses)
                break  
    print('Num Vars :', num_variables)
    print('num clauses :', num_clauses)
                
    return num_variables, num_clauses
def replace_question_mark(sequence):
    return [1 if bit == '?' else int(bit) for bit in sequence]


if __name__ == "__main__":
          
     datafile = 'C:\\Users\\freun\Desktop\WS2\Masterarbeit\\from_Scratch\\training\\Other_Test\\best_params_45_prozent_2\\data3.txt'
     data = np.loadtxt(datafile, dtype=str)  

     X = data[:, 0] 
     Y = data[:, 1] 

     X = np.core.defchararray.replace(X, '?', '1')

     df = pd.DataFrame({'X': X, 'Y': Y})

     X = np.array([list(map(int, binary_string)) for binary_string in df['X']])

     y = np.array([int(label, 2) for label in df['Y']])

     y_one_hot = to_categorical(y, num_classes=16)
     print('X :', X[:10], X.shape)
     print('Y encoded :', y_one_hot[:10], y_one_hot.shape)


     X_train, X_test, Y_train, Y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

     print('X_train :', X_train[:10])
     print('Y_train :',Y_train[:10])

     print("X_train shape:", X_train.shape)
     print("Y_train shape:", Y_train.shape)
     print('Unique :', len(np.unique(Y_train,axis=0)))

     opt=larq.optimizers.Bop(threshold=1e-08, gamma=0.0001, name="Bop")
     model = BNN(num_neuron_in_hidden_dense_layer=18, num_neuron_output_layer=len(np.unique(Y_train,axis=0)), input_dim=18, output_dim=len(np.unique(Y_train,axis=0)))

     model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
     initial_weights = model.get_weights()
     model.save_weights("initial_weights.h5")
     early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
     #Y_train_encoded = np.argmax(Y_train, axis=1)
     #Y_test_encoded = np.argmax(Y_test_encoded, axis=1)

     history = model.fit(X_train, Y_train, epochs=300, batch_size=5, validation_data=(X_test, Y_test), callbacks=[early_stopping])

     test_loss, test_accuracy = model.evaluate(X_test, Y_test)
     print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
     predictions = model.predict(X_test)

     # Convert the predictions to binary values (assuming a threshold of 0.5)
     binary_predictions = (predictions >= 0.5).astype(int)
     precision = precision_score(Y_test, binary_predictions, average='micro')
     recall = recall_score(Y_test, binary_predictions, average='micro')
     f1 = f1_score(Y_test, binary_predictions, average='micro')
     print(f'Precision: {precision:.4f}')
     print(f'Recall: {recall:.4f}')
     print(f'F1 Score: {f1:.4f}')
     correct_predictions = np.sum(np.all(binary_predictions == Y_test, axis=1))
     total_samples = len(Y_test)
     print(f'Correct Predictions: {correct_predictions} out of {total_samples}')

     # Display the predictions
     print("Predictions:")
     print(binary_predictions)
     print('True values')
     print(Y_test)
     print('X_train')
     print(X_train[:10])
     

     model.save("BNN_model.h5")
     datafile = 'weights_after_training.h5'

     model.save_weights(datafile)
     lq.models.summary(model)
     describe_network(model)
     plt.figure(figsize=(12, 6))

    # Plot training & validation accuracy values
     """
     plt.subplot(1, 2, 1)
     plt.plot(history.history['accuracy'])
     plt.plot(history.history['val_accuracy'])
     plt.title('Model accuracy')
     plt.ylabel('Accuracy')
     plt.xlabel('Epoch')
     plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss values
     plt.subplot(1, 2, 2)
     plt.plot(history.history['loss'])
     plt.plot(history.history['val_loss'])
     plt.title('Model loss')
     plt.ylabel('Loss')
     plt.xlabel('Epoch')
     plt.legend(['Train', 'Validation'], loc='upper left')

     plt.tight_layout()
     plt.show()"""