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

if __name__ == "__main__":
     #bnn_model = BNN(num_dense_layer=2,num_neuron_in_dense_layer=5,num_neuron_output_layer=1)
     #dimacs_file_path=encode_network(bnn_model)
     #describe_network(bnn_model)

     #dimacs_file_path = 'output_final.cnf'
     #n_vars = 10 
     #CNF to BDD
     #cnf_formula = read_dimacs_file(dimacs_file_path)
     #output_file_path = 'output_bdd_info.txt'

     #bdd_compiler = BDD_Compiler(n_vars, cnf_formula)
     #bdd = bdd_compiler.compile(output_file=output_file_path)
          #bdd.print_info(n_vars)
        
     datafile = 'C:\\Users\\freun\Desktop\WS2\Masterarbeit\\from_Scratch\\training\\Other_Test\\data3.txt'
     with open(datafile, "r") as file:
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
     model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
     initial_weights = model.get_weights()
     model.save_weights("initial_weights.h5")
     history=model.fit(X_train, Y_train, epochs=250, batch_size=20, validation_split=0.2)
     test_loss, test_accuracy = model.evaluate(X_test, Y_test)
     print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

     model.save("BNN_model.h5")
     datafile = 'weights_after_training.h5'

     model.save_weights(datafile)
     describe_network(model)
     plt.figure(figsize=(12, 6))

    # Plot training & validation accuracy values
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
     plt.show()
     """
     output_layer_weights = model.layers[-1].get_weights()[0]
     binarized_weights = np.all(np.isin(output_layer_weights, [-1, 1]))
     if binarized_weights:
        print(f"The weights of layer  are binarized.")
     else:
        print(f"The weights of layer are NOT binarized.")
     output_layer_weights = model.layers[-1].get_weights()[0] 
     print(output_layer_weights)
     """


     """
     n_vars,num_clauses=get_num_variables_and_clauses_from_cnf(dimacs_file_path)
     ddbcsfi_instance = dDBCSFi_2(n_variables=n_vars, perceptron=bnn_model)

     beginning = time.monotonic()
     #cnff_name = 'output_final.cnf'
     #cnff_name = encode_network(bnn_model, cnff_name)
     read_and_print_file(dimacs_file_path)
     duration = time.monotonic() - beginning
     print("Time taken to create the formula:", seconds_separator(duration),"\n")

     beginning = time.monotonic()
     mgr = SddManager()
     print('Me ????????????????????')
     ssd_manager, node = mgr.from_cnf_file(bytes(dimacs_file_path, encoding='utf-8'))
     duration = time.monotonic() - beginning
     print("Time taken to create the SDD:", seconds_separator(duration),"\n")

     beginning = time.monotonic()
     The_circuit = dDBCSFi_2(n_vars, SDD=node) # 10 is the number of inputs features or neurons
     The_circuit.compile_bnn()
     duration = time.monotonic() - beginning
     print("Time taken to create the dDBCSFi(2):", seconds_separator(duration))
     beginning = time.monotonic()
     The_circuit.corroborate_equivalence(bnn_model, -1)
     duration = time.monotonic() - beginning
     print("Time taken to verify the equivalence of dDBCSFi(2) with the BNN:", seconds_separator(duration))
     print("\nThe circuit has", The_circuit.count_nodes(), "nodes") """
