import networkx as nx
import itertools
import numpy as np
from datetime import date
import os, glob, json
import sys
import yaml
import pickle
from qiskit import QuantumCircuit,IBMQ
from qiskit.providers.aer import AerSimulator
from qiskit import qpy

# Load the noise model
with open('../../noise_model.pkl', 'rb') as f:
    noise_model = pickle.load(f)
# Load the noise model
with open('../../backend.pkl', 'rb') as f:
    backend = pickle.load(f)
        
with open('../config.yaml') as file:
    configuration = yaml.load(file, Loader=yaml.FullLoader)

shots=configuration['shots']
noise_scale_range = configuration['noise_scale_range']
pauli_string_list=configuration['pauli_op']

sim = AerSimulator()
noise_sim = AerSimulator(noise_model=noise_model)
#create folder to save result
if not os.path.exists('../data_experimental/execute_circuit'):
    os.makedirs('../data_experimental/execute_circuit')

#execute the circuit without crosstalk noise
with open('../transpiled_circuit_list/circuit_list.qpy', "rb") as f:
    total_circuit = qpy.load(f)

simulator_result = sim.run(total_circuit, shots=shots).result()
simulator_counts = simulator_result.get_counts()

with open('../data_experimental/execute_circuit/ideal_counts.json', 'w') as fl:
    json.dump(simulator_counts,fl)

simulator_result = noise_sim.run(total_circuit, shots=shots).result()
simulator_counts = simulator_result.get_counts()

with open('../data_experimental/execute_circuit/noisy_counts.json', 'w') as fl:
    json.dump(simulator_counts,fl)
    
#execute the circuit with crosstalk noise
with open('../transpiled_circuit_list/circuit_list_add_rzz.qpy', "rb") as f:
    total_circuit_add_rzz_transpiled = qpy.load(f)  
    
simulator_result = noise_sim.run(total_circuit_add_rzz_transpiled, shots=shots).result()
simulator_counts = simulator_result.get_counts()

with open('../data_experimental/execute_circuit/noisy_counts_zz_crosstalk.json', 'w') as fl:
    json.dump(simulator_counts,fl)
