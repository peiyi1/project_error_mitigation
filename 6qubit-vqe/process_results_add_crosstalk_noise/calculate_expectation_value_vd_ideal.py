import networkx as nx
import itertools
import numpy as np
from datetime import date
import os, glob, json
import sys

import yaml

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister,  IBMQ
from qiskit.providers.aer.noise import NoiseModel
from qiskit.compiler.transpiler import transpile
from qiskit.providers.aer.noise.errors import  depolarizing_error

import qiskit.quantum_info as qi

from qiskit.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import RealAmplitudes

module_path = os.path.abspath(os.path.join('../../utils4benchmark'))
#print(module_path)
if module_path not in sys.path:
    sys.path.append(module_path)
module_path = os.path.abspath(os.path.join('../../utilsVD'))
#print(module_path)
if module_path not in sys.path:
    sys.path.append(module_path)  
    
with open('../config.yaml') as file:
    configuration = yaml.load(file, Loader=yaml.FullLoader)

shots=configuration['shots']
test_qubits=configuration['test_qubits']
import pickle

# Load the noise model
with open('../../noise_model.pkl', 'rb') as f:
    noise_model = pickle.load(f)
# Load the noise model
with open('../../backend.pkl', 'rb') as f:
    backend = pickle.load(f)
    
paul_op = configuration['pauli_op']
coeffs_pauli_op = configuration['coeffs_pauli_op']

from qiskit.opflow import PauliSumOp
from qiskit.quantum_info import SparsePauliOp
# Create the SparsePauliOp object
sparse_op = SparsePauliOp(paul_op, coeffs_pauli_op)
cost_hamiltonian = PauliSumOp(sparse_op, coeff=1.0)
    
from qiskit import qpy

with open('../transpiled_circuit_list/circuit_list.qpy', "rb") as f:
    total_circuit = qpy.load(f)
    
circ_index=test_qubits*3+2    
trans_circuit=total_circuit[circ_index]
#trans_circuit.count_ops()

from benchmark_utils import obtain_record_qubits_index_dict,obtain_record_qubits_index
record_qubits_index=obtain_record_qubits_index(obtain_record_qubits_index_dict(trans_circuit))
trans_circuit.remove_final_measurements(inplace=True)
trans_circuit.save_density_matrix(qubits=record_qubits_index)


from qiskit.providers.aer import AerSimulator
sim = AerSimulator()
noise_sim = AerSimulator(noise_model=noise_model)


job_result=noise_sim.run(trans_circuit,shots=shots).result()
job_data=job_result.data()
density_matrix_ori=job_data.get('density_matrix')
#print(density_matrix)

rho_ori=np.asarray(density_matrix_ori)
#print(rho_ori)

#print(density_matrix_ori.trace())
#print(density_matrix_ori.purity())

from vd_utils import obtain_matrix_z,obtain_matrix_i,tensor_product
interest_observable=tensor_product(obtain_matrix_i(),obtain_matrix_i(),obtain_matrix_i(),obtain_matrix_i(),obtain_matrix_z(),obtain_matrix_z())

ratio_1copy_list=[]
denominator_1copy=np.trace(rho_ori@rho_ori).real
print(denominator_1copy)
numerator_1copy=np.trace(interest_observable@rho_ori@rho_ori).real
print(numerator_1copy)
ratio_1copy=numerator_1copy/denominator_1copy

#ratio_1copy = compute_ratio(ratio_1copy)
ratio_1copy_list.append(ratio_1copy)
print(ratio_1copy)

interest_observable=tensor_product(obtain_matrix_i(),obtain_matrix_i(),obtain_matrix_i(),obtain_matrix_z(),obtain_matrix_z(),obtain_matrix_i())
denominator_1copy=np.trace(rho_ori@rho_ori).real
print(denominator_1copy)
numerator_1copy=np.trace(interest_observable@rho_ori@rho_ori).real
print(numerator_1copy)
ratio_1copy=numerator_1copy/denominator_1copy

#ratio_1copy = compute_ratio(ratio_1copy)
ratio_1copy_list.append(ratio_1copy)
print(ratio_1copy)

interest_observable=tensor_product(obtain_matrix_i(),obtain_matrix_i(),obtain_matrix_z(),obtain_matrix_z(),obtain_matrix_i(),obtain_matrix_i())
denominator_1copy=np.trace(rho_ori@rho_ori).real
print(denominator_1copy)
numerator_1copy=np.trace(interest_observable@rho_ori@rho_ori).real
print(numerator_1copy)
ratio_1copy=numerator_1copy/denominator_1copy

#ratio_1copy = compute_ratio(ratio_1copy)
ratio_1copy_list.append(ratio_1copy)
print(ratio_1copy)

interest_observable=tensor_product(obtain_matrix_i(),obtain_matrix_z(),obtain_matrix_z(),obtain_matrix_i(),obtain_matrix_i(),obtain_matrix_i())
denominator_1copy=np.trace(rho_ori@rho_ori).real
print(denominator_1copy)
numerator_1copy=np.trace(interest_observable@rho_ori@rho_ori).real
print(numerator_1copy)
ratio_1copy=numerator_1copy/denominator_1copy

#ratio_1copy = compute_ratio(ratio_1copy)
ratio_1copy_list.append(ratio_1copy)
print(ratio_1copy)

interest_observable=tensor_product(obtain_matrix_z(),obtain_matrix_z(),obtain_matrix_i(),obtain_matrix_i(),obtain_matrix_i(),obtain_matrix_i())
denominator_1copy=np.trace(rho_ori@rho_ori).real
print(denominator_1copy)
numerator_1copy=np.trace(interest_observable@rho_ori@rho_ori).real
print(numerator_1copy)
ratio_1copy=numerator_1copy/denominator_1copy

#ratio_1copy = compute_ratio(ratio_1copy)
ratio_1copy_list.append(ratio_1copy)
print(ratio_1copy)

print(ratio_1copy_list)

offset=configuration['offset']
exp_ideal_vd=0.5*(sum(i for i in ratio_1copy_list))+offset
print(exp_ideal_vd)
with open('../saved_results/exp_vd_ideal.json', 'w') as fl:
        json.dump(exp_ideal_vd,fl)
