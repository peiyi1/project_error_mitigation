import networkx as nx
import itertools
import numpy as np
from datetime import date
import os, glob, json
import sys
import json

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister,  IBMQ
from qiskit.providers.aer.noise import NoiseModel
from qiskit.compiler.transpiler import transpile
from qiskit.providers.aer.noise.errors import  depolarizing_error

import qiskit.quantum_info as qi

from qiskit.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import RealAmplitudes

module_path = os.path.abspath(os.path.join('../../../utilsVD'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
import utils_postprocessing as post

import yaml

with open('../../config.yaml') as file:
    configuration = yaml.load(file, Loader=yaml.FullLoader)
with open('./pauli_op_index.yaml') as file:
    config_pauli_op_index = yaml.load(file, Loader=yaml.FullLoader)
pauli_op_index=config_pauli_op_index['pauli_op_index']
print(pauli_op_index)
pauli_string=configuration['pauli_op'][pauli_op_index]

test_qubits=configuration['test_qubits']

# Get the argument provided on the command line (in this case, the second element of sys.argv)
argument = sys.argv[1]
qbit=int(argument)

print(qbit)

new_trace_list_dict={}
basis_list=['X','Y','Z']
for basis in basis_list:
    with open('../../data_experimental/original_circuit' + '/trace_list_measure_one_zz_crosstalk_'+ basis +'.json', 'r') as fl:
        new_trace_list_dict[basis]=json.load(fl)
trace_list=[]
for basis in basis_list:
    trace_list.append(new_trace_list_dict[basis][qbit])
# Insert the element at the front
trace_list.insert(0, 1)
rho=post.obtain_density_matrix(trace_list)
rho_two_copy=np.kron(rho, rho)

from vd_utils import obtain_matrix1, obtain_matrix2, obtain_matrix_ii, obtain_matrix_swap,dagger
#denominator
diag=obtain_matrix2()@np.asarray(rho_two_copy)@dagger(obtain_matrix2())

diag_elements=np.diag(diag)

denominator_recombined_dist={'00':diag_elements[0].real, '01':diag_elements[1].real, '10':diag_elements[2].real, '11':diag_elements[3].real}
#numerator
pauli_string_reverse=pauli_string[::-1]
diag_matrix=obtain_matrix2()
if(pauli_string_reverse[qbit]=='Z'):
    diag_matrix=obtain_matrix1()
diag=diag_matrix@np.asarray(rho_two_copy)@dagger(diag_matrix)
diag_elements=np.diag(diag)

numerator_recombined_dist={'00':diag_elements[0].real, '01':diag_elements[1].real, '10':diag_elements[2].real, '11':diag_elements[3].real}
if not os.path.exists('../data_experimental/vd_cutting_measure_one_pauli_op_' +  str(pauli_op_index)):
    os.makedirs('../data_experimental/vd_cutting_measure_one_pauli_op_' +  str(pauli_op_index))
qbit=str(qbit)
denominator_recombined_dist=json.dumps(denominator_recombined_dist)
with open('../data_experimental/vd_cutting_measure_one_pauli_op_' +  str(pauli_op_index)+ '/denominator_recombined_dist_check_' + qbit +' .json', 'w') as fl:
    fl.write(denominator_recombined_dist)
    
numerator_recombined_dist=json.dumps(numerator_recombined_dist)
with open('../data_experimental/vd_cutting_measure_one_pauli_op_' +  str(pauli_op_index) + '/numerator_recombined_dist_check_' + qbit +' .json', 'w') as fl:
    fl.write(numerator_recombined_dist)
print(denominator_recombined_dist)
print(numerator_recombined_dist)