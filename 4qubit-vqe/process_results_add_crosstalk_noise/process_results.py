import networkx as nx
import itertools
import numpy as np
from datetime import date
import os, glob, json
import sys

module_path = os.path.abspath(os.path.join('../../utilsVD'))
if module_path not in sys.path:
    sys.path.append(module_path)

module_path = os.path.abspath(os.path.join('../../utils4benchmark'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
import utils_postprocessing as post

import yaml

with open('../config.yaml') as file:
    configuration = yaml.load(file, Loader=yaml.FullLoader)

shots=configuration['shots']
test_qubits=configuration['test_qubits']
paul_op = configuration['pauli_op']
coeffs_pauli_op = configuration['coeffs_pauli_op']
pauli_string_list=configuration['pauli_op']
noise_scale_range = configuration['noise_scale_range']

from qiskit.opflow import PauliSumOp
from qiskit.quantum_info import SparsePauliOp
# Create the SparsePauliOp object
sparse_op = SparsePauliOp(paul_op, coeffs_pauli_op)
cost_hamiltonian = PauliSumOp(sparse_op, coeff=1.0)

with open('../data_experimental/execute_circuit/noisy_counts_zz_crosstalk.json', 'r') as fl:
    counts=json.load(fl)
    
list_counts_numerator = counts[(3*(test_qubits+1)):(len(pauli_string_list)*len(noise_scale_range)+(3*(test_qubits+1)))]
list_counts_denominator = counts[(len(pauli_string_list)*len(noise_scale_range)+(3*(test_qubits+1))):((len(pauli_string_list)+1)*len(noise_scale_range)+(3*(test_qubits+1)))]

#create folder to save result
if not os.path.exists('../data_experimental/extrapolation'):
    os.makedirs('../data_experimental/extrapolation')
if not os.path.exists('../data_experimental/original_circuit'):
    os.makedirs('../data_experimental/original_circuit')

    
with open('../data_experimental/extrapolation/list_counts_numerator_zz_crosstalk.json', 'w') as fl:
    json.dump(list_counts_numerator,fl)
with open('../data_experimental/extrapolation/list_counts_denominator_zz_crosstalk.json', 'w') as fl:
    json.dump(list_counts_denominator,fl)    
    
new_counts=counts[0:(3*(test_qubits+1))]


from benchmark_utils import compute_expectation_from_hamiltonian
noise_exp=compute_expectation_from_hamiltonian(new_counts[-1],cost_hamiltonian).real
with open('../data_experimental/original_circuit' + '/noise_exp_zz_crosstalk.json', 'w') as fl:
    json.dump(noise_exp,fl)
    
from vd_utils import add_cross_talk_noise

half_test_qubits=int(0.5*test_qubits)
output=add_cross_talk_noise(new_counts[-1],0)

for i in range(1,half_test_qubits):
    #print(i)
    output=add_cross_talk_noise(output,i*2)    
noise_exp_zz_crosstalk_measurement_crosstalk=compute_expectation_from_hamiltonian(output,cost_hamiltonian).real
with open('../data_experimental/original_circuit' + '/noise_exp_zz_crosstalk_measurement_crosstalk.json', 'w') as fl:
    json.dump(noise_exp_zz_crosstalk_measurement_crosstalk,fl)
    
#measure all qubit
counts=new_counts#_ideal

basis='Z'
counts_index=-1

trace_list=[]
for i in range(test_qubits):
    bit_wise_dist=(post.collapse_bit_distribution(counts[counts_index],i))
    #print(bit_wise_dist)
    trace=(bit_wise_dist['0']-bit_wise_dist['1'])/shots
    trace_list.append(trace)

new_trace_list=trace_list[::-1]
#print(new_trace_list)
with open('../data_experimental/original_circuit' + '/trace_list_measure_all_zz_crosstalk_'+ basis +'.json', 'w') as fl:
    json.dump(new_trace_list,fl)
    
basis='Y'
counts_index=-2

trace_list=[]
for i in range(test_qubits):
    bit_wise_dist=(post.collapse_bit_distribution(counts[counts_index],i))
    #print(bit_wise_dist)
    trace=(bit_wise_dist['0']-bit_wise_dist['1'])/shots
    trace_list.append(trace)
    
new_trace_list=trace_list[::-1]
#print(new_trace_list)
with open('../data_experimental/original_circuit' + '/trace_list_measure_all_zz_crosstalk_'+ basis +'.json', 'w') as fl:
    json.dump(new_trace_list,fl)
    
basis='X'
counts_index=-3


trace_list=[]
for i in range(test_qubits):
    bit_wise_dist=(post.collapse_bit_distribution(counts[counts_index],i))
    #print(bit_wise_dist)
    trace=(bit_wise_dist['0']-bit_wise_dist['1'])/shots
    trace_list.append(trace)
    
new_trace_list=trace_list[::-1]
#print(new_trace_list)
with open('../data_experimental/original_circuit' + '/trace_list_measure_all_zz_crosstalk_'+ basis +'.json', 'w') as fl:
    json.dump(new_trace_list,fl)

#measure one qubit
basis='X'
counts_chunk_index=0

trace_list=[]
for i in range(test_qubits):
    bit_wise_dist=counts[counts_chunk_index*test_qubits+i]
    #print(bit_wise_dist)
    trace=(bit_wise_dist['0']-bit_wise_dist['1'])/shots
    trace_list.append(trace)
#print(trace_list)

with open('../data_experimental/original_circuit' + '/trace_list_measure_one_zz_crosstalk_'+ basis +'.json', 'w') as fl:
    json.dump(trace_list,fl)
basis='Y'
counts_chunk_index=1

trace_list=[]
for i in range(test_qubits):
    bit_wise_dist=counts[counts_chunk_index*test_qubits+i]
    #print(bit_wise_dist)
    trace=(bit_wise_dist['0']-bit_wise_dist['1'])/shots
    trace_list.append(trace)
#print(trace_list)

with open('../data_experimental/original_circuit' + '/trace_list_measure_one_zz_crosstalk_'+ basis +'.json', 'w') as fl:
    json.dump(trace_list,fl)
basis='Z'
counts_chunk_index=2

trace_list=[]
for i in range(test_qubits):
    bit_wise_dist=counts[counts_chunk_index*test_qubits+i]
    #print(bit_wise_dist)
    trace=(bit_wise_dist['0']-bit_wise_dist['1'])/shots
    trace_list.append(trace)
#print(trace_list)
with open('../data_experimental/original_circuit' + '/trace_list_measure_one_zz_crosstalk_'+ basis +'.json', 'w') as fl:
    json.dump(trace_list,fl)
