import networkx as nx
import itertools
import numpy as np
from datetime import date
import os, glob, json
import sys

import yaml

with open('../config.yaml') as file:
    configuration = yaml.load(file, Loader=yaml.FullLoader)

paul_op = configuration['pauli_op']
coeffs_pauli_op = configuration['coeffs_pauli_op']
offset = configuration['offset']

from qiskit.opflow import PauliSumOp
from qiskit.quantum_info import SparsePauliOp
# Create the SparsePauliOp object
sparse_op = SparsePauliOp(paul_op, coeffs_pauli_op)
cost_hamiltonian = PauliSumOp(sparse_op, coeff=1.0)

with open('../data_experimental/original_circuit/ideal_exp.json', 'r') as fl:
        exp_ori_circ_ideal=json.load(fl)
with open('../saved_results/exp_ideal.json', 'w') as fl:
        json.dump(exp_ori_circ_ideal+offset,fl)

with open('../data_experimental/original_circuit/noise_exp_zz_crosstalk.json', 'r') as fl:
        exp_ori_circ_noise=json.load(fl)
with open('../saved_results/exp_noise_add_zz_crosstalk.json', 'w') as fl:
        json.dump(exp_ori_circ_noise+offset,fl)
        
exp_vd=0
coef_index=0
for pauli_string in paul_op:
    with open('./data_experimental/vd_ratio' + '_' + pauli_string +'.json', 'r') as fl:
        exp_part= json.load(fl)
        exp_vd+=exp_part*coeffs_pauli_op[coef_index]
    coef_index+=1
with open('../saved_results/exp_vd_add_zz_crosstalk.json', 'w') as fl:
        json.dump(exp_vd+offset,fl)
        
        
exp_vd_extrapolation=0
coef_index=0
for pauli_string in paul_op:
    with open('./data_experimental/ratio_extrapolate' + '_' + pauli_string +'.json', 'r') as fl:
        exp_part= json.load(fl)
        exp_vd_extrapolation+=exp_part*coeffs_pauli_op[coef_index]
    coef_index+=1
with open('../saved_results/exp_vd_extrapolation_add_zz_crosstalk.json', 'w') as fl:
        json.dump(exp_vd_extrapolation+offset,fl)
        
        
exp_vd_cutting=0
coef_index=0
for i in range(len(paul_op)):
    with open('./data_experimental/vd_cutting_measure_one_pauli_op_' + str(i) +'/cutting_result_ratio.json', 'r') as fl:
        exp_part= json.load(fl)
        exp_vd_cutting+=exp_part*coeffs_pauli_op[coef_index]
    coef_index+=1
with open('../saved_results/exp_vd_cutting_add_zz_crosstalk.json', 'w') as fl:
        json.dump(exp_vd_cutting+offset,fl)

        
abs_error_exp_ori_circ_noise=abs(exp_ori_circ_noise-exp_ori_circ_ideal)
with open('../saved_results/abs_error_exp_noise_add_zz_crosstalk.json', 'w') as fl:
        json.dump(abs_error_exp_ori_circ_noise,fl)
        
abs_error_exp_vd=abs(exp_vd-exp_ori_circ_ideal)
with open('../saved_results/abs_error_exp_vd_add_zz_crosstalk.json', 'w') as fl:
        json.dump(abs_error_exp_vd,fl)
        
abs_error_exp_vd_extrapolation=abs(exp_vd_extrapolation-exp_ori_circ_ideal)
with open('../saved_results/abs_error_exp_vd_extrapolation_add_zz_crosstalk.json', 'w') as fl:
        json.dump(abs_error_exp_vd_extrapolation,fl)
        
abs_error_exp_vd_cutting=abs(exp_vd_cutting-exp_ori_circ_ideal)
with open('../saved_results/abs_error_exp_vd_cutting_add_zz_crosstalk.json', 'w') as fl:
        json.dump(abs_error_exp_vd_cutting,fl)
