import networkx as nx
import itertools
import numpy as np
from datetime import date
import os, glob, json
import sys
import json

module_path = os.path.abspath(os.path.join('../../../utilsVD'))
if module_path not in sys.path:
    sys.path.append(module_path)

module_path = os.path.abspath(os.path.join('../../../utilsExtrapolation'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
import utils_postprocessing as post

import yaml

with open('../../config.yaml') as file:
    configuration = yaml.load(file, Loader=yaml.FullLoader)

test_qubits=configuration['test_qubits']
scale_range = configuration['noise_scale_range']

# Get the argument provided on the command line (in this case, the second element of sys.argv)
argument = sys.argv[1]
pauli_op_index=int(argument)
pauli_string=configuration['pauli_op'][pauli_op_index]

path0='../../transpiled_circuit_list'
path1='../../data_experimental/extrapolation'

from vd_utils import obtain_diag_matrix,obtain_diag_matrix_sparse
from scipy import sparse
diag_matrix_for_denominator = obtain_diag_matrix_sparse(test_qubits)
eigenvalues_for_denominator = diag_matrix_for_denominator.data.tolist()
diag_matrix_for_numerator = obtain_diag_matrix_sparse(test_qubits,pauli_string)
eigenvalues_for_numerator = diag_matrix_for_numerator.data.tolist()

if not os.path.exists('../data_experimental/'):
    os.makedirs('../data_experimental/')

with open(path0 + '/cx_num_list_denominator.json', 'r') as fl:
    cx_list_denominator=json.load(fl)
with open(path0 + '/cx_num_list_numerator.json', 'r') as fl:
    cx_list_numerator=json.load(fl)
#print(cx_list_denominator)
#print(cx_list_numerator)

with open(path1 + '/list_counts_denominator_noise_free.json', 'r') as fl:
    list_counts_denominator=json.load(fl)
with open(path1 + '/list_counts_numerator_noise_free.json', 'r') as fl:
    list_counts_numerator=json.load(fl)
from vd_utils import compute_exp
from utils_postprocessing import norm_dict
list_denominator=[]
for j in range(0,len(scale_range)):
        denominator=compute_exp(eigenvalues_for_denominator,norm_dict(list_counts_denominator[j]))
        list_denominator.append(denominator)
#print(list_denominator )

list_numerator=[]
for k in range(pauli_op_index*len(scale_range),(pauli_op_index+1)*len(scale_range)):
        numerator=compute_exp(eigenvalues_for_numerator,norm_dict(list_counts_numerator[k]))
        list_numerator.append(numerator)
#print(list_numerator)

list_ratio=[]
for i in range(len(scale_range)):
    ratio=list_numerator[i]/list_denominator[i]
    list_ratio.append(ratio)
#print(list_ratio)

with open(path1 + '/list_counts_denominator_zz_crosstalk.json', 'r') as fl:
    list_counts_denominator=json.load(fl)

with open(path1 + '/list_counts_numerator_zz_crosstalk.json', 'r') as fl:
    list_counts_numerator=json.load(fl)
list_denominator=[]
for j in range(0,len(scale_range)):
        denominator=compute_exp(eigenvalues_for_denominator,norm_dict(list_counts_denominator[j]))
        list_denominator.append(denominator)
#print(list_denominator )
list_numerator=[]
for k in range(pauli_op_index*len(scale_range),(pauli_op_index+1)*len(scale_range)):
        numerator=compute_exp(eigenvalues_for_numerator,norm_dict(list_counts_numerator[k]))
        list_numerator.append(numerator)
#print(list_numerator)
list_ratio=[]
for i in range(len(scale_range)):
    ratio=list_numerator[i]/list_denominator[i]
    list_ratio.append(ratio)
#print(list_ratio)

with open('../data_experimental/vd_denominator' + '_' + pauli_string + '.json', 'w') as fl:
    json.dump(list_denominator[0],fl)
with open('../data_experimental/vd_numerator' + '_' + pauli_string + '.json', 'w') as fl:
    json.dump(list_numerator[0],fl)
with open('../data_experimental/vd_ratio' + '_' + pauli_string + '.json', 'w') as fl:
    json.dump(list_ratio[0],fl)

from extrapolation_utils import linear_fit_and_plot

fitted_func_denominator = linear_fit_and_plot(cx_list_denominator, list_denominator,  )
denominator_extrapolate=fitted_func_denominator(0)
#print(denominator_extrapolate)

fitted_func_numerator = linear_fit_and_plot(cx_list_numerator, list_numerator, )
numerator_extrapolate=fitted_func_numerator(0)
#print(numerator_extrapolate)

ratio_extrapolate=numerator_extrapolate/denominator_extrapolate
#print(ratio_extrapolate)

with open('../data_experimental/denominator_extrapolate' + '_' + pauli_string + '.json', 'w') as fl:
    json.dump(denominator_extrapolate,fl)

with open('../data_experimental/numerator_extrapolate' + '_' + pauli_string + '.json', 'w') as fl:
    json.dump(numerator_extrapolate,fl)

with open('../data_experimental/ratio_extrapolate' + '_' + pauli_string + '.json', 'w') as fl:
    json.dump(ratio_extrapolate,fl)

from vd_utils import add_cross_talk_noise
list_counts_denominator_new=[]
for counts in list_counts_denominator:
    output=add_cross_talk_noise(counts,0)
    for i in range(1,test_qubits):
        #print(i)
        output=add_cross_talk_noise(output,i*2)
    list_counts_denominator_new.append(output)
list_counts_numerator_new=[]
for counts in list_counts_numerator:
    output=add_cross_talk_noise(counts,0)
    for i in range(1,test_qubits):
        #print(i)
        output=add_cross_talk_noise(output,i*2)
    list_counts_numerator_new.append(output)
with open(path1 + '/list_counts_denominator_zz_crosstalk_measurement_crosstalk.json', 'w') as fl:
    json.dump(list_counts_denominator_new,fl)
with open(path1 + '/list_counts_numerator_zz_crosstalk_measurement_crosstalk.json', 'w') as fl:
    json.dump(list_counts_numerator_new,fl)
list_denominator=[]
for j in range(0,len(scale_range)):
        denominator=compute_exp(eigenvalues_for_denominator,norm_dict(list_counts_denominator_new[j]))
        list_denominator.append(denominator)
print(list_denominator )

list_numerator=[]
for k in range(pauli_op_index*len(scale_range),(pauli_op_index+1)*len(scale_range)):
        numerator=compute_exp(eigenvalues_for_numerator,norm_dict(list_counts_numerator_new[k]))
        list_numerator.append(numerator)
print(list_numerator)

list_ratio=[]
for i in range(len(scale_range)):
    ratio=list_numerator[i]/list_denominator[i]
    list_ratio.append(ratio)
print(list_ratio)
with open('../data_experimental/vd_denominator_add_measurement_crosstalk' + '_' + pauli_string + '.json', 'w') as fl:
    json.dump(list_denominator[0],fl)
with open('../data_experimental/vd_numerator_add_measurement_crosstalk' + '_' + pauli_string + '.json', 'w') as fl:
    json.dump(list_numerator[0],fl)
with open('../data_experimental/vd_ratio_add_measurement_crosstalk' + '_' + pauli_string + '.json', 'w') as fl:
    json.dump(list_ratio[0],fl)
fitted_func_denominator = linear_fit_and_plot(cx_list_denominator, list_denominator,  )
denominator_extrapolate=fitted_func_denominator(0)
print(denominator_extrapolate)

fitted_func_numerator = linear_fit_and_plot(cx_list_numerator, list_numerator, )
numerator_extrapolate=fitted_func_numerator(0)
print(numerator_extrapolate)

ratio_extrapolate=numerator_extrapolate/denominator_extrapolate
print(ratio_extrapolate)

with open('../data_experimental/denominator_extrapolate_add_measurement_crosstalk' + '_' + pauli_string + '.json', 'w') as fl:
    json.dump(denominator_extrapolate,fl)

with open('../data_experimental/numerator_extrapolate_add_measurement_crosstalk' + '_' + pauli_string + '.json', 'w') as fl:
    json.dump(numerator_extrapolate,fl)

with open('../data_experimental/ratio_extrapolate_add_measurement_crosstalk' + '_' + pauli_string + '.json', 'w') as fl:
    json.dump(ratio_extrapolate,fl)
