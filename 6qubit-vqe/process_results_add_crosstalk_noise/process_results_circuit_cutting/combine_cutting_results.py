import os, glob, json
import sys

module_path = os.path.abspath(os.path.join('../../../utilsVD'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
import yaml

with open('../../config.yaml') as file:
    configuration = yaml.load(file, Loader=yaml.FullLoader)
with open('./pauli_op_index.yaml') as file:
    config_pauli_op_index = yaml.load(file, Loader=yaml.FullLoader)
pauli_op_index=config_pauli_op_index['pauli_op_index']
print(pauli_op_index)
pauli_string=configuration['pauli_op'][pauli_op_index]

test_qubits=configuration['test_qubits']
scale_range=configuration['noise_scale_range']

from vd_utils import obtain_diag_matrix,obtain_diag_matrix_sparse
import numpy as np
from scipy import sparse
diag_matrix_for_denominator = obtain_diag_matrix_sparse(test_qubits)
eigenvalues_for_denominator = diag_matrix_for_denominator.data.tolist()
diag_matrix_for_numerator = obtain_diag_matrix_sparse(test_qubits,pauli_string)
eigenvalues_for_numerator = diag_matrix_for_numerator.data.tolist()

with open('../../data_experimental/extrapolation/list_counts_denominator_zz_crosstalk.json', 'r') as fl:
    list_counts_denominator=json.load(fl)
with open('../../data_experimental/extrapolation/list_counts_numerator_zz_crosstalk.json', 'r') as fl:
    list_counts_numerator=json.load(fl)
    
counts_denominator_vd= list_counts_denominator[0]

counts_numerator_vd= list_counts_numerator[pauli_op_index*len(scale_range)]

list_denominator_recombined_dist=[]
list_numerator_recombined_dist=[]

for i in range(test_qubits):
    with open('../data_experimental/vd_cutting_measure_one_pauli_op_' +  str(pauli_op_index)  + '/denominator_recombined_dist_check_' + str(i) + ' .json', 'r') as fl:
        denominator_recombined_dist= json.load(fl)
        list_denominator_recombined_dist.append([denominator_recombined_dist, i*2, 0])
    with open('../data_experimental/vd_cutting_measure_one_pauli_op_' +  str(pauli_op_index)  + '/numerator_recombined_dist_check_' + str(i) + ' .json', 'r') as fl:
        numerator_recombined_dist= json.load(fl)
        list_numerator_recombined_dist.append([numerator_recombined_dist,i*2, 0])

from vd_utils import compute_exp, bayesian_reconstruct
from utils_postprocessing import norm_dict

unmiti_denominator_dist = norm_dict(counts_denominator_vd)
unmiti_numerator_dist = norm_dict(counts_numerator_vd)

combined_denominator_dist=bayesian_reconstruct(unmiti_denominator_dist,list_denominator_recombined_dist)
combined_numerator_dist=bayesian_reconstruct(unmiti_numerator_dist,list_numerator_recombined_dist)
denominator=compute_exp(eigenvalues_for_denominator,combined_denominator_dist)
numerator=compute_exp(eigenvalues_for_numerator,combined_numerator_dist)
ratio=numerator/denominator

print(denominator)
print(numerator)
print(ratio)

with open('../data_experimental/vd_cutting_measure_one_pauli_op_' +  str(pauli_op_index)  + '/cutting_result_denominator.json', 'w') as fl:
    json.dump(denominator,fl)
with open('../data_experimental/vd_cutting_measure_one_pauli_op_' +  str(pauli_op_index)  + '/cutting_result_numerator.json', 'w') as fl:
    json.dump(numerator,fl)
with open('../data_experimental/vd_cutting_measure_one_pauli_op_' +  str(pauli_op_index)  + '/cutting_result_ratio.json', 'w') as fl:
    json.dump(ratio,fl)
