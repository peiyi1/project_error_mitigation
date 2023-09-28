#!/bin/bash

# Change the working directory to the current directory
cd "$(dirname "$0")"

python process_results.py

cd ./process_results_extrapolation/
python process_results.py 0
python process_results.py 1
python process_results.py 2

cd ../process_results_circuit_cutting/
python process_cutting_results.py 0
python process_cutting_results.py 1
python process_cutting_results.py 2
python process_cutting_results.py 3
python combine_cutting_results.py 
python combine_cutting_results_add_measurement_crosstalk.py 
sed -i 's/pauli_op_index: 0/pauli_op_index: 1/' pauli_op_index.yaml
python process_cutting_results.py 0
python process_cutting_results.py 1
python process_cutting_results.py 2
python process_cutting_results.py 3
python combine_cutting_results.py 
python combine_cutting_results_add_measurement_crosstalk.py 
sed -i 's/pauli_op_index: 1/pauli_op_index: 2/' pauli_op_index.yaml
python process_cutting_results.py 0
python process_cutting_results.py 1
python process_cutting_results.py 2
python process_cutting_results.py 3
python combine_cutting_results.py 
python combine_cutting_results_add_measurement_crosstalk.py 
cd ../
python calculate_expectation_value.py
python calculate_expectation_value_add_measurement_crosstalk.py
