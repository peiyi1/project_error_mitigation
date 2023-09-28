import numpy as np
import math
import os
from qiskit.converters import circuit_to_dag, dag_to_circuit, circuit_to_dagdependency
import rustworkx as rx
import itertools
from qiskit.dagcircuit.dagnode import DAGNode, DAGOpNode, DAGInNode, DAGOutNode
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.circuit import Parameter
from qiskit.circuit.library.n_local.qaoa_ansatz import QAOAAnsatz
from qiskit.circuit.library import TwoLocal,RealAmplitudes
from qiskit.providers.aer import AerSimulator

#to do: the function compute_expectation_from_hamiltonian is just for obs constains Pauli Z and I
def compute_expectation_from_hamiltonian(counts, pauli_sum_op):
  # Initialize the expectation value to zero
  expval = 0

  # Loop over the terms in the PauliSumOp
  for term, coeff in zip(pauli_sum_op.primitive, pauli_sum_op.coeffs):
    # Loop over the tuples of Pauli strings and coefficients in the term
    for pauli_str, coeff in term.to_list():
      # Initialize the contribution of this term to one
      term_val = 1

      # Loop over the states and frequencies in the counts
      for state, freq in counts.items():
        # Calculate the eigenvalue of this term for this state
        eigenval = 1
        for i, pauli in enumerate(pauli_str):
            if pauli == "Z" and state[i] == "1":
                eigenval *= -1
            elif pauli == "X" and state[i] == "1":
                eigenval *= -1
        # Multiply the eigenvalue by the frequency and add it to the term value
        term_val += eigenval * freq

      # Multiply the term value by the coefficient and add it to the expectation value
      expval += coeff * term_val

  # Normalize the expectation value by the total number of shots
  expval /= sum(counts.values())

  # Return the expectation value
  return expval


# Finally we write a function that executes the circuit on the chosen backend
def vqe_get_expectation_from_hamiltonian(test_qubits, cost_operator, ansatz_type= "RealAmplitudes",reps=1, shots=10000, seed=0, entanglement="circular"):
    
    backend = AerSimulator()
    backend.shots = shots
    backend.seed_simulator=seed
    
    def execute_circ(theta):
        if (ansatz_type == "RealAmplitudes"):
            qc = RealAmplitudes(test_qubits, reps=reps, entanglement=entanglement)
            qc=qc.bind_parameters(theta)
            qc=qc.decompose()
            qc.measure_all()
            counts = backend.run(qc).result().get_counts()
        elif (ansatz_type == "TwoLocal"):
            qc = TwoLocal(test_qubits, "ry", "cz", reps=reps, entanglement=entanglement)
            qc=qc.bind_parameters(theta)
            qc=qc.decompose()
            qc.measure_all()
            counts = backend.run(qc).result().get_counts()
        return compute_expectation_from_hamiltonian(counts, cost_operator)
    
    return execute_circ

    
def obtain_record_qubits_index_dict(qc):
    record_qubits_index_dict={}
    # Print the final qubit layout
    for quantum_register in qc.data:
        if(quantum_register[0].name == 'measure'):
            cbit_index = qc.find_bit(quantum_register[2][0]).index
            qubit_index = qc.find_bit(quantum_register[1][0]).index
            record_qubits_index_dict[cbit_index]=qubit_index
    return record_qubits_index_dict

def obtain_record_qubits_index(record_qubits_index_dict):
    record_qubits_index=[]
    for i in range(len(record_qubits_index_dict)):
        record_qubits_index.append(record_qubits_index_dict[i])
    return record_qubits_index
    
    
    
    
    