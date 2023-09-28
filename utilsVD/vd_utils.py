import numpy as np
import math
import os
from qiskit.converters import circuit_to_dag, dag_to_circuit, circuit_to_dagdependency
import rustworkx as rx
import itertools
from qiskit.dagcircuit.dagnode import DAGNode, DAGOpNode, DAGInNode, DAGOutNode
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.circuit import Parameter

import itertools
from qiskit.circuit import Gate

# Define a 2-qubit gate as a custom gate object
my_gate = Gate(name='my_gate', num_qubits=2, params=[])
# Define the matrix representation of the gate
my_gate.to_matrix = lambda: np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

def apply_subcircuit(qc, *qargs,pauli_obs,scale):
    subcircuit=QuantumCircuit(2)
    unitary_gate=None
    if(pauli_obs == 'I'):
            subcircuit.cx(0,1)
            subcircuit.h(0)
            for j in range(scale-1): 
                subcircuit.h(0)
                subcircuit.cx(0,1)
                subcircuit.cx(0,1)
                subcircuit.h(0)
                    
    else:
            #to do: Y case
            if(pauli_obs == 'X'):
                unitary_gate=obtain_matrix3()
            elif(pauli_obs == 'Z'):
                unitary_gate=obtain_matrix1()
            subcircuit.unitary(unitary_gate, [0,1])
            for j in range(scale-1): 
                subcircuit.unitary(dagger(unitary_gate), [0,1])
                subcircuit.unitary(unitary_gate, [0,1])
                
    #qc=qc.compose(subcircuit,qargs)    
    qc.append(subcircuit, qargs)

def repace_custom_gate(input_circ,pauli_string='I',scale=1):
    """Reduce a transpiled circuit down to only active qubits.

    Parameters:
        input_circ (QuantumCircuit): Input circuit.

    Returns:
        QuantumCircuit: Reduced circuit.

    Notes:
        Requires a circuit with flatten qregs and cregs.
    """
    pauli_string=pauli_string[::-1]
    pauli_string_index=0
    
    #active_qubits, active_clbits = cc.active_bits(input_circ)

    #num_reduced_qubits = len(active_qubits)
    #num_reduced_clbits = len(active_clbits)

    #active_qubit_map = {}
    #active_bit_map = {}
    #for idx, val in enumerate(
    #    sorted(active_qubits, key=lambda x: input_circ.find_bit(x).index)
    #):
    #    active_qubit_map[val] = idx
    #for idx, val in enumerate(
    #    sorted(active_clbits, key=lambda x: input_circ.find_bit(x).index)
    #):
    #    active_bit_map[val] = idx
    #print(active_qubit_map)
    num_qubits=input_circ.num_qubits
    num_clbits=input_circ.num_clbits
    #print(num_qubits)
    #print(num_clbits)
    new_qc = QuantumCircuit(num_qubits, num_clbits)
    
    # create a dictionary of instruction names and methods
    instruction_dict = {
        'cx': new_qc.cx,
        'id': new_qc.id,
        'rz': new_qc.rz,
        'sx': new_qc.sx,
        'x': new_qc.x,
        'reset': new_qc.reset,
        'ry': new_qc.ry,
        'u3': new_qc.u3,
        'h': new_qc.h,
        'barrier': new_qc.barrier,
        'measure': new_qc.measure,
        'my_gate': my_gate,
        # add more instructions as needed
    }
    
    for item in input_circ.data:
        # Find active qubits used by instruction (if any)
        #used_active_set = [qubit for qubit in item[1] if qubit in active_qubits]
        # If any active qubits used, add to deflated circuit
        #if any(used_active_set):
            ref = instruction_dict[item[0].name]#getattr(new_qc, item[0].name)
            params = item[0].params
            qargs = item[1]#[new_qc.qubits[active_qubit_map[qubit]] for qubit in used_active_set]
            cargs = item[2]#[new_qc.clbits[active_bit_map[clbit]] for clbit in item[2]]
            # Inside the loop where you're processing instructions
            if item[0].name == 'my_gate':
                #print(qargs)
                apply_subcircuit(new_qc, *qargs,pauli_obs=pauli_string[params[0]],scale=scale)
                pauli_string_index+=1
            else:
                ref(*params, *qargs, *cargs)
    new_qc.global_phase = input_circ.global_phase
    
    return new_qc


def gen_ancestors_circuit(input_qc, qubit_id,  qubit_id_2, nodes_diag_gate=7, nodes_cnot_gate=1):
    # Convert the input circuit to a directed acyclic graph (DAG)
    qc_dag = circuit_to_dag(input_qc)
    
    # Get the generator for the nodes on the specified qubit wire
    gen = qc_dag.nodes_on_wire(qc_dag.wires[qubit_id])
    
    node_list = qc_dag.nodes_on_wire(qc_dag.wires[qubit_id], only_ops=True)
    len_node_list = len(list(node_list))
    node_id = len_node_list - nodes_cnot_gate
    node_id_cutting = len_node_list - nodes_cnot_gate - nodes_diag_gate
    # Get the node at the specified node index from the generator
    node = next(itertools.islice(gen, node_id, None))
    # Find the ancestors of the node using rustworkx
    anc = rx.ancestors(qc_dag._multi_graph, node._node_id)
    # Convert the set of ancestor indices to a set of ancestor nodes
    anc_set = {node}
    for idx in anc:
        anc_set.add(qc_dag._multi_graph[idx])
  
    
    # Get the generator for the nodes on the specified qubit wire
    gen = qc_dag.nodes_on_wire(qc_dag.wires[qubit_id_2])
    
    node_list = qc_dag.nodes_on_wire(qc_dag.wires[qubit_id_2], only_ops=True)
    len_node_list = len(list(node_list))
    node_id_2 = len_node_list - nodes_cnot_gate
    node_id_cutting_2=len_node_list - nodes_cnot_gate - nodes_diag_gate
    # Get the node at the specified node index from the generator
    node = next(itertools.islice(gen, node_id_2, None))
    anc_set.add(node)
    # Find the ancestors of the node using rustworkx
    anc = rx.ancestors(qc_dag._multi_graph, node._node_id)
    # Convert the set of ancestor indices to a set of ancestor nodes
    for idx in anc:
        anc_set.add(qc_dag._multi_graph[idx])

    # Find the complement set of nodes that are not ancestors
    comp = list(set(qc_dag._multi_graph.nodes()) - anc_set)
    
    # Remove any DAGOpNodes that are not ancestors from the DAG
    for n in comp:
        if isinstance(n, DAGOpNode):
            qc_dag.remove_op_node(n)
    
    # Convert the trimmed DAG back to a circuit
    trimmed_qc = dag_to_circuit(qc_dag)
    
    # Return the trimmed circuit
    return (trimmed_qc, node_id_cutting, node_id_cutting_2)


def gen_ancestors_circuit_no_simplify(input_qc, qubit_id,  qubit_id_2, nodes_diag_gate=7, nodes_cnot_gate=0):
    # Convert the input circuit to a directed acyclic graph (DAG)
    qc_dag = circuit_to_dag(input_qc)
    
    # Get the generator for the nodes on the specified qubit wire
    gen = qc_dag.nodes_on_wire(qc_dag.wires[qubit_id])
    
    node_list = qc_dag.nodes_on_wire(qc_dag.wires[qubit_id], only_ops=True)
    len_node_list = len(list(node_list))
    node_id = len_node_list - nodes_cnot_gate
    node_id_cutting = len_node_list - nodes_cnot_gate - nodes_diag_gate
    # Get the node at the specified node index from the generator
    node = next(itertools.islice(gen, node_id, None))
    # Find the ancestors of the node using rustworkx
    anc = rx.ancestors(qc_dag._multi_graph, node._node_id)
    # Convert the set of ancestor indices to a set of ancestor nodes
    anc_set = {node}
    for idx in anc:
        anc_set.add(qc_dag._multi_graph[idx])
  
    
    # Get the generator for the nodes on the specified qubit wire
    gen = qc_dag.nodes_on_wire(qc_dag.wires[qubit_id_2])
    
    node_list = qc_dag.nodes_on_wire(qc_dag.wires[qubit_id_2], only_ops=True)
    len_node_list = len(list(node_list))
    node_id_2 = len_node_list - nodes_cnot_gate
    node_id_cutting_2=len_node_list - nodes_cnot_gate - nodes_diag_gate
    # Get the node at the specified node index from the generator
    node = next(itertools.islice(gen, node_id_2, None))
    anc_set.add(node)
    # Find the ancestors of the node using rustworkx
    anc = rx.ancestors(qc_dag._multi_graph, node._node_id)
    # Convert the set of ancestor indices to a set of ancestor nodes
    for idx in anc:
        anc_set.add(qc_dag._multi_graph[idx])

    # Find the complement set of nodes that are not ancestors
    comp = list(set(qc_dag._multi_graph.nodes()) - anc_set)
    
    # Remove any DAGOpNodes that are not ancestors from the DAG
    #for n in comp:
    #    if isinstance(n, DAGOpNode):
    #        qc_dag.remove_op_node(n)
    
    # Convert the trimmed DAG back to a circuit
    trimmed_qc = dag_to_circuit(qc_dag)
    
    # Return the trimmed circuit
    return (trimmed_qc, node_id_cutting, node_id_cutting_2)


##########################################################################################

def dagger(matrix):
    return np.conjugate(matrix).T

def tensor_product(*matrices):
    result = matrices[0]
    for matrix in matrices[1:]:
        result = np.kron(result, matrix)
    return result

from scipy import sparse

def tensor_product_sparse(*matrices):
    result = matrices[0]
    for matrix in matrices[1:]:
        # convert matrix to sparse matrix if it is a dense array
        if isinstance(matrix, np.ndarray):
            matrix = sparse.csr_matrix(matrix)
        result = sparse.kron(result, matrix)
    return result

#diagonalize zi@swap
def obtain_matrix1():
    matrix= [[1, 0, 0, 0],
          [0, -1.0j/math.sqrt(2), 1/math.sqrt(2), 0],
          [0, 1.0j/math.sqrt(2), 1/math.sqrt(2), 0],
          [0, 0, 0, 1]]
    return matrix

#diagonalize swap
'''
def obtain_matrix2():
    matrix= [[1, 0, 0, 0],
          [0, 1.0/math.sqrt(2), -1/math.sqrt(2), 0],
          [0, 1.0/math.sqrt(2), 1/math.sqrt(2), 0],
          [0, 0, 0, 1]]
    return matrix
'''
def obtain_matrix2():
    matrix= [[1.0/math.sqrt(2), 0, 0, 1/math.sqrt(2)],
          [1.0/math.sqrt(2), 0, 0, -1/math.sqrt(2)],
          [0, 1.0/math.sqrt(2),  1/math.sqrt(2), 0],
          [0, -1.0/math.sqrt(2), 1/math.sqrt(2), 0]]
    return matrix

#diagonalize xi@swap
def obtain_matrix3():
    matrix= [[1/2, -1/2, -1/2, 1/2],
          [-1/2, -1.0j/2, 1.0j/2, 1/2],
          [-1/2, 1.0j/2, -1.0j/2, 1/2],
          [1/2, 1/2, 1/2, 1/2]]
    return matrix

def obtain_matrix_x():
    matrix= [[0,1], [1,0]]
    return matrix

def obtain_matrix_z():
    matrix= [[1,0],[0,-1]]
    return matrix

def obtain_matrix_i():
    matrix= [[1,0],[0,1]]
    return matrix

def obtain_matrix_swap():
    matrix= [[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]
    return matrix

def obtain_matrix_xi():
    matrix= np.kron(obtain_matrix_x(),obtain_matrix_i())
    return matrix

def obtain_matrix_ix():
    matrix= np.kron(obtain_matrix_i(),obtain_matrix_x())
    return matrix

def obtain_matrix_zi():
    matrix= np.kron(obtain_matrix_z(),obtain_matrix_i())
    return matrix

def obtain_matrix_iz():
    matrix= np.kron(obtain_matrix_i(),obtain_matrix_z())
    return matrix

def obtain_matrix_ii():
    matrix= np.kron(obtain_matrix_i(),obtain_matrix_i())
    return matrix

def diag_matrix_xi():
    diag=obtain_matrix3()@np.asarray(obtain_matrix_xi()@obtain_matrix_swap())@dagger(obtain_matrix3())
    return diag
def diag_matrix_ix():
    diag=obtain_matrix3()@np.asarray(obtain_matrix_ix()@obtain_matrix_swap())@dagger(obtain_matrix3())
    return diag
def diag_matrix_zi():
    diag=obtain_matrix1()@np.asarray(obtain_matrix_zi()@obtain_matrix_swap())@dagger(obtain_matrix1())
    return diag
def diag_matrix_iz():
    diag=obtain_matrix1()@np.asarray(obtain_matrix_iz()@obtain_matrix_swap())@dagger(obtain_matrix1())
    return diag
def diag_matrix_ii():
    #matrix= [[1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]]
    matrix= [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,-1]]
    return np.asarray(matrix)

def obtain_diag_matrix(test_qubits,pauli_string=None):
        if not pauli_string:
            pauli_string=''
            for i in range(test_qubits):
                pauli_string+='I'
        #print(pauli_string)
        list_pauli_matrix=[]
        for i in pauli_string:
            if(i == 'X'):
                list_pauli_matrix.append(diag_matrix_xi())
            elif(i == 'Z'):
                list_pauli_matrix.append(diag_matrix_zi())
            else:
                list_pauli_matrix.append(diag_matrix_ii())
        #print(list_pauli_matrix)
        diag_matrix= tensor_product(*list_pauli_matrix)
        
        return diag_matrix

def obtain_diag_matrix_sparse(test_qubits,pauli_string=None):
        if not pauli_string:
            pauli_string=''
            for i in range(test_qubits):
                pauli_string+='I'
        #print(pauli_string)
        list_pauli_matrix=[]
        for i in pauli_string:
            if(i == 'X'):
                list_pauli_matrix.append(diag_matrix_xi())
            elif(i == 'Z'):
                list_pauli_matrix.append(diag_matrix_zi())
            else:
                list_pauli_matrix.append(diag_matrix_ii())
        #print(list_pauli_matrix)
        diag_matrix= tensor_product_sparse(*list_pauli_matrix)
        
        return diag_matrix
    
    
def obtain_observable(test_qubits,pauli_string=None):
        if not pauli_string:
            pauli_string=''
            for i in range(test_qubits):
                pauli_string+='I'
        #print(pauli_string)
        list_pauli_matrix=[]
        for i in pauli_string:
            if(i == 'X'):
                list_pauli_matrix.append(obtain_matrix_xi()@obtain_matrix_swap())
            elif(i == 'Z'):
                list_pauli_matrix.append(obtain_matrix_zi()@obtain_matrix_swap())
            else:
                list_pauli_matrix.append(obtain_matrix_swap())
        #print(list_pauli_matrix)
        diag_matrix= tensor_product(*list_pauli_matrix)
        
        return diag_matrix    
    
def obtain_observable_sparse(test_qubits,pauli_string=None):
        if not pauli_string:
            pauli_string=''
            for i in range(test_qubits):
                pauli_string+='I'
        #print(pauli_string)
        list_pauli_matrix=[]
        for i in pauli_string:
            if(i == 'X'):
                list_pauli_matrix.append(obtain_matrix_xi()@obtain_matrix_swap())
            elif(i == 'Z'):
                list_pauli_matrix.append(obtain_matrix_zi()@obtain_matrix_swap())
            else:
                list_pauli_matrix.append(obtain_matrix_swap())
        #print(list_pauli_matrix)
        diag_matrix= tensor_product_sparse(*list_pauli_matrix)
        
        return diag_matrix    
##########################################################################################    
def remove_files_in_directory(directory):
    # Get the list of files in the directory
    file_list = os.listdir(directory)
    
    # Iterate over the file list and remove each file
    for file_name in file_list:
        file_path = os.path.join(directory, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)

def bin_to_dec(bitstring):
    return int(bitstring, 2)

def dec_to_bin(dec, bitwidth):
    # convert decimal to binary string
    bin_str = format(dec, 'b')
    # pad with zeros if necessary
    if len(bin_str) < bitwidth:
        bin_str = '0' * (bitwidth - len(bin_str)) + bin_str
    # return binary string
    return bin_str

def create_dict(width):
    # initialize an empty dictionary
    dictionary = {}
    # loop from 0 to 2^width - 1
    for i in range(2**width):
        # convert i to a binary string with the specified width
        key = bin(i)[2:].zfill(width)
        # assign 0 as the value for the key
        dictionary[key] = 0
    # return the dictionary
    return dictionary

def reset_dict_values(input_dict):
    output_dict = {}
    for key in input_dict:
        output_dict[key] = 0
    return output_dict

def compute_exp(eigenvalues,dict_data):
    exp=0

    for key in dict_data.keys():
        exp += eigenvalues[bin_to_dec(key)]*dict_data[key]

    return exp.real

############################################################################################################
def create_2copy_circuit(test_qubits, circuit_ori):
    qreg_q = QuantumRegister(test_qubits*2, 'q')
    creg_q = ClassicalRegister(test_qubits*2, 'c')

    circuit = QuantumCircuit(qreg_q,creg_q)

    circuit.append(circuit_ori, [i for i in range(0,test_qubits*2,2)])
    circuit.append(circuit_ori, [i for i in range(1,test_qubits*2,2)])

 
    return circuit
def create_2copy_circuit_method1(test_qubits, circuit_ori):
    qreg_q = QuantumRegister(test_qubits*2, 'q')
    creg_q = ClassicalRegister(test_qubits*2, 'c')

    circuit = QuantumCircuit(qreg_q)

    circuit.append(circuit_ori, [i for i in range(0,test_qubits)])
    circuit.append(circuit_ori, [i for i in range(test_qubits,test_qubits*2,)])

 
    return circuit


def add_diag_gate_for_two_copy_circuit(circuit,my_gate_index_list,measurement_order,pauli_string='I',scale=1):
    pauli_string=pauli_string[::-1]
    gate_index=0
    for i in pauli_string:
        subcircuit=QuantumCircuit(2)
        unitary_gate=None
        if(i == 'I'):
            subcircuit.cx(0,1)
            subcircuit.h(0)
            for j in range(scale-1): 
                subcircuit.h(0)
                subcircuit.cx(0,1)
                subcircuit.cx(0,1)
                subcircuit.h(0)
                    
        else:
            #to do: Y case
            if(i == 'X'):
                unitary_gate=obtain_matrix3()
            elif(i == 'Z'):
                unitary_gate=obtain_matrix1()
            subcircuit.unitary(unitary_gate, [0,1])
            for j in range(scale-1): 
                subcircuit.unitary(dagger(unitary_gate), [0,1])
                subcircuit.unitary(unitary_gate, [0,1])
        
        circuit=circuit.compose(subcircuit,my_gate_index_list[gate_index])
        gate_index+=1
        
    c_reg=ClassicalRegister(len(measurement_order),'c')
    circuit.add_register(c_reg)
    for k in range(len(measurement_order)):
        circuit.measure(measurement_order[k],k)
    return circuit


#########################################################################################
#index: index from right side to left side
def two_bit_weight(dist, index):
    #bitwise distribution
    weight_00 = 0
    weight_01 = 0
    weight_10 = 0
    weight_11 = 0
    for key in dist.keys():
        if  (key[len(key) - 1 - index-1] == '0') and (key[len(key) - 1 - index] == '0'):
            weight_00 += dist[key]
        elif (key[len(key) - 1 - index-1] == '0') and (key[len(key) - 1 - index] == '1'):
            weight_01 += dist[key]
        elif (key[len(key) - 1 - index-1] == '1') and (key[len(key) - 1 - index] == '0'):
            weight_10 += dist[key]
        elif (key[len(key) - 1 - index-1] == '1') and (key[len(key) - 1 - index] == '1'):
            weight_11 += dist[key]
        else:
            print("Incorrect key value")
    return weight_00, weight_01, weight_10, weight_11

def update_dist(unmiti_dist, miti_dist, index, index_for_miti_dist):
    Ppost = {}
    w00, w01, w10, w11 = two_bit_weight(miti_dist, index_for_miti_dist)
    u_w00, u_w01, u_w10, u_w11 = two_bit_weight(unmiti_dist, index)
    
    if w00 == 0:
        w00 = 0.0000000000001
        #w1 = 0.9999999999999
    if w01 == 0:
        w01 = 0.0000000000001
        #w0 = 0.9999999999999
    if w10 == 0:
        w10 = 0.0000000000001
        #w1 = 0.9999999999999
    if w11 == 0:
        w11 = 0.0000000000001
        #w0 = 0.9999999999999
    if u_w00 == 0:
        u_w00 = 0.0000000000001
        #u_w1 = 0.9999999999999
    if u_w01 == 0:
        u_w01 = 0.0000000000001
        #u_w0 = 0.9999999999999
    if u_w10 == 0:
        u_w10 = 0.0000000000001
        #u_w1 = 0.9999999999999
    if u_w11 == 0:
        u_w11 = 0.0000000000001
        #u_w0 = 0.9999999999999
       
    for key in unmiti_dist.keys():
        if (key[len(key) - 1 - index-1] == '0') and (key[len(key) - 1 - index] == '0'):
            Ppost[key] = unmiti_dist[key] / u_w00 * (w00)# / w1)
            #print(w0, w1, w0/w1, Ppost[key])
        elif (key[len(key) - 1 - index-1] == '0') and (key[len(key) - 1 - index] == '1'):
            Ppost[key] = unmiti_dist[key] / u_w01 * (w01)# / w0)
            #print(w0, w1, w1/w0, Ppost[key])
        elif (key[len(key) - 1 - index-1] == '1') and (key[len(key) - 1 - index] == '0'):
            Ppost[key] = unmiti_dist[key] / u_w10 * (w10)
        elif (key[len(key) - 1 - index-1] == '1') and (key[len(key) - 1 - index] == '1'):
            Ppost[key] = unmiti_dist[key] / u_w11 * (w11)    
        else:
            print("Incorrect key value")
    return Ppost

def combine_dist(orign_dist, dist_list):
    output_dist = {}
    for key in orign_dist:
        value = orign_dist[key]
        for dist in dist_list:
            value += dist[key]
        output_dist[key] = value
    return output_dist
def total_counts(dictionary):
    total = 0
    for value in dictionary.values():
        total += value
    return total
def norm_dict(dictionary):
    total = total_counts(dictionary)
    norm_dist = {}
    for i in dictionary.keys():
        norm_dist[i] = dictionary[i]/total
    return norm_dist

def H_distance_dict(p, q):
    # distance between p an d
    # p and q are np array probability distributions
    sum = 0.0
    for key in p.keys():
        sum += (np.sqrt(p[key]) - np.sqrt(q[key]))**2
    result = (1.0 / np.sqrt(2.0)) * np.sqrt(sum)
    return result

def bayesian_reconstruct(unmiti_dist, miti_dist_list, threshold = 0.0001):
    temp_dist = unmiti_dist.copy()
    h_dist = 1
    while h_dist > threshold:
        temp_dist_start = temp_dist.copy()
        ppost = [0] * len(miti_dist_list)
        for i in range(0, len(miti_dist_list)):
            ppost[i] = update_dist(temp_dist, miti_dist_list[i][0], miti_dist_list[i][1], miti_dist_list[i][2])
        #print(ppost)
        #print(len(ppost))
        temp_dist = combine_dist(temp_dist, ppost)
        temp_dist = norm_dict(temp_dist)
        h_dist = H_distance_dict(temp_dist, temp_dist_start)
        #h_dist=0.0001
        #print("H-dist:", h_dist)
    return temp_dist

###########################################################################
from bqskit import Circuit
from bqskit.ir import Operation
from bqskit.ir.gates import RZZGate

def find_connected_pairs(coupling_map, edges):
  # coupling_map is a list of lists representing the pairs of qubits that support two-qubit gate operations
  # edges is a list of lists representing the qubits in each edge
  # returns a list of tuples containing the pairs of edges that are connected by one other edge and do not share the same node
  
  # create an empty list to store the connected pairs
  connected_pairs = []
  
  # loop through all possible pairs of edges
  for i in range(len(edges)):
    for j in range(i+1, len(edges)):
      # get the qubits in each edge
      q1, q2 = edges[i]
      q3, q4 = edges[j]
      
      # check if the edges share the same node
      if q1 == q3 or q1 == q4 or q2 == q3 or q2 == q4:
        continue
      
      # loop through the coupling map and check if there is an edge that connects q1 or q2 with q3 or q4
      for edge in coupling_map:
        p, q = edge
        if (p == q1 or p == q2) and (q == q3 or q == q4):
          # add the pair of edges to the connected pairs list
          connected_pairs.append((edges[i], edges[j]))
          break
        if (p == q3 or p == q4) and (q == q1 or q == q2):
          # add the pair of edges to the connected pairs list
          connected_pairs.append((edges[i], edges[j]))
          break
  
  # return the connected pairs list
  return connected_pairs

def obtain_circuit_add_rzz(circuit,backend):
    coupling_map=backend.configuration().coupling_map
    backend_num_qubits=backend.configuration().n_qubits
    # Create a circuit 
    new_circuit = Circuit(backend_num_qubits)

    op_list=[]
    for i in circuit.operations_with_cycles():
        op_list.append(i)
        
    #Initialize an empty dictionary to store the groups
    groups = {}
    # Loop over the tuples in the circuit
    for t, g in op_list:
        # Check if the time step is already in the dictionary
        if t in groups:
            # Append the gate and qubit index to the existing group
            groups[t].append((g))
        else:
            # Create a new group with the gate and qubit index
            groups[t] = [(g)]

            
    # Loop over the time steps in the groups
    for t in sorted(groups):
        # Print the time step
        #print(f"Time step {t}:")
        
        # Initialize a flag to indicate if there is a 2-qubit gate
        has_2q_gate = False
        edge_list =[]
        
        # Loop over the gates and qubits in the group
        for g in groups[t]:
            # Check if the gate is a 2-qubit gate
            if g.num_qudits == 2:

                # Set the flag to True
                has_2q_gate = True

                # Get the control and target qubits
                control, target = g.location
                edge=[control, target]
                edge_list.append(edge)
        #print(edge_list)
        
        checked_results=find_connected_pairs(coupling_map, edge_list)
        #for item in checked_results:
        #    print(item[0])
        #    print(item[1])
            
        for g in groups[t]:
            new_circuit.append(g)
            for item in checked_results:
                if (list(g.location) == item[0] ) or (list(g.location) == item[1] ):
                    opRZZ = Operation(RZZGate(-np.pi/3.5), g.location)
                    new_circuit.append(opRZZ)
        #print(new_circuit)        
        # Check if there is no 2-qubit gate
        #if not has_2q_gate:
            # Print that there is no 2-qubit gate
            #print("- No 2-qubit gate")

    return new_circuit
#########################################################################################################
def change_string(input_string, input_value, index):
    # Define the probabilities of changing each pair of bits
    change_probs = {'00': {'00': 0.991, '01': 0.003, '10': 0.003, '11': 0.003},
                    '01': {'00': 0.003, '01': 0.991, '10': 0.003, '11': 0.003},
                    '10': {'00': 0.003, '01': 0.003, '10': 0.991, '11': 0.003},
                    '11': {'00': 0.003, '01': 0.003, '10': 0.003, '11': 0.991}}

    # Initialize the output dictionary as an empty dictionary
    output_dict = {}

    # Split the input string into two pairs of bits
    pair1 = input_string[index:index+2]
    #pair2 = input_string[index+2:index+4]
    #print(pair1)
    #print(pair2)
    # Loop over the possible changes for the first pair of bits
    for new_pair1, prob1 in change_probs[pair1].items():
        # Loop over the possible changes for the second pair of bits
        #for new_pair2, prob2 in change_probs[pair2].items():
            # Concatenate the new pairs of bits to form a new string
            new_string = input_string[0:index] + new_pair1  + input_string[index+2:]

            # Compute the probability of the new string as the product of the change probabilities
            new_prob = input_value * prob1 #* prob2

            # Add the new string and probability to the output dictionary
            if new_string in output_dict.keys():
                output_dict[new_string]+= new_prob
            else:
                output_dict[new_string] = new_prob
            #print(new_pair1+new_pair2)
            #print(new_string)
            #print(output_dict[new_string])
    # Return the output dictionary
    return output_dict

def combine_dicts(input_list):
    # Initialize the output dictionary as an empty dictionary
    output_dict = {}

    # Loop over the dictionaries in the input list
    for dic in input_list:
        # Loop over the keys and values in each dictionary
        for key, value in dic.items():
            # Add the value to the output dictionary for the same key
            # If the key does not exist in the output dictionary, use 0 as the default value
            output_dict[key] = output_dict.get(key, 0) + value

    # Return the output dictionary
    return output_dict
def add_cross_talk_noise(counts,index):
    list_data_dict=[]
    for key in counts.keys():
        new_dict=change_string(key,counts[key],index)
        list_data_dict.append(new_dict)
        
    output=combine_dicts(list_data_dict)
    return output