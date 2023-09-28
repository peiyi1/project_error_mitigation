import numpy as np
from qiskit.converters import circuit_to_dag, dag_to_circuit, circuit_to_dagdependency
import rustworkx as rx
import itertools
from qiskit.dagcircuit.dagnode import DAGNode, DAGOpNode, DAGInNode, DAGOutNode
import itertools
def norm_dict(d):
    # Assuming the dictionary values are numeric
    # Calculate the sum of all values
    total = sum(d.values())
    # Create a new dictionary to store the normalized values
    normalized = {}
    # Loop through the original dictionary
    for key, value in d.items():
        # Divide each value by the total and store it in the new dictionary
        normalized[key] = value / total
    # Return the normalized dictionary
    return normalized

def bit_weight(dist, index):
    #bitwise distribution
    weight_0 = 0
    weight_1 = 0
    for key in dist.keys():
        if key[len(key) - 1 - index] == '0':
            weight_0 += dist[key]
        elif key[len(key) - 1 - index] == '1':
            weight_1 += dist[key]
        else:
            print("Incorrect key value")
    return weight_0, weight_1

def update_dist(unmiti_dist, miti_dist, index):
    Ppost = {}
    w0, w1 = bit_weight(miti_dist, index)
    u_w0, u_w1 = bit_weight(unmiti_dist, index)
    if w0 == 0:
        w0 = 0.0000000000001
        w1 = 0.9999999999999
    if w1 == 0:
        w1 = 0.0000000000001
        w0 = 0.9999999999999
    if u_w0 == 0:
        u_w0 = 0.0000000000001
        u_w1 = 0.9999999999999
    if u_w1 == 0:
        u_w1 = 0.0000000000001
        u_w0 = 0.9999999999999
        
    for key in unmiti_dist.keys():
        if key[len(key) - 1 - index] == '0':
            Ppost[key] = unmiti_dist[key] / u_w0 * (w0)# / w1)
            #print(w0, w1, w0/w1, Ppost[key])
        elif key[len(key) - 1 - index] == '1':
            Ppost[key] = unmiti_dist[key] / u_w1 * (w1)# / w0)
            #print(w0, w1, w1/w0, Ppost[key])
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
            ppost[i] = update_dist(temp_dist, miti_dist_list[i][0], miti_dist_list[i][1])
        temp_dist = combine_dist(temp_dist, ppost)
        temp_dist = norm_dict(temp_dist)
        h_dist = H_distance_dict(temp_dist, temp_dist_start)
        print("H-dist:", h_dist)
    return temp_dist

def transform_qubit_dict(input_dict):
    output_dict = {}
    for key, values in input_dict.items():
        # Assuming the key is a Qubit object, we get its index
        key_index = key.index
        # Each value is a list of tuples containing an integer and a Qubit object
        tuple_values = []
        for value in values:
            value_index = value[1].index
            tuple_values.append((value[0], value_index))
        output_dict[key_index] = tuple_values
    return output_dict
def transform_dict(input_dict, index):
    output_dict = {}
    for key, values in input_dict.items():
        for value in values:
            if value[0]==index:
                output_dict[value[1]] = key
    return dict(sorted(output_dict.items()))

def update_dict(d, length):
    # create a list of all possible binary keys of given length
    keys = [''.join(map(str, key)) for key in itertools.product([0, 1], repeat=length)]

    # loop through the keys
    for key in keys:
        # if the key is not in the dict, add it with value 0
        if key not in d:
            d[key] = 0

    # Calculate the sum of all values
    total = sum(d.values())

    # If total is not zero, normalize the values
    if total != 0:
        for key in d:
            d[key] /= total

    # return the updated dict
    return d


def get_ancestors_circuit(input_qc, qubit_id, node_id):
    # Convert the input circuit to a directed acyclic graph (DAG)
    qc_dag = circuit_to_dag(input_qc)
    
    # Get the generator for the nodes on the specified qubit wire
    gen = qc_dag.nodes_on_wire(qc_dag.wires[qubit_id])
    
    # Get the node at the specified node index from the generator
    node = next(itertools.islice(gen, node_id, None))
    
    # Find the ancestors of the node using rustworkx
    anc = rx.ancestors(qc_dag._multi_graph, node._node_id)
    
    # Convert the set of ancestor indices to a set of ancestor nodes
    anc_set = {node}
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
    return trimmed_qc

# pauli I matrix
observable_i = np.array([[1, 0 ],
                       [0, 1]])

# pauli Z matrix
observable_z = np.array([[1, 0, ],
                       [0, -1]])


# pauli X matrix
observable_x = np.array([[0, 1, ],
                       [1, 0]])

# pauli Y matrix
observable_y = np.array([[0, -1j, ],
                       [1j, 0]])

initial_rho=np.array([[1, 0], [0, 0]])

# Define the Hadamard gate
Hgates = (1/np.sqrt(2)) * np.array([[1, 1], 
                               [1, -1]])
def obtain_matrix_x():
    matrix= [[0,1], [1,0]]
    return matrix

def obtain_matrix_y():
    matrix= [[0, -1j],[1j, 0]]
    return matrix

def obtain_matrix_z():
    matrix= [[1,0],[0,-1]]
    return matrix

def obtain_matrix_i():
    matrix= [[1,0],[0,1]]
    return matrix

def obtain_matrix_ii():
    matrix= np.kron(obtain_matrix_i(),obtain_matrix_i())
    return matrix

def obtain_matrix_ix():
    matrix= np.kron(obtain_matrix_i(),obtain_matrix_x())
    return matrix

def obtain_matrix_iy():
    matrix= np.kron(obtain_matrix_i(),obtain_matrix_y())
    return matrix

def obtain_matrix_iz():
    matrix= np.kron(obtain_matrix_i(),obtain_matrix_z())
    return matrix

def obtain_matrix_xi():
    matrix= np.kron(obtain_matrix_x(),obtain_matrix_i())
    return matrix
def obtain_matrix_xx():
    matrix= np.kron(obtain_matrix_x(),obtain_matrix_x())
    return matrix
def obtain_matrix_xy():
    matrix= np.kron(obtain_matrix_x(),obtain_matrix_y())
    return matrix
def obtain_matrix_xz():
    matrix= np.kron(obtain_matrix_x(),obtain_matrix_z())
    return matrix

def obtain_matrix_yi():
    matrix= np.kron(obtain_matrix_y(),obtain_matrix_i())
    return matrix
def obtain_matrix_yx():
    matrix= np.kron(obtain_matrix_y(),obtain_matrix_x())
    return matrix
def obtain_matrix_yy():
    matrix= np.kron(obtain_matrix_y(),obtain_matrix_y())
    return matrix
def obtain_matrix_yz():
    matrix= np.kron(obtain_matrix_y(),obtain_matrix_z())
    return matrix

def obtain_matrix_zi():
    matrix= np.kron(obtain_matrix_z(),obtain_matrix_i())
    return matrix
def obtain_matrix_zx():
    matrix= np.kron(obtain_matrix_z(),obtain_matrix_x())
    return matrix
def obtain_matrix_zy():
    matrix= np.kron(obtain_matrix_z(),obtain_matrix_y())
    return matrix
def obtain_matrix_zz():
    matrix= np.kron(obtain_matrix_z(),obtain_matrix_z())
    return matrix

rho_list_16=[obtain_matrix_ii(),obtain_matrix_ix(),obtain_matrix_iy(),obtain_matrix_iz(),obtain_matrix_xi(),obtain_matrix_xx(),obtain_matrix_xy(),obtain_matrix_xz(),obtain_matrix_yi(),obtain_matrix_yx(),obtain_matrix_yy(),obtain_matrix_yz(),obtain_matrix_zi(),obtain_matrix_zx(),obtain_matrix_zy(),obtain_matrix_zz()]

def obtain_density_matrix_two_qubit(trace_list):
    rho_list=rho_list_16
    cof = trace_list[0]*0.25
    sum_inter=np.multiply(rho_list[0],cof)
    for i in range(1,16):
        cof = trace_list[i]*0.25
        rho=np.multiply(rho_list[i],cof)
        sum_inter=np.add(sum_inter,rho)
    return sum_inter



def obtain_density_matrix(trace_list):
    rho_list=[observable_i,observable_x,observable_y,observable_z]
    cof = trace_list[0]*0.5
    sum_inter=np.multiply(rho_list[0],cof)
    for i in range(1,4):
        cof = trace_list[i]*0.5
        rho=np.multiply(rho_list[i],cof)
        sum_inter=np.add(sum_inter,rho)
    return sum_inter

def obtain_h_gate():
    return Hgates

def obtain_ry_matrix(theta):
    """
    Returns the matrix representation of the RY gate for a given angle theta.
    
    Parameters:
    - theta (float): The rotation angle.
    
    Returns:
    - numpy.ndarray: The 2x2 matrix representation of the RY gate.
    """
    return np.array([
        [np.cos(theta / 2), -np.sin(theta / 2)],
        [np.sin(theta / 2), np.cos(theta / 2)]
    ])

def obtain_info_for_four_obs( U,rho=initial_rho):
    
    # Calculate the final density matrix after applying the quantum gate
    rho_final = U @ rho @ U.conj().T
    
    tr_i = np.trace(np.dot(observable_i, rho_final))
    tr_z = np.trace(np.dot(observable_z, rho_final))
    tr_x = np.trace(np.dot(observable_x, rho_final))
    tr_y = np.trace(np.dot(observable_y, rho_final))

    tr_list=[tr_i,tr_i,tr_x,tr_x,tr_y,tr_y,tr_z,tr_z]
    
    return tr_list

def obtain_info_for_prep_circ(counts_list,obs):
    tr_list=[]
    if (obs=='I'):
        for counts in counts_list:
            new_counts=update_dict(counts,1)
            tr=new_counts['0']+ new_counts['1'] 
            tr_list.append(tr)
    else:
        for counts in counts_list:
            new_counts=update_dict(counts,1)
            tr=new_counts['0'] - new_counts['1'] 
            tr_list.append(tr)
            
    tr1=tr_list[0]
    tr2=tr_list[1]
    tr3=tr_list[2]
    tr5=tr_list[3]
    
    tr7 = tr1 
    tr8 = tr2
    tr4=tr1+tr2-tr3
    tr6=tr1+tr2-tr5
    new_tr_list=[tr1,tr2,tr3,tr4,tr5,tr6,tr7,tr8]
    
    return   new_tr_list     

def obtain_trace(counts_list,obs):
    tr_list=[]
    if (obs=='I'):
        for counts in counts_list:
            new_counts=update_dict(counts,1)
            tr=new_counts['0']+ new_counts['1'] 
            tr_list.append(tr)
    else:
        for counts in counts_list:
            new_counts=update_dict(counts,1)
            tr=new_counts['0'] - new_counts['1'] 
            tr_list.append(tr)
    return tr_list


def obtain_eigenvalue_list():
    eigenvalue=[1,1,1,-1,1,-1,1,-1]
    return eigenvalue

def obtain_final_trace(measurement_part_trace,prepare_part_trace):
    trace=0
    eigenvalue=obtain_eigenvalue_list()
    for i in range(8):
        trace+=(eigenvalue[i]*prepare_part_trace[i]*measurement_part_trace[i])/2
    return trace

def split_list_into_sublists(input_list, num_sublists):
    sublist_size = len(input_list) // num_sublists
    sublists = [input_list[i * sublist_size:(i + 1) * sublist_size] for i in range(num_sublists)]
    return sublists

def update_output_based_on_info(counts, tr):
    p1=(1-tr)/2
    #print(p1)
    new_counts=update_dict(counts,2)
    r=(new_counts['01']+new_counts['11'])/p1
    new_dist={}
    new_dist['01']=new_counts['01']/r
    new_dist['11']=new_counts['11']/r
    new_dist['00']=(new_counts['00']-(1-r)/r*new_counts['01'])
    new_dist['10']=(new_counts['10']-(1-r)/r*new_counts['11'])
    return new_dist

def obtain_trace_two_obs(counts,prep_state,obs1,obs2):
    #obs1=['I','X','Y','Z']
    #prep_state=['I_p','I_n','X_p','X_n',Y_p',Y_n',Z_p',Z_n']
    #obs2=['I','X','Y','Z']
    dist={}
    key_list=['0','1']
    if (obs1=='I'):
        for key in key_list:
            dist[key]=counts[key+'0']+counts[key+'1']
    else:
        for key in key_list:
            dist[key]=counts[key+'0']-counts[key+'1']
            
    new_dist={}
    if((prep_state=='X_n') or (prep_state=='Y_n') or (prep_state=='Z_n')):
        for key in dist.keys():
            new_dist[key]=(-1)*dist[key]
    else:
        new_dist=dist
        
    tr=0
    if (obs2=='I'):
        tr=new_dist['0'] + new_dist['1'] 
    else:
        tr=new_dist['0'] - new_dist['1'] 
            
    return tr

def calculate_trace_two_counts(counts1,counts2,obs1,obs2,prep_state_sum):
    #do not add eigenvalue of the prep_state
    dist1={}
    dist2={}
    key_list=['0','1']
    if (obs1=='I'):
        for key in key_list:
            dist1[key]=counts1[key+'0']+counts1[key+'1']
            dist2[key]=counts2[key+'0']+counts2[key+'1']
    else:
        for key in key_list:
            dist1[key]=counts1[key+'0']-counts1[key+'1']
            dist2[key]=counts2[key+'0']-counts2[key+'1']
            
        
    tr1=0
    tr2=0
    if (obs2=='I'):
        tr1=dist1['0'] + dist1['1'] 
        tr2=dist2['0'] + dist2['1'] 
    else:
        tr1=dist1['0'] - dist1['1'] 
        tr2=dist2['0'] - dist2['1'] 
    
    tr=0
    if(prep_state_sum=='I'):
        tr=tr1+tr2
    else:
        tr=tr1-tr2
    return tr
        
    
    
########################################################################
def collapse_bit_distribution(distribution, bit_position):
    collapsed_distribution = {'0': 0, '1': 0}

    for key, count in distribution.items():
        bit_value = key[bit_position]
        if bit_value == '0':
            collapsed_distribution['0'] += count
        elif bit_value == '1':
            collapsed_distribution['1'] += count

    return collapsed_distribution

