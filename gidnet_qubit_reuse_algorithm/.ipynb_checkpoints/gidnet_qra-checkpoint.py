import numpy as np
import math
import copy

from qiskit_qubit_reuse import qubit_reuse, qubit_reuse_greedy
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.visualization import dag_drawer

from qiskit.dagcircuit.exceptions import DAGCircuitError
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.dagcircuit.dagnode import DAGInNode, DAGNode, DAGOpNode, DAGOutNode
from qiskit.circuit.quantumregister import QuantumRegister, Qubit
from qiskit.circuit.classicalregister import ClassicalRegister, Clbit
from qiskit.circuit.instruction import Instruction

from typing import List, Tuple, Optional, Union, Any

import networkx as nx
import retworkx as rx
from networkx import DiGraph, MultiDiGraph



def to_networkx(circuit_dag):
    """Returns a copy of the DAGCircuit in networkx format."""
    try:
        import networkx as nx
    except ImportError as ex:
        raise ImportError("Networkx is needed to use to_networkx(). It "
                          "can be installed with 'pip install networkx'") from ex
    G = nx.MultiDiGraph()
    for node in circuit_dag._multi_graph.nodes():
        G.add_node(node)
    for node_id in rx.topological_sort(circuit_dag._multi_graph):
        for source_id, dest_id, edge in circuit_dag._multi_graph.in_edges(node_id):
            G.add_edge(circuit_dag._multi_graph[source_id],
                       circuit_dag._multi_graph[dest_id],
                       wire=edge)
    return G



def from_networkx(graph):
    
    """Take a networkx MultiDigraph and create a new DAGCircuit.

    Args:
        graph (networkx.MultiDiGraph): The graph to create a DAGCircuit
            object from. The format of this MultiDiGraph format must be
            in the same format as returned by to_networkx.

    Returns:
        DAGCircuit: The dagcircuit object created from the networkx
            MultiDiGraph.
    Raises:
        MissingOptionalLibraryError: If networkx is not installed
        DAGCircuitError: If input networkx graph is malformed
    """
    try:
        import networkx as nx
    except ImportError as ex:
        raise MissingOptionalLibraryError(
            libname="Networkx",
            name="DAG converter from networkx",
            pip_install="pip install networkx",
        ) from ex
    dag = DAGCircuit()
    for node in nx.topological_sort(graph):
        if isinstance(node, DAGOutNode):
            continue
        if isinstance(node, DAGInNode):
            if isinstance(node.wire, Qubit):
                dag.add_qubits([node.wire])
            elif isinstance(node.wire, Clbit):
                dag.add_clbits([node.wire])
            else:
                raise DAGCircuitError(f"unknown node wire type: {node.wire}")
        elif isinstance(node, DAGOpNode):
            dag.apply_operation_back(node.op.copy(), node.qargs, node.cargs)
    return dag



def get_biadjacency_candidate_matrix(static_circuit):
    """
	Get the biadjacency and candidate matrices of the simplified bipartite graph 
	from the quantum circuit by searching for connections between the input qubits
	and the output qubits of the quantum circuit.

    Args:
        Qiskit QuantumCircuit object

    Returns:
        (numpy.ndarray, numpy.ndarray): the biadjacency and candidate matrices of 
		the simplified bipartite graph corresponding to a quantum circuit
    """
	
    circuit = copy.deepcopy(static_circuit)
    circuit.remove_final_measurements() # remove all circuit final measurements and barriers
    
    # convert circuit to qiskit DAG
    circ_dag = circuit_to_dag(circuit)
    
    graph = to_networkx(circ_dag) # convert from Qiskit DAG to Networkx DAG
    roots = list(circ_dag.input_map.values())  # the roots or input nodes of the circuit
    terminals = list(circ_dag.output_map.values()) # the output nodes of the circuit
	
    # Initialize the biadjacency matrix of an empty quantum circuit
    biadjacency_matrix = np.zeros((len(roots), len(terminals)), dtype=int)
    # For each root-terminal pair, if there is a path from root to terminal,
    # then the entry corresponding to this root-terminal pair will be one otherwise zero.
    for i, root in enumerate(roots):
        for j, terminal in enumerate(terminals):
            if nx.has_path(graph, root, terminal):
                biadjacency_matrix[i][j] += 1

    # Calculate the candidate matrix from the biadjacency matrix
    candidate_matrix = np.ones((len(terminals), len(roots)), dtype=int) - biadjacency_matrix.transpose()

    return biadjacency_matrix, candidate_matrix


def merge_sublists(list_of_pairs):
    """
    Merges sublists in a list of pairs such that each merged list contains all elements
    that are connected directly or indirectly through common elements, maintaining
    the original ordering of elements. This function is useful for identifying interconnected
    components or groups within a list of pairs, where a pair represents a direct connection
    between two elements.

    Parameters:
    - list_of_pairs (list of lists): A list where each sublist contains a pair of integers
      representing a direct connection.

    Returns:
    - list of lists: Merged sublists based on common elements, preserving original order.
      Each sublist represents a group of interconnected elements.
    """

    def find_merge_index(merged_list, pair):
        """
        Finds the index of the sublist within merged_list that shares a common element with the given pair.
        Returns -1 if no common element is found.

        Parameters:
        - merged_list (list of lists): The current list of merged sublists.
        - pair (list): The pair of elements to find a merge candidate for.

        Returns:
        - int: The index of the sublist in merged_list that has a common element with pair, or -1 if none.
        """
        for i, sublist in enumerate(merged_list):
            if any(elem in sublist for elem in pair):
                return i
        return -1

    def merge_and_order(sublist1, sublist2):
        """
        Merges two sublists into one, ensuring that elements from sublist2 are inserted
        into sublist1 in their relative order, maintaining the original ordering of elements.

        Parameters:
        - sublist1 (list): The first sublist to merge.
        - sublist2 (list): The second sublist to merge.

        Returns:
        - list: The merged and ordered sublist.
        """
        # Create a set for faster lookups
        sublist1_set = set(sublist1)
        for elem in sublist2:
            if elem not in sublist1_set:
                sublist1_set.add(elem)
                # Determine the position to insert the element based on relative order in sublist2
                position = next((i for i, x in enumerate(sublist1) if x in sublist2 and sublist2.index(x) > sublist2.index(elem)), len(sublist1))
                sublist1.insert(position, elem)
        return sublist1

    merged_list = []
    for pair in list_of_pairs:
        merge_index = find_merge_index(merged_list, pair)
        if merge_index != -1:
            # Merge the pair into the found sublist
            merged_list[merge_index] = merge_and_order(merged_list[merge_index], pair)
        else:
            # No common element found, add the pair as a new sublist
            merged_list.append(pair)

    # Additional merging passes to ensure all interconnected sublists are fully merged
    merge_occurred = True
    while merge_occurred:
        merge_occurred = False
        for i in range(len(merged_list)):
            for j in range(i+1, len(merged_list)):
                if any(elem in merged_list[j] for elem in merged_list[i]):
                    # Merge i-th and j-th sublists and remove the j-th sublist
                    merged_list[i] = merge_and_order(merged_list[i], merged_list[j])
                    del merged_list[j]
                    merge_occurred = True
                    break
            if merge_occurred:
                break

    return merged_list




def add_qubit_reuse_edges(circuit, qubit_reuse_list):
    """
    Given the original circuit, and the a list, qubit_reuse_list, that 
    contains the ordering of qubits in the dynamic circuit, the function 
    returns a DAG that is a modified version of the original circuit DAG.
    This modified DAG is a copy of the original DAG, but with edges that 
    connects the reused qubits to the corresponding qubits that reused them
    """
    # this should be done before adding connecting edges.
    circ_dag = circuit_to_dag(circuit)
    # display(circ_dag.draw())
    circ_dag_copy = copy.deepcopy(circ_dag)
    total_qubits = len(circ_dag_copy.qubits)


    # Create Edges
    roots = list(circ_dag_copy.input_map.values())  # the roots or input nodes of the circuit
    terminals = list(circ_dag_copy.output_map.values()) # the output nodes of the circuit

    new_edges = []
    for qubit_list in qubit_reuse_list:
        if len(qubit_list)>0:
            # Group the qubit indices to form edges
            new_edges_indx = [(qubit_list[i], qubit_list[i + 1]) for 
                              i in range(len(qubit_list) - 1)]
            for root, terminal in new_edges_indx:
                # the root here was the terminal before.
                # The terminal is the qubit that will be 
                # reused on the root
                new_edges.append((terminals[root], roots[terminal]))

    # display(circ_dag_copy.draw())             
    for inp_node, outp_node in new_edges:
        circ_dag_copy._multi_graph.add_edge(inp_node._node_id, outp_node._node_id, inp_node.wire)
                
    return circ_dag_copy, new_edges


def add_reset_op(qargs):
    """This function creates the reset operation to be added to DAG"""
	# reset operation has no cargs
    reset_op = Instruction(name='reset', num_qubits=1, num_clbits=0, params=[])
    reset_node = DAGOpNode(op=reset_op, qargs=qargs, cargs=())
    return reset_node


def generate_dynamic_circuit(circuit, qubit_reuse_list, draw=False):
    """
    Converts a static quantum circuit into a dynamic quantum circuit by adding edges to the
    original Directed Acyclic Graph (DAG) of the circuit and reordering qubits based on 
    the qubit reuse list. This transformation is aimed at optimizing qubit reuse and improving 
    the efficiency of quantum circuit execution.

    The function first computes qubit reuse sets using the `compute_qubit_reuse_sets` function. 
    It then adds edges to the original DAG circuit based on these reuse sets and removes any 
    barrier nodes to avoid cyclical DAGs and enable topological sorting. The qubits of the 
    original circuit are reordered into a new dynamic circuit based on the qubit reuse list.

    Parameters:
    - circuit (QuantumCircuit): The static quantum circuit to be transformed into a dynamic circuit.
    - iqnet (bool, optional): If True, uses an improved qnet algorithm for qubit reuse. Defaults to False.
    - draw (bool, optional): If True, displays the generated dynamic circuit using Matplotlib. Defaults to False.

    Returns:
    - QuantumCircuit: The transformed dynamic quantum circuit.

    The function also handles the edge case where a cycle might be detected in the DAG after adding new edges.
    In such cases, an AssertionError is raised. For each node in the DAG, the function applies operations
    based on the remapped qubits, ensuring the dynamic circuit maintains the intended functionality of the
    original static circuit.
    """

    # qubit_reuse_list = compute_qubit_reuse_sets(circuit)

    circ_dag_copy, new_edges = add_qubit_reuse_edges(circuit, qubit_reuse_list) # add edges using the qubit_reuse_list

	# remove barrier nodes from the new dag to avoid cyclical DAG and enable topological sorting
    barrier_nodes = [node for node in list(circ_dag_copy.nodes()) if isinstance(node, DAGOpNode)  and node.op.name == "barrier"]

    for op_node in barrier_nodes:
        circ_dag_copy.remove_op_node(op_node)

    # Check if there is a cycle in the DAG
    if len(list(circ_dag_copy.nodes())) > len(list(circ_dag_copy.topological_nodes())):
        assert False, "A cycle detected in the DAG"

	
    old_to_new_qubit_remap_dict = {} # map old qubits to new qubits
    for new_qubit in range(len(qubit_reuse_list)):
        for old_qubit in qubit_reuse_list[new_qubit]:
            old_to_new_qubit_remap_dict[old_qubit] = new_qubit

    # loop through the nodes in the circuit DAG
    new_dag_copy = DAGCircuit()
	# Add quantum registers to new_dag_copy
    num_qubits = len(qubit_reuse_list) # number of qubits is the length of the qubit_reuse_list
    qregs = QuantumRegister(num_qubits)
    new_dag_copy.add_qreg(qregs)
    qubits_indices = {i: qubit for i, qubit in enumerate(new_dag_copy.qubits)}  # Assign index to each clbit from 0 to total number of 

    # Add classical registers to new_dag_copy
    num_clbits = circ_dag_copy.num_clbits()
    cregs = ClassicalRegister(num_clbits)
    new_dag_copy.add_creg(cregs)
    clbits_indices = {i: clbit for i, clbit in enumerate(new_dag_copy.clbits)}  # Assign index to each clbit from 0 to total number of clbits-1
    
    total_qr = len(qubit_reuse_list) # total number of qubits in dynamic circuit
    measure_reset_nodes = [edge[0].__hash__() for edge in new_edges] # nodes to replace with measure and reset
    remove_input_nodes = [edge[1].__hash__() for edge in new_edges] # nodes to replace with measure and reset
    track_qubits = []

    for node in circ_dag_copy.topological_nodes(): # it is important to organize it topologically
        if not isinstance(node, DAGInNode) and not isinstance(node, DAGOutNode):
            old_qubits = [n.index for n in node.qargs]
            clbits = [n.index for n in node.cargs]

            if len(old_qubits) > 1: # this is the two-qubit gate case
                new_qubits = [old_to_new_qubit_remap_dict[q] for q in old_qubits]
                new_wires = [qubits_indices[q] for q in new_qubits]
                new_qargs = tuple(new_wires)

                clbits_wires = [clbits_indices[c] for c in clbits]
                new_cargs = tuple(clbits_wires)

                new_node = DAGOpNode(op=node.op, qargs=new_qargs, cargs=new_cargs)
                new_dag_copy.apply_operation_back(new_node.op, qargs=new_node.qargs, cargs=new_node.cargs)

            else: # this is the one-qubit gate case
                old_qubits = old_qubits[0] # to eliminate list for 1-qubit
                new_qubits = old_to_new_qubit_remap_dict[old_qubits]
                new_wires = [qubits_indices[new_qubits]]
                new_qargs = tuple(new_wires)

                clbits_wires = [clbits_indices[c] for c in clbits]
                new_cargs = tuple(clbits_wires)
                new_node = DAGOpNode(op=node.op, qargs=new_qargs, cargs=new_cargs)
                new_dag_copy.apply_operation_back(new_node.op, qargs=new_node.qargs, cargs=new_node.cargs)
        else:
            old_qubits = node.wire.index
            new_qubits = old_to_new_qubit_remap_dict[old_qubits]
            new_wires = [qubits_indices[new_qubits]]
            new_qargs = tuple(new_wires)

            if isinstance(node, DAGInNode):
                continue # since we have accounted for the qubits
            elif isinstance(node, DAGOutNode):
                # DAGOutNode and hence, reset operation has no cargs
                if node.__hash__() in measure_reset_nodes:
                    reset_node = add_reset_op(new_qargs)
                    new_dag_copy.apply_operation_back(reset_node.op, qargs=reset_node.qargs, cargs=reset_node.cargs)
                else:
                    continue

    
    new_circ = dag_to_circuit(new_dag_copy)
    
    if draw:
        # display(new_circ.draw("mpl", fold=-1))
        display(new_circ.draw("mpl"))
    
    return new_circ


from biadu_qnet_qubit_reuse.baidu_qnet_qr import (compute_qnet_qubit_reuse_list, 
                                                  from_qiskit_to_qnet,
                                                 compute_qnet_qubit_reuse_list_timing)


def compile_dynamic_circuit(circuit, qnet_circuit=None, algorithm_type="gidnet", improved_gidnet=True, qnet_method="random", qnet_shots=1, maximize_gidnet_depth=False, draw=False):
    """
    Converts a static quantum circuit into a dynamic quantum circuit by adding edges to the
    original Directed Acyclic Graph (DAG) of the circuit and reordering qubits based on 
    the qubit reuse list. This transformation is aimed at optimizing qubit reuse and improving 
    the efficiency of quantum circuit execution.

    The function first computes qubit reuse sets using the `compute_qubit_reuse_sets` function. 
    It then adds edges to the original DAG circuit based on these reuse sets and removes any 
    barrier nodes to avoid cyclical DAGs and enable topological sorting. The qubits of the 
    original circuit are reordered into a new dynamic circuit based on the qubit reuse list.

    Parameters:
    - circuit (QuantumCircuit): The static quantum circuit to be transformed into a dynamic circuit.
    - iqnet (bool, optional): If True, uses an improved qnet algorithm for qubit reuse. Defaults to False.
    - draw (bool, optional): If True, displays the generated dynamic circuit using Matplotlib. Defaults to False.

    Returns:
    - QuantumCircuit: The transformed dynamic quantum circuit.
	- algorithm_type:  This is either "gidnet" or "qnet" for the sake of runtime comparism

    The function also handles the edge case where a cycle might be detected in the DAG after adding new edges.
    In such cases, an AssertionError is raised. For each node in the DAG, the function applies operations
    based on the remapped qubits, ensuring the dynamic circuit maintains the intended functionality of the
    original static circuit.
    """

    if algorithm_type == "gidnet":
        qubit_reuse_list = compute_qubit_reuse_sets(circuit, improved_gidnet=improved_gidnet, maximize_circuit_depth=maximize_gidnet_depth)
    elif algorithm_type == "qnet":
        qubit_reuse_list = compute_qnet_qubit_reuse_list(qnet_circuit, method=qnet_method, shots=qnet_shots)

    # Return the original circuit if no qubit reuse is found
    if len(qubit_reuse_list) == circuit.num_qubits:
        # print("This circuit is irreducible")
        return circuit

    circ_dag_copy, new_edges = add_qubit_reuse_edges(circuit, qubit_reuse_list) # add edges using the qubit_reuse_list

	# remove barrier nodes from the new dag to avoid cyclical DAG and enable topological sorting
    barrier_nodes = [node for node in list(circ_dag_copy.nodes()) if isinstance(node, DAGOpNode)  and node.op.name == "barrier"]

    for op_node in barrier_nodes:
        circ_dag_copy.remove_op_node(op_node)

    # Check if there is a cycle in the DAG
    if len(list(circ_dag_copy.nodes())) > len(list(circ_dag_copy.topological_nodes())):
        assert False, "A cycle detected in the DAG"

    # print("qubit_reuse_list length = ", len(qubit_reuse_list))
    old_to_new_qubit_remap_dict = {} # map old qubits to new qubits
    for new_qubit in range(len(qubit_reuse_list)):
        for old_qubit in qubit_reuse_list[new_qubit]:
            old_to_new_qubit_remap_dict[old_qubit] = new_qubit

    # print("old_to_new_qubit_remap_dict = ", old_to_new_qubit_remap_dict)

    # loop through the nodes in the circuit DAG
    new_dag_copy = DAGCircuit()
	# Add quantum registers to new_dag_copy
    num_qubits = len(qubit_reuse_list) # number of qubits is the length of the qubit_reuse_list
    qregs = QuantumRegister(num_qubits)
    new_dag_copy.add_qreg(qregs)
    qubits_indices = {i: qubit for i, qubit in enumerate(new_dag_copy.qubits)}  # Assign index to each clbit from 0 to total number of 

    # Add classical registers to new_dag_copy
    num_clbits = circ_dag_copy.num_clbits()
    cregs = ClassicalRegister(num_clbits)
    new_dag_copy.add_creg(cregs)
    clbits_indices = {i: clbit for i, clbit in enumerate(new_dag_copy.clbits)}  # Assign index to each clbit from 0 to total number of clbits-1
    
    total_qr = len(qubit_reuse_list) # total number of qubits in dynamic circuit
    measure_reset_nodes = [edge[0].__hash__() for edge in new_edges] # nodes to replace with measure and reset
    remove_input_nodes = [edge[1].__hash__() for edge in new_edges] # nodes to replace with measure and reset
    track_qubits = []

    for node in circ_dag_copy.topological_nodes(): # it is important to organize it topologically
        if not isinstance(node, DAGInNode) and not isinstance(node, DAGOutNode):
            old_qubits = [n.index for n in node.qargs]
            clbits = [n.index for n in node.cargs]

            if len(old_qubits) > 1: # this is the two-qubit gate case
                new_qubits = [old_to_new_qubit_remap_dict[q] for q in old_qubits]
                new_wires = [qubits_indices[q] for q in new_qubits]
                new_qargs = tuple(new_wires)

                clbits_wires = [clbits_indices[c] for c in clbits]
                new_cargs = tuple(clbits_wires)

                new_node = DAGOpNode(op=node.op, qargs=new_qargs, cargs=new_cargs)
                new_dag_copy.apply_operation_back(new_node.op, qargs=new_node.qargs, cargs=new_node.cargs)

            else: # this is the one-qubit gate case
                old_qubits = old_qubits[0] # to eliminate list for 1-qubit
                new_qubits = old_to_new_qubit_remap_dict[old_qubits]
                new_wires = [qubits_indices[new_qubits]]
                new_qargs = tuple(new_wires)

                clbits_wires = [clbits_indices[c] for c in clbits]
                new_cargs = tuple(clbits_wires)
                new_node = DAGOpNode(op=node.op, qargs=new_qargs, cargs=new_cargs)
                new_dag_copy.apply_operation_back(new_node.op, qargs=new_node.qargs, cargs=new_node.cargs)
        else:
            old_qubits = node.wire.index
            new_qubits = old_to_new_qubit_remap_dict[old_qubits]
            new_wires = [qubits_indices[new_qubits]]
            new_qargs = tuple(new_wires)

            if isinstance(node, DAGInNode):
                continue # since we have accounted for the qubits
            elif isinstance(node, DAGOutNode):
                # DAGOutNode and hence, reset operation has no cargs
                if node.__hash__() in measure_reset_nodes:
                    reset_node = add_reset_op(new_qargs)
                    new_dag_copy.apply_operation_back(reset_node.op, qargs=reset_node.qargs, cargs=reset_node.cargs)
                else:
                    continue

    
    new_circ = dag_to_circuit(new_dag_copy)
    
    if draw:
        # display(new_circ.draw("mpl", fold=-1))
        display(new_circ.draw("mpl"))
    
    return new_circ



#### GIDNET ALGORITHM BEGINS #####
import itertools
def update_candidate_matrix_after_selection(candidate_matrix, t, r):
    """
    Updates the candidate matrix after a qubit reuse operation is selected. This involves setting the selected
    row and column to zero to prevent further selections that could conflict with the current operation.
    we updated the candidate matrix by removing entries in the candidate matrix that can result in cycle in the 
    DAG of the resulting dynamic circuit if subsequently selected. Also, the update is meant to remove the 
    possibility of simultaneously reusing a qubit by more than one qubit at the same time. It also prevents the 
    possibility of two qubits being reused simultaneously by one qubit.

    Parameters:
    - C: The current state of the candidate matrix.
    - t: The index of the terminal (row) involved in the reuse operation.
    - r: The index of the root (column) involved in the reuse operation.

    Returns:
    - The updated candidate matrix.
    """
    row, col = candidate_matrix.shape
    # Identify nodes indirectly connected through the newly added edge
    root_set = {i for i in range(col) if candidate_matrix[t, i] == 0}
    terminal_set = {j for j in range(row) if candidate_matrix[j, r] == 0}

    # Set entries to 0 for any pair in the product of terminal_set and root_set
    for pair in itertools.product(terminal_set, root_set):
        candidate_matrix[pair[0], pair[1]] = 0
        

    # Set the entire row and column for the selected edge to 0
    candidate_matrix[t, :] = 0
    candidate_matrix[:, r] = 0

    return candidate_matrix



def finalize_qubit_reuse_list(qubit_reuse_list, num_qubits):
    """
    Finalizes the qubit reuse list by ensuring that all qubits are included, either as part of a reuse chain
    or as individual elements if they were not involved in any reuse operation.

    Parameters:
    - qubit_reuse_list: The current list of qubit reuse operations.
    - num_qubits: The total number of qubits in the quantum circuit.

    Returns:
    - The finalized qubit reuse list with all qubits accounted for.
    """
    included_qubits = set(qubit for sublist in qubit_reuse_list for qubit in sublist)
    all_qubits = set(range(num_qubits))
    missing_qubits = all_qubits - included_qubits

    for qubit in missing_qubits:
        qubit_reuse_list.append([qubit])  # Add missing qubits as individual sets.

    return qubit_reuse_list


def update_potential_r_for_t(candidate_matrix, t_reuse_path):
    """
    Finds common columns in candidate_matrix where all specified rows (elements in t_reuse_path) 
    have a value of 1.

    This is a subroutine for `maximum_qubit_reuse_paths` that updates the `potential_r_for_t` 
    given `t_reuse_path`. For example, if `t_reuse_path` = [t, r], we find columns in row r 
    where the values are 1 in both rows t and r in the candidate matrix. This columns will form 
    the new or updated `potential_r_for_t` that we can choose from subsequently.

    Parameters:
    - candidate_matrix (numpy.ndarray): The input candidate matrix showing potential qubit reuse opportunities.
    - t_reuse_path (list of int): The set of qubits that form a reuse path (chain) for which we want to find input qubits they share in common.

    Returns:
    - common_nodes: The indices of columns (input qubits) where all specified rows (output qubits) have a value of 1.
    """
    updated_potential_r_for_t = set(range(candidate_matrix.shape[1]))  # Initialize with all column indices
    # for each qubit, q in t_reuse_path, find the columns in the corresponding row, q of the 
    # candidate matrix where the entries are 1. Afterwards, find the columns that is common
    # to all the qubits in t_reuse_path.
    for row_index in t_reuse_path:
        current_row_columns = set(np.where(candidate_matrix[row_index, :] == 1)[0])
        updated_potential_r_for_t &= current_row_columns # find the columns (input qubits) in the candidate_matrix 
                                            # that have values of 1 in all the rows (output qubits) in t_reuse_path.
        if not updated_potential_r_for_t:  # Early exit if no common columns left
            return []
    return list(updated_potential_r_for_t)




def maximum_qubit_reuse_paths(candidate_matrix, t):
    """
    This helper function explores all possible qubit reuse paths for a specific qubit (or row) 't'
    in the candidate matrix and identifies the path with most qubits or optimal qubits for reuse. 
    It updates the candidate matrix to reflect the selected reuse paths (or edges).
    
    First, it determines the potential input qubits, r that can be used on qubit t after it has 
    completed its operations. Qubits r, are the corresponding columns in row t whose entries are
    1. These qubits are stored in potential_r_for_t. The qubits that make up the maximum reuse path
    are selected from these sets of qubits and stored in t_reuse_path.
    
    To determine the qubits that are parts of the maximum reuse path for t, it uses `update_potential_r_for_t`
    function. Qubit t is combined with each element in `potential_r_for_t` one at a time into a list. For
    example if we chose qubit r_0 from `potential_r_for_t`, then we have a `t_reuse_path`, [t, r_0]. This list shows 
    that qubit r_0 can be used on qubit t after qubit t has completed its operation. Remember that every qubit,
    r in `potential_r_for_t` has the potential of being used on t after t has completed its operation.
    The goal is to choose the qubit r, that gives a new `potential_r_for_t` with more qubits than that of others.
    
    To determine the new `potential_r_for_t` after r_0 is chose, we find columns in row r_0 where the values are
    1 in both rows t and r_0 in the candidate matrix. This columns will form the new or updated `potential_r_for_t`
    that we can choose from. This is done for all the qubits, r in `potential_r_for_t`. The qubit, r with the highest
    number of qubits in the updated `potential_r_for_t`, is chosen since it indicates potential for more qubits
    that can be used on t after t and r_0 have completed their operations. The next r is chosen from the 
    updated `potential_r_for_t`.
    
    For all the qubits in the updated `potential_r_for_t`, we repeat the above and choose the one with the highest 
    number of qubits in its updated `potential_r_for_t`. Say for example, we subsequently chose r_1 from the updated 
    `potential_r_for_t`, the updated `t_reuse_path` becomes [t, r_0, r_1]. The new updated `potential_r_for_t` will
    be all the columns (input qubits) in row r_1 with 1 as values in both rows t, r_0 and r_1. This is same as finding 
    the input qubits that are common to the output qubits, t, r_0 and r_1 in the original candidate matrix. We repeat 
    this process untill the `potential_r_for_t` is empty.
    
    Note that it is possible that for all the qubits in `potential_r_for_t` we couldn't find a common input qubit with 
    the qubits in `t_reuse_path`. This means that choosing any of the qubits in `potential_r_for_t` and including it in
    the `t_reuse_path` ends the reuse path for t. In that case, we just choose any of the qubits in `potential_r_for_t`.
    The default for this algorithm is to choose the first qubit in `potential_r_for_t` and add it to `t_reuse_path`.
    
    After the maximum `t_reuse_path` is determined for t, we updated the candidate matrix by removing entries in the 
    candidate matrix that can result in cycle in the DAG of the resulting dynamic circuit if subsequently selected. Also,
    the update is meant to remove the possibility of simultaneously reusing a qubit by more than one qubit at the same time.
    It also prevents the possibility of two qubits being reused simultaneously by one qubit. 
    

    Parameters:
    - candidate_matrix: the candidate matrix of the quantum circuit
    - t: The qubit to be reused. It corresponds to row t in the candidate matrix.

    Returns:
    - t_reuse_path: A list of qubits forming an optimal reuse path for `t`.
    - candidate_matrix: The updated candidate matrix after the maximum t_reuse_path is determined.
    """

    # Implementation details for exploring qubit reuse paths would go here.
    # This would involve selecting qubits from 'potential_r_for_t', updating 'C_copy' and 'B_copy',
    # and determining the 't_reuse_path' that represents the chosen path of qubit reuse.
    
    
    t_reuse_path = [t] # we store the maximum qubit reuse path for t here
    
    # The maximum qubit reuse path for t, t_reuse_path  starts with t. Subsequent nodes (or qubits) are
    # selected from the input qubits, r in potential_r_for_t. These are the qubits that have the potential
    # of being used on qubit t after qubit t has completed its operations. 
    # Qubits in potential_r_for_t corresponds to columns (or input qubits) in row t whose values are 1.
    potential_r_for_t = np.where(candidate_matrix[t, :] == 1)[0] # these are the potential root nodes for t. It if from here we make selections
    potential_r_for_t = list(potential_r_for_t) # convert to list
   
    # loop continues until potential_r_for_t is empty. That is there is no more qubits to select from and the
    # reuse path for t has terminated.
    while potential_r_for_t:
        # compute the updated potential_r_for_t (i.e. the qubits left in potential_r_for_t) for each qubit, potential_r in 
        # potential_r_for_t assuming potential_r was chosen to be part of t_reuse_path.
        potential_r_dict = {} # we store the updated potential_r_for_t for each qubit in potential_r_for_t in this dictionary
        for potential_r in potential_r_for_t: # loop through the elements in potential_r_for_t
            potential_t_reuse_path = t_reuse_path + [potential_r] # this is the potential reuse path for t if this r was chosen
            # compute the qubits that would be left in potential_r_for_t if this potential_r where included in t_reuse_path
            potential_r_dict[potential_r] = update_potential_r_for_t(candidate_matrix, potential_t_reuse_path) # this is updated potential_r_for_t if
                                                                                                               # potential_r was chosen

        # If choosing any of the qubits in potential_r_for_t leaves no qubits in updated potential_r_for_t.
        # This means choosing any qubit in potential_r_for_t terminates t_reuse_path. 
        are_all_empty = all(len(value) == 0 for value in potential_r_dict.values())

        # If choosing any qubit in potential_r_for_t terminates t_reuse_path, any of the qubits can be selected.
        # However, the default of this algorithm is to choose the first qubit in potential_r_for_t
        if are_all_empty:
            if len(potential_r_for_t)>0: # check if potential_r_for_t is not empty
                t_reuse_path.append(potential_r_for_t[0]) # choose the first qubit in potential_r_for_t
                potential_r_for_t.remove(potential_r_for_t[0]) # update potential_r_for_t by removing the chosen qubit
        
            # if there we found qubits that can be used on t, updated the candidate matrix
            if len(t_reuse_path) > 1: 
                edge_pair_list = [(t_reuse_path[i], t_reuse_path[i + 1]) for 
                                              i in range(len(t_reuse_path) - 1)]

                for terminal, root in edge_pair_list:
                    # Update the candiate matrix to avoid cycles and double nodes in the circuit DAG
                    candidate_matrix = update_candidate_matrix_after_selection(candidate_matrix, terminal, root)
                
            return t_reuse_path, candidate_matrix
        
        # If choosing any qubit in potential_r_for_t does not terminates t_reuse_path, we proceed
        # by choosing the the potential_r that give a reuse path which leaves behind the highers number
        # of qubits in the updated potential_r_for_t. In the event that we have more than one potential_r
        # with the same number of qubits in their updated potential_r_for_t, we proceed with another test.
        else:
            # find the potential node (qubit), potential_r with the maximum number of qubits left in updated 
            # potential_r_for_t, if it was selected.
            max_nodes = max(len(lst) for lst in potential_r_dict.values()) # this is the maximum number of qubits in updated potential_r_for_t
            potential_r_with_max_nodes = [key for key, lst in potential_r_dict.items() if len(lst) == max_nodes] # get the qubit with max potential_r_for_t

            # If there are more than one qubits in potential_t_with_max_nodes, we have to determine which one to choose.
            # This is done by computing the intersect of the updated potential_r_for_t for each qubits in potential_r_with_max_nodes
            # with the updated potential_r_for_t for the other qubits in potential_r_with_max_nodes. The qubit, with the largest 
            # intersect with the other qubits is selected. In order words, we want to select the qubit whose updated potential_r_for_t
            # has qubits that are more common across the updated potential_r_for_t of other qubits in potential_r_with_max_nodes.
            if len(potential_r_with_max_nodes) > 1:
                score = -np.inf # score to keep track of the highest intersection
                chosen_r = None # the potential_r with the highest intersection with the other r's in potential_r_with_max_nodes
                for r in potential_r_with_max_nodes: # loop through the potential_r in potential_r_with_max_nodes
                    potential_r_for_t_intersections = [len(set(potential_r_dict[r]).intersection(potential_r_dict[i])) for i in 
                                    potential_r_with_max_nodes if i != r]  # find the intersection of the updated potential_r_for_t
                                                                           # for qubit r with the updated potential_r_for_t for each 
                                                                           # qubit in potential_r_with_max_nodes

                    total_interset = sum(potential_r_for_t_intersections)  # How many qubits in potential_r_with_max_nodes does it 
                                                                           # share common qubits with
                    
                    # we want to choose the qubit r in potential_r_with_max_nodes with the highest intersect
                    if total_interset > score:
                        chosen_r = r  # the chosen r with the maximum intersect
                        potential_r_for_t = potential_r_dict[chosen_r] # this is the updated potential_r_for_t if r was chosen
                        score = total_interset # keep track of the maximum intersect

                # If we found a suitable r, add it to t_reuse_path
                if chosen_r is not None:
                    t_reuse_path.append(chosen_r) # update the qubit reuse path, t_reuse_path
                    
                    # If after add the suitable or chosen r to t_reuse_path, we are left with only one qubit in the updated
                    # potential_r_for_t, just add that qubit to t_reuse_path. It shows that that qubit terminates the qubit
                    # reuse path, t_reuse_path.
                    if len(potential_r_for_t) == 1: # this accounts for the last element in potential_r_for_t
                        t_reuse_path.append(potential_r_for_t[0]) # update the qubit reuse path, t_reuse_path
                        potential_r_for_t = [] # update potential_r_for_t to an empty list since no more qubit to select from
            
            # If there is only one qubits in potential_t_with_max_nodes, we have just choose that qubit.
            else:
                chosen_r = potential_r_with_max_nodes[0] # choose the only qubit in potential_t_with_max_nodes
                t_reuse_path.append(chosen_r) # update the qubit reuse path, t_reuse_path
                potential_r_for_t = potential_r_dict[chosen_r] # get the updated potential_r_for_t
                
                # If after add the suitable or chosen r to t_reuse_path, we are left with only one qubit in the updated
                # potential_r_for_t, just add that qubit to t_reuse_path. It shows that that qubit terminates the qubit
                # reuse path, t_reuse_path.
                if len(potential_r_for_t) == 1: # this accounts for the last element in potential_r_for_t
                        t_reuse_path.append(potential_r_for_t[0]) # update the qubit reuse path, t_reuse_path
                        potential_r_for_t = []  # update potential_r_for_t to an empty list since no more qubit to select from


    # if we found qubit for the reuse path of t, that is, the length of t_reuse_path is greater than 1,
    # We have to update the candidate matrix to reflect this our choices and eliminate cycles in the DAG.
    if len(t_reuse_path) > 1:
        edge_pair_list = [(t_reuse_path[i], t_reuse_path[i + 1]) for 
                                      i in range(len(t_reuse_path) - 1)]

        for terminal, root in edge_pair_list:
            candidate_matrix = update_candidate_matrix_after_selection(candidate_matrix, terminal, root) # update made to the candidate matrix

    return t_reuse_path, candidate_matrix





def gidnet_qubit_reuse_algorithm(circ, initial_qubit=0):
    """
    Computes qubit reuse sets for a given quantum circuit. The biadjacency matrix, B is the matrix
    representation of the biadjacency graph of the quantum circuit. It is compute along side the candidate
    matrix, C given by: C = np.ones(n, n) - B.T, from the quantum circuit. The biadjacency graph represents
    the connections of and input qubit to the output qubits. The rows and columns of the biadjacency matrix
    are the input qubits (or root nodes) and output qubits (terminal nodes) respectively. An entry in the 
    matrix, (r, t) with value 1, indicates that there is a connection between input qubit r and output qubit
    t, while a zero entery indicates no connection. On the other other hand, the rows and columns of the 
    Candidate matrix, C are output qubits (terminal nodes) and input qubits (or root nodes). This is the 
    opposite of the biadjacency matrix because of the transpose in the equation relationship. An entry (t, r)
    in the candidate matrix with value 1, indicates that qubit r could be used on qubit t after qubit t has
    completed its operations. In order words, it indicates a potential reuse opportunity on qubit t.
    It is important to note that the first qubit selected (i.e `initial_qubit`) is very critical for optimal
    qubit reuse outcome.
    
    After computing the biadjacency and candidate matrices, the algorithm begins by selecting one of the qubits
    (rule not yet determined, but we can use the argument `initial_qubit` to specify the qubit we want to 
    begin with). It computes the maximum reuse for that qubit using `maximum_qubit_reuse_paths`. This is done
    by assuming all other qubits are not reusable and only reusing the qubit of interest, `initial_qubit` until 
    we can no longer reuse it. Note that the selected qubit is one of the rows of the candidate matrix whose
    sum of its entries is greater than zero. That is, there is at least one non-zero entry in that row.
    
    After the first selection, the candidate matrix is updated. Subsequent qubit selections are taken from 
    the first row of the candidate matrix whose sum is greater than zero. This is conditioned on the fact that 
    the rows are arranged in ascending order (a better and more optimal rule is possible, but available yet).
    The next qubit selected undergoes the same process of determining its maximum qubit reuse path (i.e. all the 
    qubits that can be used on it). This process repeats until the candidate matrix is entirely zeros, indicating 
    that no further qubit reuse opportunities exist.

    Parameters:
    - circ: The QuantumCircuit object from which the biadjacency and candidate matrices are derived.

    Returns:
    - qubit_reuse_list: A list of sublists, where each sublist contains indices of qubits that form a chain
                        of reuse operations, based on the iterative selection and updating process.
    """
    # compute the biadjacency matrix and candidate matrix from the circuit
    biadjacency_matrix, candidate_matrix = get_biadjacency_candidate_matrix(circ)
    
    # This is used to store the qubit reuse sets which is a list of sublists
    qubit_reuse_list = []
    
    counter = 0 # used to keep track of the initial_qubit selection, `initial_qubit`

    # the loop continues untill the all the entries of the candidate matrix are all zeros.
    # that is untill, there are no more reusable edges to select from.
    while np.sum(candidate_matrix) > 0:
        # compute sum of each row
        row_sums = np.sum(candidate_matrix, axis=1)

        # Get the rows in the candidate matrix whose sums are greater than 0.
        # Each non-zero row signifies that the qubit corresponding to that row
        # can be reused.
        rows_with_sum_greater_than_zero = np.where(row_sums > 0)[0]

        # We begin with the row (qubit) that has the highest potential for qubit reuse.
        # Since we haven't determined a row for selecting such row, we use the the argument
        # `initial_qubit` as the initial qubit selected. This initial choice is very critical for 
        # determining optimal qubit reuse sets for the circuit. The default choise is set to
        # qubit at row index 0 for convenience. Subsequent choice is done by selecting the first
        # none-zero in the candidate matrix after it has been updated.
        if counter == 0:
            t = rows_with_sum_greater_than_zero[initial_qubit] # we will loop over this row selection
        else:
            t = rows_with_sum_greater_than_zero[0]

        # After determining the initial (or subsequent) qubit, t we compute its maximum qubit reuse path.
        # This is done by assuming that all other non-zero qubits are not reusable, and then determining
        # the qubits that the qubit combination that results in the maximum qubit reuse for t. 
        # This is where we determine the maximum qubit reuse set (or path) for output (or terminal) qubit t.
        t_reuse_path, candidate_matrix = maximum_qubit_reuse_paths(candidate_matrix, t)

        counter += 1 # this is used to ensure that we apply initial_qubit only once.

        # if we found a reuse set for t, add it to the qubit reuse list.
        if len(t_reuse_path) > 1: 
            qubit_reuse_list.append(t_reuse_path)

    return qubit_reuse_list




def improved_gidnet_qubit_reuse_algorithm(circuit, maximize_circuit_depth=False):
    """
    Computes qubit reuse sets for a given quantum circuit. The biadjacency matrix, B is the matrix
    representation of the biadjacency graph of the quantum circuit. It is compute along side the candidate
    matrix, C given by: C = np.ones(n, n) - B.T, from the quantum circuit. The biadjacency graph represents
    the connections of and input qubit to the output qubits. The rows and columns of the biadjacency matrix
    are the input qubits (or root nodes) and output qubits (terminal nodes) respectively. An entry in the 
    matrix, (r, t) with value 1, indicates that there is a connection between input qubit r and output qubit
    t, while a zero entery indicates no connection. On the other other hand, the rows and columns of the 
    Candidate matrix, C are output qubits (terminal nodes) and input qubits (or root nodes). This is the 
    opposite of the biadjacency matrix because of the transpose in the equation relationship. An entry (t, r)
    in the candidate matrix with value 1, indicates that qubit r could be used on qubit t after qubit t has
    completed its operations. In order words, it indicates a potential reuse opportunity on qubit t.
    It is important to note that the first qubit selected (i.e `initial_qubit`) is very critical for optimal
    qubit reuse outcome.
    
    After computing the biadjacency and candidate matrices, the algorithm begins by selecting one of the qubits
    (rule not yet determined, but we can use the argument `initial_qubit` to specify the qubit we want to 
    begin with). It computes the maximum reuse for that qubit using `maximum_qubit_reuse_paths`. This is done
    by assuming all other qubits are not reusable and only reusing the qubit of interest, `initial_qubit` until 
    we can no longer reuse it. Note that the selected qubit is one of the rows of the candidate matrix whose
    sum of its entries is greater than zero. That is, there is at least one non-zero entry in that row.
    
    After the first selection, the candidate matrix is updated. Subsequent qubit selections are taken from 
    the first row of the candidate matrix whose sum is greater than zero. This is conditioned on the fact that 
    the rows are arranged in ascending order (a better and more optimal rule is possible, but available yet).
    The next qubit selected undergoes the same process of determining its maximum qubit reuse path (i.e. all the 
    qubits that can be used on it). This process repeats until the candidate matrix is entirely zeros, indicating 
    that no further qubit reuse opportunities exist.

    Parameters:
    - circ: The QuantumCircuit object from which the biadjacency and candidate matrices are derived.

    Returns:
    - qubit_reuse_list: A list of sublists, where each sublist contains indices of qubits that form a chain
                        of reuse operations, based on the iterative selection and updating process.
    """
    # compute the biadjacency matrix and candidate matrix from the circuit
    biadjacency_matrix, candidate_matrix = get_biadjacency_candidate_matrix(circuit)
    # print("candidate_matrix = ", candidate_matrix)
    # Check if all elements in candidate_matrix are ones
    if np.all(candidate_matrix == 0):
        # raise AssertionError("This circuit is irreducible")
        # print("This circuit is irreducible")
        num_rows = candidate_matrix.shape[0]
        return [[q] for q in range(num_rows)]

    num_rows = candidate_matrix.shape[0]
    current_qubit_reuse_list = [[]]*num_rows
    current_max_length = np.inf  # this is used to maximize circuit depth
    
    # print("initial_qubit_trials = ", initial_qubit_trials)
    
    for initial_qubit in range(num_rows):
        candidate_matrix_copy = copy.copy(candidate_matrix)
        # This is used to store the qubit reuse sets which is a list of sublists
        qubit_reuse_list = []

        counter = 0 # used to keep track of the initial_qubit selection, `initial_qubit`

        # the loop continues untill the all the entries of the candidate matrix are all zeros.
        # that is untill, there are no more reusable edges to select from.
        while np.sum(candidate_matrix_copy) > 0:
            # compute sum of each row
            row_sums = np.sum(candidate_matrix_copy, axis=1)

            # Get the rows in the candidate matrix whose sums are greater than 0.
            # Each non-zero row signifies that the qubit corresponding to that row
            # can be reused.
            rows_with_sum_greater_than_zero = np.where(row_sums > 0)[0]

            # We begin with the row (qubit) that has the highest potential for qubit reuse.
            # Since we haven't determined a row for selecting such row, we use the the argument
            # `initial_qubit` as the initial qubit selected. This initial choice is very critical for 
            # determining optimal qubit reuse sets for the circuit. The default choise is set to
            # qubit at row index 0 for convenience. Subsequent choice is done by selecting the first
            # none-zero in the candidate matrix after it has been updated.
            if counter == 0 and len(rows_with_sum_greater_than_zero)>initial_qubit:
                t = rows_with_sum_greater_than_zero[initial_qubit] # we will loop over this row selection
            else:
                t = rows_with_sum_greater_than_zero[0]

            # After determining the initial (or subsequent) qubit, t we compute its maximum qubit reuse path.
            # This is done by assuming that all other non-zero qubits are not reusable, and then determining
            # the qubits that the qubit combination that results in the maximum qubit reuse for t. 
            # This is where we determine the maximum qubit reuse set (or path) for output (or terminal) qubit t.
            t_reuse_path, candidate_matrix_copy = maximum_qubit_reuse_paths(candidate_matrix_copy, t)

            # if we found a reuse set for t, add it to the qubit reuse list.
            if len(t_reuse_path) > 1: 
                qubit_reuse_list.append(t_reuse_path)
                
            # for debugging
            # if counter == 0:
            #     print("t = ", t)
                
            counter += 1 # this is used to ensure that we apply initial_qubit only once.
        
        # for debugging
        # print("qubit_reuse_list = ", qubit_reuse_list)
        # print()
        
        qubit_reuse_list = merge_sublists(qubit_reuse_list)

        # Optionally, add any qubits not included in the reuse list as their individual sets.
        qubit_reuse_list = finalize_qubit_reuse_list(qubit_reuse_list, len(circuit.qubits))
        
        if maximize_circuit_depth:
            # Find the maximum length of the elements of the list of lists
            max_length = max(len(sublist) for sublist in qubit_reuse_list)
            
            if len(qubit_reuse_list) <= len(current_qubit_reuse_list):
                if max_length <= current_max_length:
                    current_qubit_reuse_list = qubit_reuse_list
                    current_max_length = max_length
        else:
            if len(qubit_reuse_list) < len(current_qubit_reuse_list):
                current_qubit_reuse_list = qubit_reuse_list
                
        # print("current_qubit_reuse_list = ", current_qubit_reuse_list)
    # print("current_qubit_reuse_list = ", current_qubit_reuse_list)
    # print("candidate_matrix = ", candidate_matrix)

    return current_qubit_reuse_list





def compute_qubit_reuse_sets(circuit, initial_qubit=0, improved_gidnet=False, maximize_circuit_depth=False):
    """
    Computes qubit reuse lists by iteratively selecting individual qubits for reuse based on the row
    with the highest sum in the candidate matrix. After each selection, the candidate matrix is updated,
    potentially identifying a new row with the largest sum for the next selection. This process repeats
    until the candidate matrix is entirely zeros, indicating that no further qubit reuse opportunities exist.

    Parameters:
    - circ: The QuantumCircuit object from which the biadjacency and candidate matrices are derived.

    Returns:
    - qubit_reuse_list: A list of sublists, where each sublist contains indices of qubits that form a chain
                        of reuse operations, based on the iterative selection and updating process.
    """

    if improved_gidnet:
        # compute the qubit reuse sets using improved gidnet_qubit_reuse_algorithm
        qubit_reuse_list = improved_gidnet_qubit_reuse_algorithm(circuit, maximize_circuit_depth)
    else:
        # compute the qubit reuse sets using gidnet_qubit_reuse_algorithm
        qubit_reuse_list = gidnet_qubit_reuse_algorithm(circuit, initial_qubit)

        # merge qubit reuse sets into reuse lists
        qubit_reuse_list = merge_sublists(qubit_reuse_list)

        # add any qubits not included in the reuse list as their individual sets.
        qubit_reuse_list = finalize_qubit_reuse_list(qubit_reuse_list, len(circuit.qubits))

    return qubit_reuse_list



##************ Qiskit Qubit Reuse ************************###
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit_qubit_reuse import qubit_reuse, qubit_reuse_greedy
def apply_qiskit_qubit_reuse(cirucit):
    qr = qubit_reuse.QubitReuseModified()
    cirucit_dag= circuit_to_dag(cirucit)
    qr_cirucit = dag_to_circuit(qr.run(cirucit_dag))
    return qr_cirucit



########## QUANTUM CVIRCUIT BENCHMARKS #####################################

def create_qiskit_and_qnet_QAOA_circuits(num_qubits, num_circuits, layer_num=1):
    r"""
    Generates a set of QAOA circuits using both Qiskit and QNET frameworks for a given number of qubits, 
    applied to random unweighted 3-regular graphs (U3R). These graphs are used as the underlying topology 
    in the Max-Cut QAOA problem.

    Parameters:
    num_qubits : int
        The number of qubits for each QAOA circuit, which is also the number of vertices in the U3R graph.
    num_circuits : int
        The total number of random circuits to generate. Each circuit is initialized using an incremental
        seed value to ensure reproducibility and diversity in graph generation.
    layer_num : int, optional
        The number of QAOA layers (or the p-value in QAOA terms), specifying how many times the problem 
        and mixing unitaries are applied. Default is 1.

    Returns:
    tuple
        A tuple containing two lists: the first list contains Qiskit quantum circuits and the second list 
        contains QNET quantum circuits, both representing the QAOA implementation for the Max-Cut problem 
        on randomly generated U3R graphs.
    """
    
    from math import pi
    import numpy as np
    import networkx as nx
    from qiskit import QuantumCircuit
    from Extensions.QuantumNetwork.qcompute_qnet.quantum.circuit import Circuit
    
    qnet_circuits = []
    qiskit_circuits = []

    for seed in range(num_circuits):
        # Generate a random unweighted 3-regular graph (U3R) using the seed
        u3r_graph = nx.random_regular_graph(3, num_qubits, seed)

        # Initialize a QNET circuit for QAOA
        qnet_circuit = Circuit()

        # Initialize a Qiskit quantum circuit with the same number of qubits
        qiskit_circuit = QuantumCircuit(num_qubits)

        # Construct the Max-Cut QAOA circuit based on the U3R graph
        # Prepare a uniform superposition over all qubits
        for i in range(num_qubits):
            qnet_circuit.h(i)
            qiskit_circuit.h(i)

        # Apply the problem and mixing unitaries iteratively for each layer
        for p in range(layer_num):
            # Problem unitary (corresponding to the graph structure)
            for edge in u3r_graph.edges():
                qnet_circuit.cx([edge[0], edge[1]])
                qnet_circuit.rz(edge[1], pi)
                qnet_circuit.cx([edge[0], edge[1]])

                qiskit_circuit.cx(edge[0], edge[1])
                qiskit_circuit.rz(pi, edge[1])
                qiskit_circuit.cx(edge[0], edge[1])

            # Mixing unitary
            for node in u3r_graph.nodes():
                qnet_circuit.rx(node, pi)
                qiskit_circuit.rx(pi, node)

        # Measure all qubits at the end of the circuit
        qnet_circuit.measure()
        qiskit_circuit.measure_all()

        # Store the constructed circuits
        qnet_circuits.append(qnet_circuit)
        qiskit_circuits.append(qiskit_circuit)
        
    return qiskit_circuits, qnet_circuits




def create_qiskit_and_qnet_GRCS_circuits(num_qubits, cycle_num = 13, directory_path="../data"):
    """
    Generates Google Random Circuit Sampling (GRCS) circuits for both Qiskit and QNET based on stored 
    instruction files.

    Parameters:
    - num_qubits (str): Number of qubits for the GRCS circuit as a string, used for file naming conventions.
    - cycle_num (int): Specifies the cycle number which influences the file to be loaded.
    - directory_path (str): The path to the directory where circuit instruction files are stored.

    Returns:
    - tuple: A tuple containing two elements; the first is a Qiskit circuit and the second is a QNET circuit.
    """

    import os
    from math import pi
    from Extensions.QuantumNetwork.qcompute_qnet.quantum.circuit import Circuit
    from qiskit import QuantumCircuit

    # for qubit_num in circuit_size:
    folder_path = os.path.join(directory_path, num_qubits)
    inst_name = "inst_" + num_qubits + "_" + str(cycle_num) + "_0.txt"
    inst_path = os.path.join(folder_path, inst_name)

    input_cir = []
    with open(inst_path, "r") as file:
        for line in file:
            input_cir.append(list(line.strip("\n").split(" ")))

    # Create a QNET quantum circuit
    qnet_circuit = Circuit()

    # # Construct the QNET quantum circuit based on the instruction set
    for gate in input_cir[1:]:
        if gate[1] == "h":
            qnet_circuit.h(int(gate[2]))
        elif gate[1] == "cz":
            qnet_circuit.cz([int(gate[2]), int(gate[3])])
        elif gate[1] == "t":
            qnet_circuit.t(int(gate[2]))
        elif gate[1] == "x_1_2":
            qnet_circuit.rx(int(gate[2]), pi / 2)
        elif gate[1] == "y_1_2":
            qnet_circuit.ry(int(gate[2]), pi / 2)
        else:
            raise NotImplementedError

    # Measure all qubits
    qnet_circuit.measure()


    # Create a QISKIT quantum circuit
    qiskit_circuit = QuantumCircuit(int(safe_eval(num_qubits)))

    ## Construct the Qiskit quantum circuit using the same instructions
    for gate in input_cir[1:]:
        if gate[1] == "h":
            qiskit_circuit.h(int(gate[2]))
        elif gate[1] == "cz":
            qiskit_circuit.cz(int(gate[2]), int(gate[3]))
        elif gate[1] == "t":
            qiskit_circuit.t(int(gate[2]))
        elif gate[1] == "x_1_2":
            qiskit_circuit.rx(pi / 2, int(gate[2]))
        elif gate[1] == "y_1_2":
            qiskit_circuit.ry(pi / 2, int(gate[2]))
        else:
            raise NotImplementedError

    # Measure all qubits
    qiskit_circuit.measure_all()
    
    return qiskit_circuit, qnet_circuit


import ast
import operator

def safe_eval(expr):
    """
    Safely evaluate a simple arithmetic expression using Abstract Syntax Trees (AST).

    This function provides a safe alternative to the built-in eval() function, which can execute arbitrary code.
    Instead, it parses a mathematical expression into an AST and only allows specific operations, thus preventing
    the execution of unsafe code.

    Parameters:
    expr (str): A string representing a simple arithmetic expression, such as "4*4" or "4x4".

    Returns:
    int or float: The result of the evaluated expression.

    Raises:
    ValueError: If the expression contains any operators or constructs that are not explicitly allowed.

    Example:
    >>> safe_eval("4x4")
    16
    >>> safe_eval("3*3")
    9

    Note:
    The function currently supports only multiplication, but can be extended to include additional operations
    by modifying the `allowed_operators` dictionary.
    """

    # Replace 'x' with '*' to standardize the multiplication symbol in the expression
    expr = expr.replace('x', '*')

    # Parse the expression into an AST
    tree = ast.parse(expr, mode='eval')

    # Define allowed operations using a dictionary mapping AST operator types to Python's operator functions
    allowed_operators = {ast.Mult: operator.mul}
    
    # Visitor class to evaluate the AST; it restricts operations to those specified in `allowed_operators`
    class Evaluator(ast.NodeVisitor):
        def visit_BinOp(self, node):
            """Visit binary operations and ensure they are allowed."""
            if type(node.op) in allowed_operators:
                return allowed_operators[type(node.op)](self.visit(node.left), self.visit(node.right))
            raise ValueError("Unsupported operator: {}".format(ast.dump(node.op)))

        def visit_Num(self, node):
            """Return the value of a number node."""
            return node.n

        def visit_Expr(self, node):
            """Evaluate and return the expression value."""
            return self.visit(node.value)

    # Instantiate the evaluator and evaluate the parsed AST
    return Evaluator().visit(tree.body)
    
    

	
	
	
	
############## modifications to GidNET ###################

##### begining of main function #####
def modified_maximum_qubit_reuse_paths(candidate_matrix, t):
    """
    This helper function explores all possible qubit reuse paths for a specific qubit (or row) 't'
    in the candidate matrix and identifies the path with most qubits or optimal qubits for reuse. 
    It updates the candidate matrix to reflect the selected reuse paths (or edges).
    
    First, it determines the potential input qubits, r that can be used on qubit t after it has 
    completed its operations. Qubits r, are the corresponding columns in row t whose entries are
    1. These qubits are stored in potential_r_for_t. The qubits that make up the maximum reuse path
    are selected from these sets of qubits and stored in t_reuse_path.
    
    To determine the qubits that are parts of the maximum reuse path for t, it uses `update_potential_r_for_t`
    function. Qubit t is combined with each element in `potential_r_for_t` one at a time into a list. For
    example if we chose qubit r_0 from `potential_r_for_t`, then we have a `t_reuse_path`, [t, r_0]. This list shows 
    that qubit r_0 can be used on qubit t after qubit t has completed its operation. Remember that every qubit,
    r in `potential_r_for_t` has the potential of being used on t after t has completed its operation.
    The goal is to choose the qubit r, that gives a new `potential_r_for_t` with more qubits than that of others.
    
    To determine the new `potential_r_for_t` after r_0 is chose, we find columns in row r_0 where the values are
    1 in both rows t and r_0 in the candidate matrix. This columns will form the new or updated `potential_r_for_t`
    that we can choose from. This is done for all the qubits, r in `potential_r_for_t`. The qubit, r with the highest
    number of qubits in the updated `potential_r_for_t`, is chosen since it indicates potential for more qubits
    that can be used on t after t and r_0 have completed their operations. The next r is chosen from the 
    updated `potential_r_for_t`.
    
    For all the qubits in the updated `potential_r_for_t`, we repeat the above and choose the one with the highest 
    number of qubits in its updated `potential_r_for_t`. Say for example, we subsequently chose r_1 from the updated 
    `potential_r_for_t`, the updated `t_reuse_path` becomes [t, r_0, r_1]. The new updated `potential_r_for_t` will
    be all the columns (input qubits) in row r_1 with 1 as values in both rows t, r_0 and r_1. This is same as finding 
    the input qubits that are common to the output qubits, t, r_0 and r_1 in the original candidate matrix. We repeat 
    this process untill the `potential_r_for_t` is empty.
    
    Note that it is possible that for all the qubits in `potential_r_for_t` we couldn't find a common input qubit with 
    the qubits in `t_reuse_path`. This means that choosing any of the qubits in `potential_r_for_t` and including it in
    the `t_reuse_path` ends the reuse path for t. In that case, we just choose any of the qubits in `potential_r_for_t`.
    The default for this algorithm is to choose the first qubit in `potential_r_for_t` and add it to `t_reuse_path`.
    
    After the maximum `t_reuse_path` is determined for t, we updated the candidate matrix by removing entries in the 
    candidate matrix that can result in cycle in the DAG of the resulting dynamic circuit if subsequently selected. Also,
    the update is meant to remove the possibility of simultaneously reusing a qubit by more than one qubit at the same time.
    It also prevents the possibility of two qubits being reused simultaneously by one qubit. 
    

    Parameters:
    - candidate_matrix: the candidate matrix of the quantum circuit
    - t: The qubit to be reused. It corresponds to row t in the candidate matrix.

    Returns:
    - t_reuse_path: A list of qubits forming an optimal reuse path for `t`.
    - candidate_matrix: The updated candidate matrix after the maximum t_reuse_path is determined.
    """

    # Implementation details for exploring qubit reuse paths would go here.
    # This would involve selecting qubits from 'potential_r_for_t', updating 'C_copy' and 'B_copy',
    # and determining the 't_reuse_path' that represents the chosen path of qubit reuse.
    
    
    t_reuse_path = [t] # we store the maximum qubit reuse path for t here
    
    # The maximum qubit reuse path for t, t_reuse_path  starts with t. Subsequent nodes (or qubits) are
    # selected from the input qubits, r in potential_r_for_t. These are the qubits that have the potential
    # of being used on qubit t after qubit t has completed its operations. 
    # Qubits in potential_r_for_t corresponds to columns (or input qubits) in row t whose values are 1.
    potential_r_for_t = np.where(candidate_matrix[t, :] == 1)[0] # these are the potential root nodes for t. It if from here we make selections
    potential_r_for_t = list(potential_r_for_t) # convert to list
   
    # loop continues until potential_r_for_t is empty. That is there is no more qubits to select from and the
    # reuse path for t has terminated.
    while potential_r_for_t:
        # compute the updated potential_r_for_t (i.e. the qubits left in potential_r_for_t) for each qubit, potential_r in 
        # potential_r_for_t assuming potential_r was chosen to be part of t_reuse_path.
        potential_r_dict = {} # we store the updated potential_r_for_t for each qubit in potential_r_for_t in this dictionary
        for potential_r in potential_r_for_t: # loop through the elements in potential_r_for_t
            potential_t_reuse_path = t_reuse_path + [potential_r] # this is the potential reuse path for t if this r was chosen
            # compute the qubits that would be left in potential_r_for_t if this potential_r where included in t_reuse_path
            potential_r_dict[potential_r] = update_potential_r_for_t(candidate_matrix, potential_t_reuse_path) # this is updated potential_r_for_t if
                                                                                                               # potential_r was chosen

        # If choosing any of the qubits in potential_r_for_t leaves no qubits in updated potential_r_for_t.
        # This means choosing any qubit in potential_r_for_t terminates t_reuse_path. 
        are_all_empty = all(len(value) == 0 for value in potential_r_dict.values())

        # If choosing any qubit in potential_r_for_t terminates t_reuse_path, any of the qubits can be selected.
        # However, the default of this algorithm is to choose the first qubit in potential_r_for_t
        if are_all_empty:
            if len(potential_r_for_t)>0: # check if potential_r_for_t is not empty
                t_reuse_path.append(potential_r_for_t[0]) # choose the first qubit in potential_r_for_t
                potential_r_for_t.remove(potential_r_for_t[0]) # update potential_r_for_t by removing the chosen qubit
        
            # if there we found qubits that can be used on t, updated the candidate matrix
            if len(t_reuse_path) > 1: 
                edge_pair_list = [(t_reuse_path[i], t_reuse_path[i + 1]) for 
                                              i in range(len(t_reuse_path) - 1)]

                for terminal, root in edge_pair_list:
                    # Update the candiate matrix to avoid cycles and double nodes in the circuit DAG
                    candidate_matrix = update_candidate_matrix_after_selection(candidate_matrix, terminal, root)
                
            return t_reuse_path, candidate_matrix
        
        # If choosing any qubit in potential_r_for_t does not terminates t_reuse_path, we proceed
        # by choosing the the potential_r that give a reuse path which leaves behind the highers number
        # of qubits in the updated potential_r_for_t. In the event that we have more than one potential_r
        # with the same number of qubits in their updated potential_r_for_t, we proceed with another test.
        else:
            # find the potential node (qubit), potential_r with the maximum number of qubits left in updated 
            # potential_r_for_t, if it was selected.
            max_nodes = max(len(lst) for lst in potential_r_dict.values()) # this is the maximum number of qubits in updated potential_r_for_t
            potential_r_with_max_nodes = [key for key, lst in potential_r_dict.items() if len(lst) == max_nodes] # get the qubit with max potential_r_for_t

            # If there are more than one qubits in potential_t_with_max_nodes, we have to determine which one to choose.
            # This is done by computing the intersect of the updated potential_r_for_t for each qubits in potential_r_with_max_nodes
            # with the updated potential_r_for_t for the other qubits in potential_r_with_max_nodes. The qubit, with the largest 
            # intersect with the other qubits is selected. In order words, we want to select the qubit whose updated potential_r_for_t
            # has qubits that are more common across the updated potential_r_for_t of other qubits in potential_r_with_max_nodes.
            if len(potential_r_with_max_nodes) > 1:
                score = -np.inf # score to keep track of the highest intersection
                chosen_r = None # the potential_r with the highest intersection with the other r's in potential_r_with_max_nodes
                select_dict = {}
                for r in potential_r_with_max_nodes: # loop through the potential_r in potential_r_with_max_nodes
                    potential_r_for_t_intersections = [len(set(potential_r_dict[r]).intersection(potential_r_dict[i])) for i in 
                                    potential_r_with_max_nodes if i != r]  # find the intersection of the updated potential_r_for_t
                                                                           # for qubit r with the updated potential_r_for_t for each 
                                                                           # qubit in potential_r_with_max_nodes

                    total_interset = sum(potential_r_for_t_intersections)  # How many qubits in potential_r_with_max_nodes does it 
                                                                           # share common qubits with
                        
                    select_dict[r] = total_interset
                
                # we want to choose the qubit r in potential_r_with_max_nodes with the highest intersect
                max_score = max([s for r, s in select_dict.items()])
                select_list = [r for r, s in select_dict.items() if s==max_score]
                
                chosen_r = np.random.choice(select_list)  # randomly select one if they are more than one
                potential_r_for_t = potential_r_dict[chosen_r]

                # If we found a suitable r, add it to t_reuse_path
                if chosen_r is not None:
                    t_reuse_path.append(chosen_r) # update the qubit reuse path, t_reuse_path
                    
                    # If after add the suitable or chosen r to t_reuse_path, we are left with only one qubit in the updated
                    # potential_r_for_t, just add that qubit to t_reuse_path. It shows that that qubit terminates the qubit
                    # reuse path, t_reuse_path.
                    if len(potential_r_for_t) == 1: # this accounts for the last element in potential_r_for_t
                        t_reuse_path.append(potential_r_for_t[0]) # update the qubit reuse path, t_reuse_path
                        potential_r_for_t = [] # update potential_r_for_t to an empty list since no more qubit to select from
            
            # If there is only one qubits in potential_t_with_max_nodes, we have just choose that qubit.
            else:
                chosen_r = potential_r_with_max_nodes[0] # choose the only qubit in potential_t_with_max_nodes
                t_reuse_path.append(chosen_r) # update the qubit reuse path, t_reuse_path
                potential_r_for_t = potential_r_dict[chosen_r] # get the updated potential_r_for_t
                
                # If after add the suitable or chosen r to t_reuse_path, we are left with only one qubit in the updated
                # potential_r_for_t, just add that qubit to t_reuse_path. It shows that that qubit terminates the qubit
                # reuse path, t_reuse_path.
                if len(potential_r_for_t) == 1: # this accounts for the last element in potential_r_for_t
                        t_reuse_path.append(potential_r_for_t[0]) # update the qubit reuse path, t_reuse_path
                        potential_r_for_t = []  # update potential_r_for_t to an empty list since no more qubit to select from


    # if we found qubit for the reuse path of t, that is, the length of t_reuse_path is greater than 1,
    # We have to update the candidate matrix to reflect this our choices and eliminate cycles in the DAG.
    if len(t_reuse_path) > 1:
        edge_pair_list = [(t_reuse_path[i], t_reuse_path[i + 1]) for 
                                      i in range(len(t_reuse_path) - 1)]

        for terminal, root in edge_pair_list:
            candidate_matrix = update_candidate_matrix_after_selection(candidate_matrix, terminal, root) # update made to the candidate matrix

    return t_reuse_path, candidate_matrix
##### End of main ##########################


import math
def modified_maximum_qubit_reuse_paths(candidate_matrix, t):
    """
    This helper function explores all possible qubit reuse paths for a specific qubit (or row) 't'
    in the candidate matrix and identifies the path with most qubits or optimal qubits for reuse. 
    It updates the candidate matrix to reflect the selected reuse paths (or edges).
    
    First, it determines the potential input qubits, r that can be used on qubit t after it has 
    completed its operations. Qubits r, are the corresponding columns in row t whose entries are
    1. These qubits are stored in potential_r_for_t. The qubits that make up the maximum reuse path
    are selected from these sets of qubits and stored in t_reuse_path.
    
    To determine the qubits that are parts of the maximum reuse path for t, it uses `update_potential_r_for_t`
    function. Qubit t is combined with each element in `potential_r_for_t` one at a time into a list. For
    example if we chose qubit r_0 from `potential_r_for_t`, then we have a `t_reuse_path`, [t, r_0]. This list shows 
    that qubit r_0 can be used on qubit t after qubit t has completed its operation. Remember that every qubit,
    r in `potential_r_for_t` has the potential of being used on t after t has completed its operation.
    The goal is to choose the qubit r, that gives a new `potential_r_for_t` with more qubits than that of others.
    
    To determine the new `potential_r_for_t` after r_0 is chose, we find columns in row r_0 where the values are
    1 in both rows t and r_0 in the candidate matrix. This columns will form the new or updated `potential_r_for_t`
    that we can choose from. This is done for all the qubits, r in `potential_r_for_t`. The qubit, r with the highest
    number of qubits in the updated `potential_r_for_t`, is chosen since it indicates potential for more qubits
    that can be used on t after t and r_0 have completed their operations. The next r is chosen from the 
    updated `potential_r_for_t`.
    
    For all the qubits in the updated `potential_r_for_t`, we repeat the above and choose the one with the highest 
    number of qubits in its updated `potential_r_for_t`. Say for example, we subsequently chose r_1 from the updated 
    `potential_r_for_t`, the updated `t_reuse_path` becomes [t, r_0, r_1]. The new updated `potential_r_for_t` will
    be all the columns (input qubits) in row r_1 with 1 as values in both rows t, r_0 and r_1. This is same as finding 
    the input qubits that are common to the output qubits, t, r_0 and r_1 in the original candidate matrix. We repeat 
    this process untill the `potential_r_for_t` is empty.
    
    Note that it is possible that for all the qubits in `potential_r_for_t` we couldn't find a common input qubit with 
    the qubits in `t_reuse_path`. This means that choosing any of the qubits in `potential_r_for_t` and including it in
    the `t_reuse_path` ends the reuse path for t. In that case, we just choose any of the qubits in `potential_r_for_t`.
    The default for this algorithm is to choose the first qubit in `potential_r_for_t` and add it to `t_reuse_path`.
    
    After the maximum `t_reuse_path` is determined for t, we updated the candidate matrix by removing entries in the 
    candidate matrix that can result in cycle in the DAG of the resulting dynamic circuit if subsequently selected. Also,
    the update is meant to remove the possibility of simultaneously reusing a qubit by more than one qubit at the same time.
    It also prevents the possibility of two qubits being reused simultaneously by one qubit. 
    

    Parameters:
    - candidate_matrix: the candidate matrix of the quantum circuit
    - t: The qubit to be reused. It corresponds to row t in the candidate matrix.

    Returns:
    - t_reuse_path: A list of qubits forming an optimal reuse path for `t`.
    - candidate_matrix: The updated candidate matrix after the maximum t_reuse_path is determined.
    """

    # Implementation details for exploring qubit reuse paths would go here.
    # This would involve selecting qubits from 'potential_r_for_t', updating 'C_copy' and 'B_copy',
    # and determining the 't_reuse_path' that represents the chosen path of qubit reuse.
    
    
    t_reuse_path = [t] # we store the maximum qubit reuse path for t here
    
    # The maximum qubit reuse path for t, t_reuse_path  starts with t. Subsequent nodes (or qubits) are
    # selected from the input qubits, r in potential_r_for_t. These are the qubits that have the potential
    # of being used on qubit t after qubit t has completed its operations. 
    # Qubits in potential_r_for_t corresponds to columns (or input qubits) in row t whose values are 1.
    potential_r_for_t = np.where(candidate_matrix[t, :] == 1)[0] # these are the potential root nodes for t. It if from here we make selections
    potential_r_for_t = list(potential_r_for_t) # convert to list
    
    # main modification *******#
    alpha = 0.2
    n = math.ceil(len(potential_r_for_t) * alpha)
    # potential_r_for_t = potential_r_for_t[0:math.ceil(n/2)+1]
    # potential_r_for_t = potential_r_for_t[0]
    # print("I am here")
    # potential_r_for_t = [np.random.choice(potential_r_for_t)]
    potential_r_for_t = list(np.random.choice(potential_r_for_t, n, replace=False))
    ###################################################################
   
    # loop continues until potential_r_for_t is empty. That is there is no more qubits to select from and the
    # reuse path for t has terminated.
    while potential_r_for_t:
        # compute the updated potential_r_for_t (i.e. the qubits left in potential_r_for_t) for each qubit, potential_r in 
        # potential_r_for_t assuming potential_r was chosen to be part of t_reuse_path.
        potential_r_dict = {} # we store the updated potential_r_for_t for each qubit in potential_r_for_t in this dictionary
        for potential_r in potential_r_for_t: # loop through the elements in potential_r_for_t
            potential_t_reuse_path = t_reuse_path + [potential_r] # this is the potential reuse path for t if this r was chosen
            # compute the qubits that would be left in potential_r_for_t if this potential_r where included in t_reuse_path
            potential_r_dict[potential_r] = update_potential_r_for_t(candidate_matrix, potential_t_reuse_path) # this is updated potential_r_for_t if
                                                                                                               # potential_r was chosen

        # If choosing any of the qubits in potential_r_for_t leaves no qubits in updated potential_r_for_t.
        # This means choosing any qubit in potential_r_for_t terminates t_reuse_path. 
        are_all_empty = all(len(value) == 0 for value in potential_r_dict.values())

        # If choosing any qubit in potential_r_for_t terminates t_reuse_path, any of the qubits can be selected.
        # However, the default of this algorithm is to choose the first qubit in potential_r_for_t
        if are_all_empty:
            if len(potential_r_for_t)>0: # check if potential_r_for_t is not empty
                # t_reuse_path.append(potential_r_for_t[0]) # choose the first qubit in potential_r_for_t
                t_reuse_path.append(np.random.choice(potential_r_for_t)) # choose a random qubit in potential_r_for_t
                potential_r_for_t.remove(potential_r_for_t[0]) # update potential_r_for_t by removing the chosen qubit
        
            # if there we found qubits that can be used on t, updated the candidate matrix
            if len(t_reuse_path) > 1: 
                edge_pair_list = [(t_reuse_path[i], t_reuse_path[i + 1]) for 
                                              i in range(len(t_reuse_path) - 1)]

                for terminal, root in edge_pair_list:
                    # Update the candiate matrix to avoid cycles and double nodes in the circuit DAG
                    candidate_matrix = update_candidate_matrix_after_selection(candidate_matrix, terminal, root)
                
            return t_reuse_path, candidate_matrix
        
        # If choosing any qubit in potential_r_for_t does not terminates t_reuse_path, we proceed
        # by choosing the the potential_r that give a reuse path which leaves behind the highers number
        # of qubits in the updated potential_r_for_t. In the event that we have more than one potential_r
        # with the same number of qubits in their updated potential_r_for_t, we proceed with another test.
        else:
            # find the potential node (qubit), potential_r with the maximum number of qubits left in updated 
            # potential_r_for_t, if it was selected.
            max_nodes = max(len(lst) for lst in potential_r_dict.values()) # this is the maximum number of qubits in updated potential_r_for_t
            potential_r_with_max_nodes = [key for key, lst in potential_r_dict.items() if len(lst) == max_nodes] # get the qubit with max potential_r_for_t

            # If there are more than one qubits in potential_t_with_max_nodes, we have to determine which one to choose.
            # This is done by computing the intersect of the updated potential_r_for_t for each qubits in potential_r_with_max_nodes
            # with the updated potential_r_for_t for the other qubits in potential_r_with_max_nodes. The qubit, with the largest 
            # intersect with the other qubits is selected. In order words, we want to select the qubit whose updated potential_r_for_t
            # has qubits that are more common across the updated potential_r_for_t of other qubits in potential_r_with_max_nodes.
            if len(potential_r_with_max_nodes) > 1:
                score = -np.inf # score to keep track of the highest intersection
                chosen_r = None # the potential_r with the highest intersection with the other r's in potential_r_with_max_nodes
                select_dict = {}
                for r in potential_r_with_max_nodes: # loop through the potential_r in potential_r_with_max_nodes
                    potential_r_for_t_intersections = [len(set(potential_r_dict[r]).intersection(potential_r_dict[i])) for i in 
                                    potential_r_with_max_nodes if i != r]  # find the intersection of the updated potential_r_for_t
                                                                           # for qubit r with the updated potential_r_for_t for each 
                                                                           # qubit in potential_r_with_max_nodes

                    total_interset = sum(potential_r_for_t_intersections)  # How many qubits in potential_r_with_max_nodes does it 
                                                                           # share common qubits with
                        
                    select_dict[r] = total_interset
                
                # we want to choose the qubit r in potential_r_with_max_nodes with the highest intersect
                max_score = max([s for r, s in select_dict.items()])
                select_list = [r for r, s in select_dict.items() if s==max_score]
                
                chosen_r = np.random.choice(select_list)  # randomly select one if they are more than one
                potential_r_for_t = potential_r_dict[chosen_r]

                # If we found a suitable r, add it to t_reuse_path
                if chosen_r is not None:
                    t_reuse_path.append(chosen_r) # update the qubit reuse path, t_reuse_path
                    
                    # If after add the suitable or chosen r to t_reuse_path, we are left with only one qubit in the updated
                    # potential_r_for_t, just add that qubit to t_reuse_path. It shows that that qubit terminates the qubit
                    # reuse path, t_reuse_path.
                    if len(potential_r_for_t) == 1: # this accounts for the last element in potential_r_for_t
                        t_reuse_path.append(potential_r_for_t[0]) # update the qubit reuse path, t_reuse_path
                        potential_r_for_t = [] # update potential_r_for_t to an empty list since no more qubit to select from
            
            # If there is only one qubits in potential_t_with_max_nodes, we have just choose that qubit.
            else:
                chosen_r = potential_r_with_max_nodes[0] # choose the only qubit in potential_t_with_max_nodes
                t_reuse_path.append(chosen_r) # update the qubit reuse path, t_reuse_path
                potential_r_for_t = potential_r_dict[chosen_r] # get the updated potential_r_for_t
                
                # If after add the suitable or chosen r to t_reuse_path, we are left with only one qubit in the updated
                # potential_r_for_t, just add that qubit to t_reuse_path. It shows that that qubit terminates the qubit
                # reuse path, t_reuse_path.
                if len(potential_r_for_t) == 1: # this accounts for the last element in potential_r_for_t
                        t_reuse_path.append(potential_r_for_t[0]) # update the qubit reuse path, t_reuse_path
                        potential_r_for_t = []  # update potential_r_for_t to an empty list since no more qubit to select from


    # if we found qubit for the reuse path of t, that is, the length of t_reuse_path is greater than 1,
    # We have to update the candidate matrix to reflect this our choices and eliminate cycles in the DAG.
    if len(t_reuse_path) > 1:
        edge_pair_list = [(t_reuse_path[i], t_reuse_path[i + 1]) for 
                                      i in range(len(t_reuse_path) - 1)]

        for terminal, root in edge_pair_list:
            candidate_matrix = update_candidate_matrix_after_selection(candidate_matrix, terminal, root) # update made to the candidate matrix

    return t_reuse_path, candidate_matrix




#### This is for random selection
# def modified_maximum_qubit_reuse_paths(candidate_matrix, t):
#     """
#     This helper function explores all possible qubit reuse paths for a specific qubit (or row) 't'
#     in the candidate matrix and identifies the path with most qubits or optimal qubits for reuse. 
#     It updates the candidate matrix to reflect the selected reuse paths (or edges).
    
#     First, it determines the potential input qubits, r that can be used on qubit t after it has 
#     completed its operations. Qubits r, are the corresponding columns in row t whose entries are
#     1. These qubits are stored in potential_r_for_t. The qubits that make up the maximum reuse path
#     are selected from these sets of qubits and stored in t_reuse_path.
    
#     To determine the qubits that are parts of the maximum reuse path for t, it uses `update_potential_r_for_t`
#     function. Qubit t is combined with each element in `potential_r_for_t` one at a time into a list. For
#     example if we chose qubit r_0 from `potential_r_for_t`, then we have a `t_reuse_path`, [t, r_0]. This list shows 
#     that qubit r_0 can be used on qubit t after qubit t has completed its operation. Remember that every qubit,
#     r in `potential_r_for_t` has the potential of being used on t after t has completed its operation.
#     The goal is to choose the qubit r, that gives a new `potential_r_for_t` with more qubits than that of others.
    
#     To determine the new `potential_r_for_t` after r_0 is chose, we find columns in row r_0 where the values are
#     1 in both rows t and r_0 in the candidate matrix. This columns will form the new or updated `potential_r_for_t`
#     that we can choose from. This is done for all the qubits, r in `potential_r_for_t`. The qubit, r with the highest
#     number of qubits in the updated `potential_r_for_t`, is chosen since it indicates potential for more qubits
#     that can be used on t after t and r_0 have completed their operations. The next r is chosen from the 
#     updated `potential_r_for_t`.
    
#     For all the qubits in the updated `potential_r_for_t`, we repeat the above and choose the one with the highest 
#     number of qubits in its updated `potential_r_for_t`. Say for example, we subsequently chose r_1 from the updated 
#     `potential_r_for_t`, the updated `t_reuse_path` becomes [t, r_0, r_1]. The new updated `potential_r_for_t` will
#     be all the columns (input qubits) in row r_1 with 1 as values in both rows t, r_0 and r_1. This is same as finding 
#     the input qubits that are common to the output qubits, t, r_0 and r_1 in the original candidate matrix. We repeat 
#     this process untill the `potential_r_for_t` is empty.
    
#     Note that it is possible that for all the qubits in `potential_r_for_t` we couldn't find a common input qubit with 
#     the qubits in `t_reuse_path`. This means that choosing any of the qubits in `potential_r_for_t` and including it in
#     the `t_reuse_path` ends the reuse path for t. In that case, we just choose any of the qubits in `potential_r_for_t`.
#     The default for this algorithm is to choose the first qubit in `potential_r_for_t` and add it to `t_reuse_path`.
    
#     After the maximum `t_reuse_path` is determined for t, we updated the candidate matrix by removing entries in the 
#     candidate matrix that can result in cycle in the DAG of the resulting dynamic circuit if subsequently selected. Also,
#     the update is meant to remove the possibility of simultaneously reusing a qubit by more than one qubit at the same time.
#     It also prevents the possibility of two qubits being reused simultaneously by one qubit. 
    

#     Parameters:
#     - candidate_matrix: the candidate matrix of the quantum circuit
#     - t: The qubit to be reused. It corresponds to row t in the candidate matrix.

#     Returns:
#     - t_reuse_path: A list of qubits forming an optimal reuse path for `t`.
#     - candidate_matrix: The updated candidate matrix after the maximum t_reuse_path is determined.
#     """

#     # Implementation details for exploring qubit reuse paths would go here.
#     # This would involve selecting qubits from 'potential_r_for_t', updating 'C_copy' and 'B_copy',
#     # and determining the 't_reuse_path' that represents the chosen path of qubit reuse.
    
    
#     t_reuse_path = [t] # we store the maximum qubit reuse path for t here
    
#     # The maximum qubit reuse path for t, t_reuse_path  starts with t. Subsequent nodes (or qubits) are
#     # selected from the input qubits, r in potential_r_for_t. These are the qubits that have the potential
#     # of being used on qubit t after qubit t has completed its operations. 
#     # Qubits in potential_r_for_t corresponds to columns (or input qubits) in row t whose values are 1.
#     potential_r_for_t = np.where(candidate_matrix[t, :] == 1)[0] # these are the potential root nodes for t. It if from here we make selections
#     potential_r_for_t = list(potential_r_for_t) # convert to list
   
#     # loop continues until potential_r_for_t is empty. That is there is no more qubits to select from and the
#     # reuse path for t has terminated.
#     while potential_r_for_t:
#         # compute the updated potential_r_for_t (i.e. the qubits left in potential_r_for_t) for each qubit, potential_r in 
#         # potential_r_for_t assuming potential_r was chosen to be part of t_reuse_path.
#         potential_r_dict = {} # we store the updated potential_r_for_t for each qubit in potential_r_for_t in this dictionary
#         for potential_r in potential_r_for_t: # loop through the elements in potential_r_for_t
#             potential_t_reuse_path = t_reuse_path + [potential_r] # this is the potential reuse path for t if this r was chosen
#             # compute the qubits that would be left in potential_r_for_t if this potential_r where included in t_reuse_path
#             potential_r_dict[potential_r] = update_potential_r_for_t(candidate_matrix, potential_t_reuse_path) # this is updated potential_r_for_t if
#                                                                                                                # potential_r was chosen

#         # If choosing any of the qubits in potential_r_for_t leaves no qubits in updated potential_r_for_t.
#         # This means choosing any qubit in potential_r_for_t terminates t_reuse_path. 
#         are_all_empty = all(len(value) == 0 for value in potential_r_dict.values())

#         # If choosing any qubit in potential_r_for_t terminates t_reuse_path, any of the qubits can be selected.
#         # However, the default of this algorithm is to choose the first qubit in potential_r_for_t
#         if are_all_empty:
#             if len(potential_r_for_t)>0: # check if potential_r_for_t is not empty
#                 t_reuse_path.append(potential_r_for_t[0]) # choose the first qubit in potential_r_for_t
#                 potential_r_for_t.remove(potential_r_for_t[0]) # update potential_r_for_t by removing the chosen qubit
        
#             # if there we found qubits that can be used on t, updated the candidate matrix
#             if len(t_reuse_path) > 1: 
#                 edge_pair_list = [(t_reuse_path[i], t_reuse_path[i + 1]) for 
#                                               i in range(len(t_reuse_path) - 1)]

#                 for terminal, root in edge_pair_list:
#                     # Update the candiate matrix to avoid cycles and double nodes in the circuit DAG
#                     candidate_matrix = update_candidate_matrix_after_selection(candidate_matrix, terminal, root)
                
#             return t_reuse_path, candidate_matrix
        
#         # If choosing any qubit in potential_r_for_t does not terminates t_reuse_path, we proceed
#         # by choosing the the potential_r that give a reuse path which leaves behind the highers number
#         # of qubits in the updated potential_r_for_t. In the event that we have more than one potential_r
#         # with the same number of qubits in their updated potential_r_for_t, we proceed with another test.
#         else:
#             # find the potential node (qubit), potential_r with the maximum number of qubits left in updated 
#             # potential_r_for_t, if it was selected.
#             max_nodes = max(len(lst) for lst in potential_r_dict.values()) # this is the maximum number of qubits in updated potential_r_for_t
#             potential_r_with_max_nodes = [key for key, lst in potential_r_dict.items() if len(lst) == max_nodes] # get the qubit with max potential_r_for_t

#             # If there are more than one qubits in potential_t_with_max_nodes, we have to determine which one to choose.
#             # This is done by computing the intersect of the updated potential_r_for_t for each qubits in potential_r_with_max_nodes
#             # with the updated potential_r_for_t for the other qubits in potential_r_with_max_nodes. The qubit, with the largest 
#             # intersect with the other qubits is selected. In order words, we want to select the qubit whose updated potential_r_for_t
#             # has qubits that are more common across the updated potential_r_for_t of other qubits in potential_r_with_max_nodes.
#             if len(potential_r_with_max_nodes) > 1:                
#                 chosen_r = np.random.choice(potential_r_with_max_nodes)  # randomly select one if they are more than one
#                 potential_r_for_t = potential_r_dict[chosen_r]

#                 # If we found a suitable r, add it to t_reuse_path
#                 if chosen_r is not None:
#                     t_reuse_path.append(chosen_r) # update the qubit reuse path, t_reuse_path
                    
#                     # If after add the suitable or chosen r to t_reuse_path, we are left with only one qubit in the updated
#                     # potential_r_for_t, just add that qubit to t_reuse_path. It shows that that qubit terminates the qubit
#                     # reuse path, t_reuse_path.
#                     if len(potential_r_for_t) == 1: # this accounts for the last element in potential_r_for_t
#                         t_reuse_path.append(potential_r_for_t[0]) # update the qubit reuse path, t_reuse_path
#                         potential_r_for_t = [] # update potential_r_for_t to an empty list since no more qubit to select from
            
#             # If there is only one qubits in potential_t_with_max_nodes, we have just choose that qubit.
#             else:
#                 chosen_r = potential_r_with_max_nodes[0] # choose the only qubit in potential_t_with_max_nodes
#                 t_reuse_path.append(chosen_r) # update the qubit reuse path, t_reuse_path
#                 potential_r_for_t = potential_r_dict[chosen_r] # get the updated potential_r_for_t
                
#                 # If after add the suitable or chosen r to t_reuse_path, we are left with only one qubit in the updated
#                 # potential_r_for_t, just add that qubit to t_reuse_path. It shows that that qubit terminates the qubit
#                 # reuse path, t_reuse_path.
#                 if len(potential_r_for_t) == 1: # this accounts for the last element in potential_r_for_t
#                         t_reuse_path.append(potential_r_for_t[0]) # update the qubit reuse path, t_reuse_path
#                         potential_r_for_t = []  # update potential_r_for_t to an empty list since no more qubit to select from


#     # if we found qubit for the reuse path of t, that is, the length of t_reuse_path is greater than 1,
#     # We have to update the candidate matrix to reflect this our choices and eliminate cycles in the DAG.
#     if len(t_reuse_path) > 1:
#         edge_pair_list = [(t_reuse_path[i], t_reuse_path[i + 1]) for 
#                                       i in range(len(t_reuse_path) - 1)]

#         for terminal, root in edge_pair_list:
#             candidate_matrix = update_candidate_matrix_after_selection(candidate_matrix, terminal, root) # update made to the candidate matrix

#     return t_reuse_path, candidate_matrix







import random
from collections import Counter
import copy
    
def compute_input_and_output_causal_cones(candidate_matrix):
    """
    computes the causal cones of the input qubits as well as the
    output qubits using the candidate matrix of the circuit.
    """
    input_causal_cones = {}
    output_causal_cones = {}
    n = candidate_matrix.shape[0]
    for q in range(n):
        input_causal_cones[q] = list(np.where(candidate_matrix[q, :] == 1)[0])
        output_causal_cones[q] = [k for k in range(n) if k 
                                  not in input_causal_cones[q]] # this is the main causal cone
        
    return input_causal_cones, output_causal_cones
    


def qubit_occurrences(forward_cone):
    """
    Finds the most frequent entries in a dictionary where values are lists, and returns the entries
    with their maximum occurrence count.
    
    Parameters:
    - forward_cone (dict): A dictionary where values are lists containing numbers.
    
    Returns:
    - list of tuples: Each tuple contains a unique entry and its maximum occurrence count.
    """
    # Combine all lists into one large list
    combined_list = []
    for values in forward_cone.values():
        combined_list.extend(values)
    
    # Count the occurrences of each number in the combined list
    count = Counter(combined_list)
    
    # Find the maximum occurrence value
    if count:
        max_occurrence = max(count.values())
        # Extract the entries that have the maximum occurrence
        max_items = [(item, count) for item, count in count.items() if count == max_occurrence]
        # return max_items
        return count
    else:
        return []
    


def selectStartingQubit(candidate_matrix, rows_with_sum_greater_than_zero):
    """
    Finds the most frequent entries in a dictionary where values are lists, and returns the entries
    with their maximum occurrence count.
    
    Parameters:
    - forward_cone (dict): A dictionary where values are lists containing numbers.
    
    Returns:
    - list of tuples: Each tuple contains a unique entry and its maximum occurrence count.
    """
    
    input_causal_cones, output_causal_cones = compute_input_and_output_causal_cones(candidate_matrix)
    n = candidate_matrix.shape[0]
    # influence_dict = qubit_scores(n, output_causal_cones)
    influence_dict = qubit_occurrences(output_causal_cones)
    
    # print("rows_with_sum_greater_than_zero = ", rows_with_sum_greater_than_zero)
    # print("influence_dict = ", influence_dict)
    
    # print("before_count = ", count)
    
    # We are only interested in the non-zero rows
    count = {q: s for q, s in influence_dict.items() if q in rows_with_sum_greater_than_zero}
    
    # Find the maximum occurrence value
    if count:
        max_occurrence = max(count.values())
        max_items = [item for item, count in count.items() if count == max_occurrence]
        
        select_list = [output_causal_cones[q] for q in max_items]
        flattend_select_list = list(set([q for sublist in select_list for q in sublist]))
        
        return np.random.choice(flattend_select_list)
    else:
        return []
    



def modified_gidnet_qubit_reuse_algorithm(circuit, shots):
    """
    Computes qubit reuse sets for a given quantum circuit. The biadjacency matrix, B is the matrix
    representation of the biadjacency graph of the quantum circuit. It is compute along side the candidate
    matrix, C given by: C = np.ones(n, n) - B.T, from the quantum circuit. The biadjacency graph represents
    the connections of and input qubit to the output qubits. The rows and columns of the biadjacency matrix
    are the input qubits (or root nodes) and output qubits (terminal nodes) respectively. An entry in the 
    matrix, (r, t) with value 1, indicates that there is a connection between input qubit r and output qubit
    t, while a zero entery indicates no connection. On the other other hand, the rows and columns of the 
    Candidate matrix, C are output qubits (terminal nodes) and input qubits (or root nodes). This is the 
    opposite of the biadjacency matrix because of the transpose in the equation relationship. An entry (t, r)
    in the candidate matrix with value 1, indicates that qubit r could be used on qubit t after qubit t has
    completed its operations. In order words, it indicates a potential reuse opportunity on qubit t.
    It is important to note that the first qubit selected (i.e `initial_qubit`) is very critical for optimal
    qubit reuse outcome.
    
    After computing the biadjacency and candidate matrices, the algorithm begins by selecting one of the qubits
    (rule not yet determined, but we can use the argument `initial_qubit` to specify the qubit we want to 
    begin with). It computes the maximum reuse for that qubit using `maximum_qubit_reuse_paths`. This is done
    by assuming all other qubits are not reusable and only reusing the qubit of interest, `initial_qubit` until 
    we can no longer reuse it. Note that the selected qubit is one of the rows of the candidate matrix whose
    sum of its entries is greater than zero. That is, there is at least one non-zero entry in that row.
    
    After the first selection, the candidate matrix is updated. Subsequent qubit selections are taken from 
    the first row of the candidate matrix whose sum is greater than zero. This is conditioned on the fact that 
    the rows are arranged in ascending order (a better and more optimal rule is possible, but available yet).
    The next qubit selected undergoes the same process of determining its maximum qubit reuse path (i.e. all the 
    qubits that can be used on it). This process repeats until the candidate matrix is entirely zeros, indicating 
    that no further qubit reuse opportunities exist.

    Parameters:
    - circ: The QuantumCircuit object from which the biadjacency and candidate matrices are derived.

    Returns:
    - qubit_reuse_list: A list of sublists, where each sublist contains indices of qubits that form a chain
                        of reuse operations, based on the iterative selection and updating process.
    """
    # compute the biadjacency matrix and candidate matrix from the circuit
    main_biadjacency_matrix, main_candidate_matrix = get_biadjacency_candidate_matrix(circuit)
    # forward_cone, causal_cone = compute_input_and_output_causal_cones(main_candidate_matrix)
    
    n = circuit.num_qubits

    if np.all(main_candidate_matrix == 0):
        # raise AssertionError("This circuit is irreducible")
        # print("This circuit is irreducible")
        return [[q] for q in range(n)]

    logn = int(np.log2(n))
    # print("logn*shots", logn*shots)
    # this is where the final qubit reuse will be stored
    final_qubit_reuse_list = [[]]*n

    # choose the num_runs not to exceed n
    # if logn*shots > n:
    #     num_runs = n
    # else:
    #     num_runs = logn*shots

    
    for _ in range(logn*shots):
        candidate_matrix = copy.deepcopy(main_candidate_matrix)
        # This is used to store the qubit reuse sets which is a list of sublists
        qubit_reuse_list = []

        # print("np.sum(candidate_matrix) = ", np.sum(candidate_matrix))
        # the loop continues untill the all the entries of the candidate matrix are all zeros.
        # that is untill, there are no more reusable edges to select from.
        while np.sum(candidate_matrix) > 0:
            # compute sum of each row
            row_sums = np.sum(candidate_matrix, axis=1)

            # Get the rows in the candidate matrix whose sums are greater than 0.
            # Each non-zero row signifies that the qubit corresponding to that row
            # can be reused.
            rows_with_sum_greater_than_zero = np.where(row_sums > 0)[0]

            # We begin with the row (qubit) that has the highest potential for qubit reuse.
            # Since we haven't determined a row for selecting such row, we use the the argument
            # `initial_qubit` as the initial qubit selected. This initial choice is very critical for 
            # determining optimal qubit reuse sets for the circuit. The default choise is set to
            # qubit at row index 0 for convenience. Subsequent choice is done by selecting the first
            # none-zero in the candidate matrix after it has been updated.

            # t = selectStartingQubit(candidate_matrix, rows_with_sum_greater_than_zero)
            t = np.random.choice(rows_with_sum_greater_than_zero)



            # After determining the initial (or subsequent) qubit, t we compute its maximum qubit reuse path.
            # This is done by assuming that all other non-zero qubits are not reusable, and then determining
            # the qubits that the qubit combination that results in the maximum qubit reuse for t. 
            # This is where we determine the maximum qubit reuse set (or path) for output (or terminal) qubit t.
            t_reuse_path, candidate_matrix = modified_maximum_qubit_reuse_paths(candidate_matrix, t)

            # if we found a reuse set for t, add it to the qubit reuse list.
            if len(t_reuse_path) > 1: 
                qubit_reuse_list.append(t_reuse_path)

        # merge qubit reuse sets into reuse lists
        qubit_reuse_list = merge_sublists(qubit_reuse_list)

        # add any qubits not included in the reuse list as their individual sets.
        qubit_reuse_list = finalize_qubit_reuse_list(qubit_reuse_list, n)

        if len(qubit_reuse_list) < len(final_qubit_reuse_list):
                final_qubit_reuse_list = qubit_reuse_list

    return final_qubit_reuse_list


#####******************************
def tupleize(lst):
    """ Recursively convert lists to tuples """
    return tuple(tupleize(x) if isinstance(x, list) else x for x in lst)

def unique_qubit_reuse_lists(list_of_lists):
    # Convert each inner list and its nested lists to a tuple
    seen_tuples = set(tupleize(inner_list) for inner_list in list_of_lists)
    
    # Convert tuples back to lists
    unique_lists = [list(map(list, t)) if isinstance(t, tuple) else t for t in seen_tuples]
    
    return unique_lists


def depth_aware_gidnet_qubit_reuse_algorithm(circuit, shots):
    """
    Computes qubit reuse sets for a given quantum circuit. The biadjacency matrix, B is the matrix
    representation of the biadjacency graph of the quantum circuit. It is compute along side the candidate
    matrix, C given by: C = np.ones(n, n) - B.T, from the quantum circuit. The biadjacency graph represents
    the connections of and input qubit to the output qubits. The rows and columns of the biadjacency matrix
    are the input qubits (or root nodes) and output qubits (terminal nodes) respectively. An entry in the 
    matrix, (r, t) with value 1, indicates that there is a connection between input qubit r and output qubit
    t, while a zero entery indicates no connection. On the other other hand, the rows and columns of the 
    Candidate matrix, C are output qubits (terminal nodes) and input qubits (or root nodes). This is the 
    opposite of the biadjacency matrix because of the transpose in the equation relationship. An entry (t, r)
    in the candidate matrix with value 1, indicates that qubit r could be used on qubit t after qubit t has
    completed its operations. In order words, it indicates a potential reuse opportunity on qubit t.
    It is important to note that the first qubit selected (i.e `initial_qubit`) is very critical for optimal
    qubit reuse outcome.
    
    After computing the biadjacency and candidate matrices, the algorithm begins by selecting one of the qubits
    (rule not yet determined, but we can use the argument `initial_qubit` to specify the qubit we want to 
    begin with). It computes the maximum reuse for that qubit using `maximum_qubit_reuse_paths`. This is done
    by assuming all other qubits are not reusable and only reusing the qubit of interest, `initial_qubit` until 
    we can no longer reuse it. Note that the selected qubit is one of the rows of the candidate matrix whose
    sum of its entries is greater than zero. That is, there is at least one non-zero entry in that row.
    
    After the first selection, the candidate matrix is updated. Subsequent qubit selections are taken from 
    the first row of the candidate matrix whose sum is greater than zero. This is conditioned on the fact that 
    the rows are arranged in ascending order (a better and more optimal rule is possible, but available yet).
    The next qubit selected undergoes the same process of determining its maximum qubit reuse path (i.e. all the 
    qubits that can be used on it). This process repeats until the candidate matrix is entirely zeros, indicating 
    that no further qubit reuse opportunities exist.

    Parameters:
    - circ: The QuantumCircuit object from which the biadjacency and candidate matrices are derived.

    Returns:
    - qubit_reuse_list: A list of sublists, where each sublist contains indices of qubits that form a chain
                        of reuse operations, based on the iterative selection and updating process.
    """
    # compute the biadjacency matrix and candidate matrix from the circuit
    main_biadjacency_matrix, main_candidate_matrix = get_biadjacency_candidate_matrix(circuit)
    # forward_cone, causal_cone = compute_input_and_output_causal_cones(main_candidate_matrix)
    
    n = circuit.num_qubits

    if np.all(main_candidate_matrix == 0):
        # raise AssertionError("This circuit is irreducible")
        print("This circuit is irreducible")
        # return [[q] for q in range(n)]
        return None

    logn = int(np.log2(n))
    # print("logn*shots", logn*shots)
    # this is where the final qubit reuse will be stored
    final_qubit_reuse_list = [[]]*n
	
    tmp_final_qubit_reuse_list = []

    # choose the num_runs not to exceed n
    # if logn*shots > n:
    #     num_runs = n
    # else:
    #     num_runs = logn*shots

    
    for _ in range(logn*shots):
        candidate_matrix = copy.deepcopy(main_candidate_matrix)
        # This is used to store the qubit reuse sets which is a list of sublists
        qubit_reuse_list = []

        # print("np.sum(candidate_matrix) = ", np.sum(candidate_matrix))
        # the loop continues untill the all the entries of the candidate matrix are all zeros.
        # that is untill, there are no more reusable edges to select from.
        while np.sum(candidate_matrix) > 0:
            # compute sum of each row
            row_sums = np.sum(candidate_matrix, axis=1)

            # Get the rows in the candidate matrix whose sums are greater than 0.
            # Each non-zero row signifies that the qubit corresponding to that row
            # can be reused.
            rows_with_sum_greater_than_zero = np.where(row_sums > 0)[0]

            # We begin with the row (qubit) that has the highest potential for qubit reuse.
            # Since we haven't determined a row for selecting such row, we use the the argument
            # `initial_qubit` as the initial qubit selected. This initial choice is very critical for 
            # determining optimal qubit reuse sets for the circuit. The default choise is set to
            # qubit at row index 0 for convenience. Subsequent choice is done by selecting the first
            # none-zero in the candidate matrix after it has been updated.

            # t = selectStartingQubit(candidate_matrix, rows_with_sum_greater_than_zero)
            t = np.random.choice(rows_with_sum_greater_than_zero)



            # After determining the initial (or subsequent) qubit, t we compute its maximum qubit reuse path.
            # This is done by assuming that all other non-zero qubits are not reusable, and then determining
            # the qubits that the qubit combination that results in the maximum qubit reuse for t. 
            # This is where we determine the maximum qubit reuse set (or path) for output (or terminal) qubit t.
            t_reuse_path, candidate_matrix = modified_maximum_qubit_reuse_paths(candidate_matrix, t)

            # if we found a reuse set for t, add it to the qubit reuse list.
            if len(t_reuse_path) > 1: 
                qubit_reuse_list.append(t_reuse_path)

        # merge qubit reuse sets into reuse lists
        qubit_reuse_list = merge_sublists(qubit_reuse_list)

        # add any qubits not included in the reuse list as their individual sets.
        qubit_reuse_list = finalize_qubit_reuse_list(qubit_reuse_list, n)
		
        tmp_final_qubit_reuse_list.append(qubit_reuse_list)

        # if len(qubit_reuse_list) < len(final_qubit_reuse_list):
        #         final_qubit_reuse_list = qubit_reuse_list

    min_reuse_size = min([len(reuse_list) for reuse_list in tmp_final_qubit_reuse_list])
    final_qubit_reuse_lists = [reuse_list for reuse_list in tmp_final_qubit_reuse_list if len(reuse_list)== min_reuse_size]
    # return final_qubit_reuse_lists
    return unique_qubit_reuse_lists(tmp_final_qubit_reuse_list)


def find_balanced_circuit(circuits, qubit_weight=0.6, depth_weight=0.4):
    """
    Finds the optimal circuit from a list of circuits by applying a weighted score
    to each circuit's number of qubits and depth. The weights reflect the relative
    importance of minimizing the number of qubits versus the depth.

    Parameters:
    - circuits (list of QuantumCircuit): The list of quantum circuits to evaluate.
    - qubit_weight (float): Weight for the number of qubits.
    - depth_weight (float): Weight for the circuit depth.

    Returns:
    - QuantumCircuit: The circuit with the lowest weighted score.
    """
    best_score = float('inf')
    optimal_circuit = None

    for circuit in circuits:
        num_qubits = circuit.num_qubits
        # depth = circuit.depth() 
        depth = circuit_depth_without_measure_and_reset(circuit)
        # Calculate the weighted score based on the provided weights.
        score = (qubit_weight * num_qubits) + (depth_weight * depth)

        # print(f"Evaluating Circuit with {num_qubits} qubits and depth of {depth}, Score: {score}")

        # Check if the current score is better and update the best score and optimal circuit.
        if score < best_score:
            best_score = score
            optimal_circuit = circuit

    return optimal_circuit

def pareto_optimal_circuits(circuits):
    """
    Finds the Pareto optimal set of circuits from a list, where no other circuit
    is better in both the number of qubits and depth. This identifies all circuits
    that are not strictly worse in both criteria compared to any other circuit.

    Parameters:
    - circuits (list of QuantumCircuit): The list of quantum circuits to evaluate.

    Returns:
    - list of QuantumCircuit: A list of Pareto optimal circuits.
    """
    optimal = []

    for circuit in circuits:
        num_qubits = circuit.num_qubits
        depth = circuit.depth()
        dominated = False
        to_remove = []

        # Compare against all currently optimal circuits to determine dominance
        for idx, comp in enumerate(optimal):
            comp_num_qubits, comp_depth, _ = comp

            # Check if current circuit dominates another in the optimal list
            if (num_qubits <= comp_num_qubits and depth < comp_depth) or (num_qubits < comp_num_qubits and depth <= comp_depth):
                to_remove.append(idx)
            # Check if current circuit is dominated
            elif (num_qubits >= comp_num_qubits and depth > comp_depth) or (num_qubits > comp_num_qubits and depth >= comp_depth):
                dominated = True

        # Remove dominated circuits from the optimal list
        for idx in sorted(to_remove, reverse=True):
            optimal.pop(idx)

        # Add current circuit if it's not dominated
        if not dominated:
            optimal.append((num_qubits, depth, circuit))

    # Return only the circuits, not the complete tuples
    return [circ[-1] for circ in optimal]

#########*************************


def modified_compute_qubit_reuse_sets(circuit, shots, improved_gidnet=False, maximize_circuit_depth=False):
    """
    Computes qubit reuse lists by iteratively selecting individual qubits for reuse based on the row
    with the highest sum in the candidate matrix. After each selection, the candidate matrix is updated,
    potentially identifying a new row with the largest sum for the next selection. This process repeats
    until the candidate matrix is entirely zeros, indicating that no further qubit reuse opportunities exist.

    Parameters:
    - circ: The QuantumCircuit object from which the biadjacency and candidate matrices are derived.

    Returns:
    - qubit_reuse_list: A list of sublists, where each sublist contains indices of qubits that form a chain
                        of reuse operations, based on the iterative selection and updating process.
    """

    if improved_gidnet:
        # compute the qubit reuse sets using improved gidnet_qubit_reuse_algorithm
        qubit_reuse_list = improved_gidnet_qubit_reuse_algorithm(circuit, maximize_circuit_depth)
    elif maximize_circuit_depth:
        qubit_reuse_list = depth_aware_gidnet_qubit_reuse_algorithm(circuit, shots)
        # print("qubit_reuse_lists = ", qubit_reuse_list)
    else:

        # compute the qubit reuse sets using gidnet_qubit_reuse_algorithm
        qubit_reuse_list = modified_gidnet_qubit_reuse_algorithm(circuit, shots)

    return qubit_reuse_list





def static_to_dynamic_circuit(circuit, qubit_reuse_list, draw=False):
    """
    Converts a static quantum circuit into a dynamic quantum circuit by adding edges to the
    original Directed Acyclic Graph (DAG) of the circuit and reordering qubits based on 
    the qubit reuse list. This transformation is aimed at optimizing qubit reuse and improving 
    the efficiency of quantum circuit execution.

    The function first computes qubit reuse sets using the `compute_qubit_reuse_sets` function. 
    It then adds edges to the original DAG circuit based on these reuse sets and removes any 
    barrier nodes to avoid cyclical DAGs and enable topological sorting. The qubits of the 
    original circuit are reordered into a new dynamic circuit based on the qubit reuse list.

    Parameters:
    - circuit (QuantumCircuit): The static quantum circuit to be transformed into a dynamic circuit.
    - iqnet (bool, optional): If True, uses an improved qnet algorithm for qubit reuse. Defaults to False.
    - draw (bool, optional): If True, displays the generated dynamic circuit using Matplotlib. Defaults to False.

    Returns:
    - QuantumCircuit: The transformed dynamic quantum circuit.
	- algorithm_type:  This is either "gidnet" or "qnet" for the sake of runtime comparism

    The function also handles the edge case where a cycle might be detected in the DAG after adding new edges.
    In such cases, an AssertionError is raised. For each node in the DAG, the function applies operations
    based on the remapped qubits, ensuring the dynamic circuit maintains the intended functionality of the
    original static circuit.
    """

    # Return the original circuit if no qubit reuse is found
    # if len(qubit_reuse_list) == circuit.num_qubits:
    if qubit_reuse_list is None:
        # print("This circuit is irreducible")
        return circuit

    circ_dag_copy, new_edges = add_qubit_reuse_edges(circuit, qubit_reuse_list) # add edges using the qubit_reuse_list

	# remove barrier nodes from the new dag to avoid cyclical DAG and enable topological sorting
    barrier_nodes = [node for node in list(circ_dag_copy.nodes()) if isinstance(node, DAGOpNode)  and node.op.name == "barrier"]

    for op_node in barrier_nodes:
        circ_dag_copy.remove_op_node(op_node)

    # Check if there is a cycle in the DAG
    if len(list(circ_dag_copy.nodes())) > len(list(circ_dag_copy.topological_nodes())):
        assert False, "A cycle detected in the DAG"

    # print("qubit_reuse_list length = ", len(qubit_reuse_list))
    old_to_new_qubit_remap_dict = {} # map old qubits to new qubits
    for new_qubit in range(len(qubit_reuse_list)):
        for old_qubit in qubit_reuse_list[new_qubit]:
            old_to_new_qubit_remap_dict[old_qubit] = new_qubit

    # print("qubit_reuse_list = ", qubit_reuse_list)

    # print("old_to_new_qubit_remap_dict = ", old_to_new_qubit_remap_dict)

    # loop through the nodes in the circuit DAG
    new_dag_copy = DAGCircuit()
	# Add quantum registers to new_dag_copy
    num_qubits = len(qubit_reuse_list) # number of qubits is the length of the qubit_reuse_list
    qregs = QuantumRegister(num_qubits)
    new_dag_copy.add_qreg(qregs)
    qubits_indices = {i: qubit for i, qubit in enumerate(new_dag_copy.qubits)}  # Assign index to each clbit from 0 to total number of 

    # Add classical registers to new_dag_copy
    num_clbits = circ_dag_copy.num_clbits()
    cregs = ClassicalRegister(num_clbits)
    new_dag_copy.add_creg(cregs)
    clbits_indices = {i: clbit for i, clbit in enumerate(new_dag_copy.clbits)}  # Assign index to each clbit from 0 to total number of clbits-1
    
    total_qr = len(qubit_reuse_list) # total number of qubits in dynamic circuit
    measure_reset_nodes = [edge[0].__hash__() for edge in new_edges] # nodes to replace with measure and reset
    remove_input_nodes = [edge[1].__hash__() for edge in new_edges] # nodes to replace with measure and reset
    track_qubits = []

    for node in circ_dag_copy.topological_nodes(): # it is important to organize it topologically
        if not isinstance(node, DAGInNode) and not isinstance(node, DAGOutNode):
            old_qubits = [n.index for n in node.qargs]
            clbits = [n.index for n in node.cargs]

            if len(old_qubits) > 1: # this is the two-qubit gate case
                new_qubits = [old_to_new_qubit_remap_dict[q] for q in old_qubits]
                new_wires = [qubits_indices[q] for q in new_qubits]
                new_qargs = tuple(new_wires)

                clbits_wires = [clbits_indices[c] for c in clbits]
                new_cargs = tuple(clbits_wires)

                new_node = DAGOpNode(op=node.op, qargs=new_qargs, cargs=new_cargs)
                new_dag_copy.apply_operation_back(new_node.op, qargs=new_node.qargs, cargs=new_node.cargs)

            else: # this is the one-qubit gate case
                old_qubits = old_qubits[0] # to eliminate list for 1-qubit
                new_qubits = old_to_new_qubit_remap_dict[old_qubits]
                new_wires = [qubits_indices[new_qubits]]
                new_qargs = tuple(new_wires)

                clbits_wires = [clbits_indices[c] for c in clbits]
                new_cargs = tuple(clbits_wires)
                new_node = DAGOpNode(op=node.op, qargs=new_qargs, cargs=new_cargs)
                new_dag_copy.apply_operation_back(new_node.op, qargs=new_node.qargs, cargs=new_node.cargs)
        else:
            old_qubits = node.wire.index
            new_qubits = old_to_new_qubit_remap_dict[old_qubits]
            new_wires = [qubits_indices[new_qubits]]
            new_qargs = tuple(new_wires)

            if isinstance(node, DAGInNode):
                continue # since we have accounted for the qubits
            elif isinstance(node, DAGOutNode):
                # DAGOutNode and hence, reset operation has no cargs
                if node.__hash__() in measure_reset_nodes:
                    reset_node = add_reset_op(new_qargs)
                    new_dag_copy.apply_operation_back(reset_node.op, qargs=reset_node.qargs, cargs=reset_node.cargs)
                else:
                    continue

    
    new_circ = dag_to_circuit(new_dag_copy)
    
    if draw:
        # display(new_circ.draw("mpl", fold=-1))
        display(new_circ.draw("mpl"))
    
    return new_circ



from qiskit.circuit import Measure, Reset
from qiskit import QuantumCircuit

def cx_depth(circuit):
    return circuit.depth(lambda x: x[0].num_qubits == 2)

# Function to remove specific types of gates
def filter_circuit_depth(original_circuit, excluded_gates=(Measure, Reset)):
    # Exclude measurement and reset gates
    # Create a new empty circuit with the same registers as the original
    new_circuit = QuantumCircuit(*original_circuit.qregs, *original_circuit.cregs)
    
    # Add only the gates that are not of the type to be excluded
    for instr, qargs, cargs in original_circuit.data:
        if not isinstance(instr, tuple(excluded_gates)):
            new_circuit.append(instr, qargs, cargs)
    depth_without_measure_reset = new_circuit.depth()
    return depth_without_measure_reset

from qiskit import QuantumCircuit
from qiskit.circuit import Measure, Reset

def circuit_depth_without_measure_and_reset(circuit):
    # Return depth, theoretically filtering out Measure and Reset
    # return circuit.depth(lambda x: not isinstance(x[0], Measure) and not isinstance(x[0], Reset))
    return circuit.depth(lambda x: not isinstance(x[0], (Measure, Reset)))


import qiskit
from importlib import resources

def convert_from_qasm_file_to_circuit(name):
    """
    Converts a QASM file into a Qiskit QuantumCircuit object while removing any redundant qubits.

    This function reads a QASM file from a specified resource package, creates a QuantumCircuit object from it,
    and then removes any qubits that are not involved in any quantum operations.

    Parameters:
    - name (str): The name of the QASM file located within the 'Qubit_Reuse.benchmarks' package resources.

    Returns:
    - QuantumCircuit: A Qiskit QuantumCircuit object with redundant qubits removed.
    """
    
    # Open the QASM file from the specified package resource
    with resources.open_text('Qubit_Reuse.benchmarks', name) as file:
        # Convert the contents of the file directly into a QuantumCircuit object
        # The 'from_qasm_file' method reads QASM from a file path, here provided by 'file.name'
        circuit = qiskit.QuantumCircuit.from_qasm_file(file.name)
        
        # Remove redundant qubits that do not participate in any quantum operations
        # to optimize the circuit complexity
        circuit = remove_redundant_qubits(circuit)
    
    # Return the optimized QuantumCircuit
    return circuit



from qiskit import QuantumCircuit

def remove_redundant_qubits(circuit):
    """
    Removes redundant qubits from a given quantum circuit. Redundant qubits are those
    that do not participate in any operation throughout the circuit.

    Parameters:
    - circuit (QuantumCircuit): The input quantum circuit from which redundant qubits are to be removed.

    Returns:
    - QuantumCircuit: A new quantum circuit with all redundant qubits removed, retaining only
      qubits and classical bits that are involved in operations.
    """
    # Initialize sets to track indices of active (used) qubits and classical bits
    active_qubits = set()
    active_clbits = set()

    # Iterate through all operations in the circuit to find active qubits and classical bits
    for instr, qargs, cargs in circuit.data:
        for q in qargs:
            # Using find_bit to get the index properly as per the latest Qiskit version
            active_qubits.add(circuit.find_bit(q).index)
        for c in cargs:
            active_clbits.add(circuit.find_bit(c).index)

    # Create a set of all qubit indices in the circuit
    all_qubits = set(range(circuit.num_qubits))

    # Calculate redundant qubits as the difference between all qubits and active qubits
    redundant_qubits = all_qubits - active_qubits

    # Create a new QuantumCircuit instance with the number of active qubits and classical bits
    new_circuit = QuantumCircuit(len(active_qubits), len(active_clbits))

    # Build mappings from old qubit/clbit indices to new indices based on active sets
    qubit_mapping = {q: idx for idx, q in enumerate(sorted(active_qubits))}
    clbit_mapping = {c: idx for idx, c in enumerate(sorted(active_clbits))}

    # Reconstruct the circuit using only active qubits and clbits
    for instr, qargs, cargs in circuit.data:
        new_qargs = [new_circuit.qubits[qubit_mapping[circuit.find_bit(q).index]] for q in qargs]
        new_cargs = [new_circuit.clbits[clbit_mapping[circuit.find_bit(c).index]] for c in cargs]
        new_circuit.append(instr, new_qargs, new_cargs)

    return new_circuit







# def modified_compile_dynamic_circuit(circuit, shots=1, qnet_circuit=None, algorithm_type="gidnet", improved_gidnet=False, qnet_method="random", qnet_shots=1, maximize_gidnet_depth=False, draw=False):
#     """
#     Converts a static quantum circuit into a dynamic quantum circuit by adding edges to the
#     original Directed Acyclic Graph (DAG) of the circuit and reordering qubits based on 
#     the qubit reuse list. This transformation is aimed at optimizing qubit reuse and improving 
#     the efficiency of quantum circuit execution.

#     The function first computes qubit reuse sets using the `compute_qubit_reuse_sets` function. 
#     It then adds edges to the original DAG circuit based on these reuse sets and removes any 
#     barrier nodes to avoid cyclical DAGs and enable topological sorting. The qubits of the 
#     original circuit are reordered into a new dynamic circuit based on the qubit reuse list.

#     Parameters:
#     - circuit (QuantumCircuit): The static quantum circuit to be transformed into a dynamic circuit.
#     - iqnet (bool, optional): If True, uses an improved qnet algorithm for qubit reuse. Defaults to False.
#     - draw (bool, optional): If True, displays the generated dynamic circuit using Matplotlib. Defaults to False.

#     Returns:
#     - QuantumCircuit: The transformed dynamic quantum circuit.
# 	- algorithm_type:  This is either "gidnet" or "qnet" for the sake of runtime comparism

#     The function also handles the edge case where a cycle might be detected in the DAG after adding new edges.
#     In such cases, an AssertionError is raised. For each node in the DAG, the function applies operations
#     based on the remapped qubits, ensuring the dynamic circuit maintains the intended functionality of the
#     original static circuit.
#     """
#     # print("I am here")
#     if algorithm_type == "gidnet" and maximize_gidnet_depth:
#         qubit_reuse_lists = modified_compute_qubit_reuse_sets(circuit, shots=shots, improved_gidnet=improved_gidnet, maximize_circuit_depth=maximize_gidnet_depth)
#         # print("qubit_reuse_lists = ", qubit_reuse_lists)
#         circuit_depth = np.inf
#         dynamic_circ = None
#         for qubit_reuse_list in qubit_reuse_lists:
#             dynamic_circuit = static_to_dynamic_circuit(circuit, qubit_reuse_list, draw)
#             if dynamic_circuit.depth() < circuit_depth:
#                 # circuit_depth = dynamic_circuit.depth()
#                 # circuit_depth = filter_circuit_depth(dynamic_circuit)
#                 circuit_depth = circuit_depth_without_measure_and_reset(dynamic_circuit)
#                 dynamic_circ = dynamic_circuit
#         return dynamic_circ
#     elif algorithm_type == "gidnet" and not maximize_gidnet_depth:
#         qubit_reuse_list = modified_compute_qubit_reuse_sets(circuit, shots=shots, improved_gidnet=improved_gidnet, maximize_circuit_depth=maximize_gidnet_depth)
#         return static_to_dynamic_circuit(circuit, qubit_reuse_list, draw)
#     elif algorithm_type == "qnet":
#         qubit_reuse_list = compute_qnet_qubit_reuse_list(qnet_circuit, method=qnet_method, shots=qnet_shots)



def modified_compile_dynamic_circuit(circuit, shots=1, qnet_circuit=None, algorithm_type="gidnet", improved_gidnet=False, qnet_method="random", qnet_shots=1, maximize_gidnet_depth=False, qubit_weight=0.6, depth_weight=0.4, draw=False):
    """
    Converts a static quantum circuit into a dynamic quantum circuit by adding edges to the
    original Directed Acyclic Graph (DAG) of the circuit and reordering qubits based on 
    the qubit reuse list. This transformation is aimed at optimizing qubit reuse and improving 
    the efficiency of quantum circuit execution.

    The function first computes qubit reuse sets using the `compute_qubit_reuse_sets` function. 
    It then adds edges to the original DAG circuit based on these reuse sets and removes any 
    barrier nodes to avoid cyclical DAGs and enable topological sorting. The qubits of the 
    original circuit are reordered into a new dynamic circuit based on the qubit reuse list.

    Parameters:
    - circuit (QuantumCircuit): The static quantum circuit to be transformed into a dynamic circuit.
    - iqnet (bool, optional): If True, uses an improved qnet algorithm for qubit reuse. Defaults to False.
    - draw (bool, optional): If True, displays the generated dynamic circuit using Matplotlib. Defaults to False.

    Returns:
    - QuantumCircuit: The transformed dynamic quantum circuit.
	- algorithm_type:  This is either "gidnet" or "qnet" for the sake of runtime comparism

    The function also handles the edge case where a cycle might be detected in the DAG after adding new edges.
    In such cases, an AssertionError is raised. For each node in the DAG, the function applies operations
    based on the remapped qubits, ensuring the dynamic circuit maintains the intended functionality of the
    original static circuit.
    """
    # print("I am here")
    if algorithm_type == "gidnet" and maximize_gidnet_depth:
        qubit_reuse_lists = modified_compute_qubit_reuse_sets(circuit, shots=shots, improved_gidnet=improved_gidnet, maximize_circuit_depth=maximize_gidnet_depth)
        if qubit_reuse_lists is None: # when there is no reuse possible
            return circuit
        # print("qubit_reuse_lists = ", qubit_reuse_lists)
        dynamic_circuits = [static_to_dynamic_circuit(circuit, qubit_reuse_list) for qubit_reuse_list in qubit_reuse_lists]
        best_dynamic_circ = find_balanced_circuit(dynamic_circuits, qubit_weight, depth_weight)
        return best_dynamic_circ
    elif algorithm_type == "gidnet" and not maximize_gidnet_depth:
        qubit_reuse_list = modified_compute_qubit_reuse_sets(circuit, shots=shots, improved_gidnet=improved_gidnet, maximize_circuit_depth=maximize_gidnet_depth)
        if qubit_reuse_lists is None:
            return circuit
        return static_to_dynamic_circuit(circuit, qubit_reuse_list, draw)
    elif algorithm_type == "qnet":
        qubit_reuse_list = compute_qnet_qubit_reuse_list(qnet_circuit, method=qnet_method, shots=qnet_shots)
        if qubit_reuse_lists is None:
            return circuit
        return static_to_dynamic_circuit(circuit, qubit_reuse_list, draw)










