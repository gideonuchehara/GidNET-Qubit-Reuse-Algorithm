################### From Baidu QNET ############################

import numpy

from Extensions.QuantumNetwork.qcompute_qnet.quantum.circuit import Circuit
import numpy
from typing import List, Tuple, Optional, Union, Any
import copy
import itertools
import random

from QCompute import *
from pprint import pprint
from QCompute.Define import Settings
Settings.outputInfo = False


from QCompute.QPlatform import Error
from QCompute.OpenService import ModuleErrorCode
# from QCompute.OpenService.service_ubqc.client.qobject import Circuit
from Extensions.QuantumNetwork.qcompute_qnet.quantum.circuit import Circuit
from QCompute.QProtobuf import PBCircuitLine, PBFixedGate, PBRotationGate, PBMeasure, PBProgram

FileErrorCode = 1

def pb_to_circuit(program: PBProgram):
        r"""Translate ``PBProgram`` into the ``Circuit`` class.

        Args:
            program (PBProgram): the published quantum program

        Returns:
            Circuit: a quantum circuit which supports the translation to its equivalent MBQC model
        """
        pbCircuit = program.body.circuit

        # Obtain the circuit width
        bit_idxes = set()
        for gate in pbCircuit:
            bit_idxes.update(gate.qRegList)
        width = max(bit_idxes) + 1
        if width <= 0:
            raise Error.ArgumentError(f"Invalid circuit ({program}) in the program!\n"
                                      "This circuit is empty and has no qubit.",
                                      ModuleErrorCode,
                                      FileErrorCode, 7)

        # Instantiate ``Circuit`` and map gates and measurements to methods in ``Circuit``
        circuit = Circuit(width)

        # Warning: In the circuit model, the quantum states are initialized with |0>.
        # While in MBQC model, the quantum states are initialized with |+>.
        # Therefore, each qubit in the MBQC circuit should be operated by a ``Hadamard`` gate in the front.
        for i in range(width):
            circuit.h(i)

        for PBGate in pbCircuit:
            op = PBGate.WhichOneof('op')
            # Map ``fixedGate`` (including 'H', 'CX', 'X', 'Y', 'Z', 'S', 'T', 'CZ') to the methods in ``Circuit``
            if op == 'fixedGate':
                fixedGate: PBFixedGate = PBGate.fixedGate
                gateName = PBFixedGate.Name(fixedGate)
                bit_idx = PBGate.qRegList
                if gateName == 'H':
                    # print("I am here")
                    circuit.h(bit_idx[0])
                elif gateName == 'CX':
                    circuit.cnot(bit_idx)
                elif gateName == 'X':
                    circuit.x(bit_idx[0])
                elif gateName == 'Y':
                    circuit.y(bit_idx[0])
                elif gateName == 'Z':
                    circuit.z(bit_idx[0])
                elif gateName == 'S':
                    circuit.s(bit_idx[0])
                elif gateName == 'T':
                    circuit.t(bit_idx[0])
                elif gateName == 'CZ':
                    # CZ [q1, q2] = H [q2] + CNOT [q1, q2] + H [q2]
                    circuit.h(bit_idx[1])
                    circuit.cnot(bit_idx)
                    circuit.h(bit_idx[1])
                else:
                    raise Error.ArgumentError(f"Invalid gate: ({gateName})!\n"
                                              "Only 'H', 'CX', 'X', 'Y', 'Z', 'S', 'T', 'CZ' are supported as the "
                                              "fixed gates in UBQC in this version.", ModuleErrorCode,
                                              FileErrorCode, 8)

            # Map ``rotationGate`` (including 'RX', 'RY', 'RZ', 'U') to the methods in ``Circuit``
            elif op == 'rotationGate':
                rotationGate: PBRotationGate = PBGate.rotationGate
                gateName = PBRotationGate.Name(rotationGate)
                bit_idx = PBGate.qRegList

                if gateName == 'RX':
                    # print("PBGate.argumentValueList[0] = ", PBGate.argumentValueList[0])
                    # circuit.rx(PBGate.argumentValueList[0], bit_idx[0])
                    circuit.rx(bit_idx[0], PBGate.argumentValueList[0])
                elif gateName == 'RY':
                    # circuit.ry(PBGate.argumentValueList[0], bit_idx[0])
                    circuit.ry(bit_idx[0], PBGate.argumentValueList[0])
                elif gateName == 'RZ':
                    # circuit.rz(PBGate.argumentValueList[0], bit_idx[0])
                    circuit.rz(bit_idx[0], PBGate.argumentValueList[0])

                # Warning: unitary gate in MBQC has a decomposition form different from the commonly used ``U3`` gate!
                elif gateName == 'U':
                    # In circuit model, the ``U3`` gate has a decomposition form of "Rz Ry Rz",
                    # with angles of "theta, phi, lamda", that is:
                    # U3(theta, phi, lamda) = Rz(phi) Ry(theta) Rz(lamda)
                    angles = PBGate.argumentValueList

                    # Warning: Sometimes, The angles have only one or two valid parameters!
                    # In these cases, set the other parameters to be zeros
                    if len(angles) == 1:
                        theta1 = angles[0]
                        phi1 = 0
                        lamda1 = 0
                    elif len(angles) == 2:
                        theta1 = angles[0]
                        phi1 = angles[1]
                        lamda1 = 0
                    else:
                        theta1 = angles[0]
                        phi1 = angles[1]
                        lamda1 = angles[2]
                    u3 = u3_gate(theta1, phi1, lamda1)
                    # u3 = circuit.u(theta1, phi1, lamda1)

                    # In MBQC model, the unitary gate has a decomposition form of "Rz Rx Rz",
                    # with angles of "theta, phi, lamda", that is:
                    # U(theta, phi, lamda) = Rz(phi) Rx(theta) Rz(lamda)
                    theta2, phi2, lamda2 = decompose(u3)

                    circuit.u(theta2, phi2, lamda2, bit_idx[0])

            elif op == 'customizedGate':
                raise Error.ArgumentError(f"Invalid gate type: ({op})!\n"
                                          "Customized gates are not supported in UBQC in this version.",
                                          ModuleErrorCode,
                                          FileErrorCode, 9)

            elif op == 'measure':
                # measurement_qubits = set(PBGate.qRegList)
                # if measurement_qubits != set(range(width)):
                #     raise Error.ArgumentError(f"Invalid measurement qubits: ({measurement_qubits})!\n"
                #                               "All qubits must be measured in UBQC in this version.",
                #                               ModuleErrorCode,
                #                               FileErrorCode, 10)

                for qReg in PBGate.qRegList:
                    typeName: PBMeasure = PBMeasure.Type.Name(PBGate.measure.type)
                    if typeName == 'Z':
                        circuit.measure(qReg)
                    else:
                        raise Error.ArgumentError(f"Invalid measurement type: ({typeName})!\n"
                                                  "Only 'Z measurement' is supported as the measurement type "
                                                  "in UBQC in this version.",
                                                  ModuleErrorCode,
                                                  FileErrorCode, 11)
            else:
                raise Error.ArgumentError(f"Invalid operation: ({op})!\n"
                                          "This operation is not supported in UBQC in this version.",
                                          ModuleErrorCode,
                                          FileErrorCode, 12)

        return circuit


def remove_barrier_from_circuit(circuit):
    """
    This function removes barrier from a quantum circuit
    """

    from qiskit.converters import circuit_to_dag, dag_to_circuit
    from qiskit.dagcircuit.dagnode import DAGInNode, DAGNode, DAGOpNode, DAGOutNode

    circuit_dag = circuit_to_dag(circuit)

    # remove barrier nodes from the new dag to avoid cyclical DAG and enable topological sorting
    barrier_nodes = [node for node in list(circuit_dag.nodes()) if isinstance(node, DAGOpNode)  and node.op.name == "barrier"]

    for op_node in barrier_nodes:
        circuit_dag.remove_op_node(op_node)

    new_circuit = dag_to_circuit(circuit_dag)

    return new_circuit


def from_qiskit_to_qnet(circuit):
    """
    This function converts a qiskit circuit to Baidu's
    quantum circuit format
    """
    # Remove barriers from quantum circuit
    new_circ = remove_barrier_from_circuit(circuit)

    # Qiskit circuit to QASM
    qasmStr = new_circ.qasm()

    # Convert QASM to PBProgram and output
    pb_circuit = QasmToCircuit().convert(qasmStr)

    # Convert PBProgram to QNET Circuit
    converted_circuit = pb_to_circuit(pb_circuit)

    # reduce_by_greedy(converted_circuit, method="deterministic", draw=False)
    return converted_circuit





@staticmethod
def __update_candidate_matrix(candidate_matrix: numpy.ndarray, t: int, r: int) -> numpy.ndarray:
    r"""Update the candidate matrix after adding an edge from terminal indexed by t to root indexed by r.

    Args:
        candidate_matrix (numpy.ndarray): the candidate matrix to update
        t (int): index of the terminal vertex
        r (int): index of the root vertex

    Returns:
        numpy.ndarray: the updated candidate matrix after adding an edge
        from terminal indexed by t and root indexed by r
    """
    row, col = candidate_matrix.shape
    # Identify all roots capable of reaching terminal indexed by t
    root_set = {i for i in range(col) if candidate_matrix[t][i] == 0}
    # Identify all terminals that are reachable from root indexed by r
    terminal_set = {j for j in range(row) if candidate_matrix[j][r] == 0}
    # After adding this edge, any terminal in terminal_set is reachable from any root in root_set
    pair_collections = []
    for pair in itertools.product(terminal_set, root_set):
        candidate_matrix[pair[0]][pair[1]] = 0
        pair_collections.append(pair)
    # The added edges should not share common vertices
    candidate_matrix[t, :] = 0
    candidate_matrix[:, r] = 0

    return candidate_matrix

def _greedy_heuristic(
        circuit, candidate_matrix: numpy.ndarray, roots: List[Any], terminals: List[Any], method: str
    ) -> list:
        r"""Apply greedy heuristic algorithm to the candidate matrix of the simplified bipartite graph
        corresponding to a quantum circuit to obtain a list of edges can be added to the graph
        without introducing any directed cycles.

        Args:
            candidate_matrix (numpy.ndarray): the candidate matrix of the simplified bipartite graph
            roots (List[Any]): a list of root vertices in the graph
            terminals (List[Any]): a list of terminal vertices in the graph
            method (str): specific method to identify the local optimum during each iteration of the greedy algorithm

        Returns:
            list: a list of edges can be added to the simplified bipartite graph
            without introducing any directed cycles
        """

        def _add_one_edge_by_greedy(candidate: numpy.ndarray, method=method) -> Tuple[tuple, numpy.ndarray]:
            r"""Identify a candidate edge by greedy heuristic algorithm and update the candidate matrix after
            adding this candidate edge.

            Args:
                candidate (numpy.ndarray): the candidate matrix of the graph
                method (str): specific method to identify the candidate edge during each iteration

            Returns:
                tuple: a tuple containing a candidate edge added to the graph and the updated candidate matrix
                after adding this edge to the graph.
            """
            # If there is no more candidate edge, then return an empty edge
            if numpy.count_nonzero(candidate) == 0:
                return tuple(), candidate

            # Initialize a zero matrix to record scores of candidate edges
            score_matrix = numpy.zeros((candidate.shape[0], candidate.shape[1]), dtype=int)
            # Calculate the scores of all candidate edges
            for i in range(candidate.shape[0]):
                for j in range(candidate.shape[1]):
                    if candidate[i][j] != 0:  # non-zero entry indicates a candidate edge
                        # Update the candidate matrix after adding the candidate edge (t_i, r_j)
                        candidate_copy = copy.deepcopy(candidate)
                        candidate_copy = __update_candidate_matrix(candidate_copy, i, j)
                        # Use the number of remaining candidate as the score of candidate edge (t_i, r_j)
                        # Plus one is used to distinguish candidate edges from non-candidate edges
                        score_matrix[i][j] += numpy.sum(candidate_copy) + 1

            # Identify the target edge as the candidate edge with the highest score
            if method == "deterministic":
                # Deterministically select the first one if multiple candidate edges share the highest score
                edge = numpy.unravel_index(numpy.argmax(score_matrix), score_matrix.shape)
                u, v = edge
            elif method == "random":
                # Randomly select one if multiple candidate edges share the highest score
                max_score_candidates = numpy.where(score_matrix == numpy.max(score_matrix))
                index = random.randint(0, len(max_score_candidates[0]) - 1)
                u = max_score_candidates[0][index]
                v = max_score_candidates[1][index]
                edge = (u, v)
            else:
                raise NotImplementedError

            # Update the candidate matrix after adding the target edge
            candidate = __update_candidate_matrix(candidate, u, v)

            return edge, candidate

        added_edges = []
        added_qubits = []
        
        # Iteratively add candidate edges to the graph until the candidate matrix becomes a zero matrix
        flag = 1
        while flag > 0:
            new_edge, candidate_matrix = _add_one_edge_by_greedy(candidate_matrix, method=method)
            if len(new_edge) != 0:
                added_edge = (terminals[new_edge[0]], roots[new_edge[1]])
                added_edges.append(added_edge)
                added_qubits.append([new_edge[0], new_edge[1]])
            flag = len(new_edge)

        return added_edges, added_qubits

def reduce_by_greedy(circuit, method=None, shots=1, draw=False) -> None:
        r"""Compile a quantum circuit into an equivalent dynamic circuit with fewer qubits
        by the greedy heuristic algorithm.

        Args:
            method (optional, str): specific method to identify the candidate edge at each iteration
            shots: (optional, int): number of times to run random greedy algorithm
            draw: (optional, bool): whether to draw the modified graph with added edges

        Note:
            If there are multiple local optimal candidate edges during one iteration,
            there are two different methods to identify which one to choose:

            1. Deterministic selection: always select the first one encountered ("deterministic")
            2. Random selection: randomly select one from all local optima ("random")

            If no method is specified, then by default the random greedy algorithm will be executed once.

            If 'shots' is set to a value greater than one, the random greedy algorithm will be run multiple
            times. It will return the dynamic circuit with the minimal circuit width across all runs.

            Multiple runs of the deterministic greedy algorithm produce consistent result.
        """
        
        # Get the graph representation of the circuit
        graph, roots, terminals = circuit.to_dag()
        # Obtain the initial candidate matrix of the graph
        candidate_matrix = circuit._get_candidate_matrix_by_boolean_matrix()
        # Get a list of edges added to the graph with specified method
        if method == "random" or method is None:
            added_edges = []
            added_qubits = []
            # Run random greedy algorithm multiple times and return the dynamic circuit with the minimal width
            for shot in range(shots):
                candidate_copy = copy.deepcopy(candidate_matrix)
                new_edges, new_qubits = _greedy_heuristic(circuit, candidate_copy, roots, terminals, method="random")
                if len(new_edges) > len(added_edges):
                    added_edges = new_edges
                    added_qubits = new_qubits
        elif method == "deterministic":
            if shots > 1:
                print("\nThe deterministic greedy algorithm produces consistent results across multiple runs.")
            added_edges, added_qubits = _greedy_heuristic(circuit, candidate_matrix, roots, terminals, method="deterministic")
        else:
            raise NotImplementedError

        # Compile the input static circuit into dynamic circuit with the modified graph and added edges
        new_graph = graph.copy()
        new_graph.add_edges_from(added_edges)
        if draw is True:  # draw the modified graph with added edges
            circuit._draw_dag(new_graph, roots, terminals, added_edges)
        circuit._reorder_by_dag(new_graph, added_edges)
        # circuit.remap_indices(print_index=False)
        circuit.remap_indices(print_index=True)




def apply_qnet_qubit_reuse(circ, method="deterministic", draw=False):
    """
    Given a Qiskit Quantum circuit, this function converts it to
    QNET circuit format and applies QNET's 'greedy' qubit reuse
    algorithm to convert the circuit to a dynamic circuit.
    """

    # convert circuit to qnet circuit format
    converted_circuit = from_qiskit_to_qnet(circ)

    # Check whether the current circuit width can be reduced
    if converted_circuit.is_reducible():
        # Reduce the circuit by 'hybrid' algorithm
        # converted_circuit.reduce_by_hybrid(level=3, draw=False)

        # Reduce the circuit by 'greedy' algorithm
        # converted_circuit.reduce("greedy")
        if method=="deterministic":
            converted_circuit.reduce_by_greedy(method="deterministic", shots=1)
        else:
            converted_circuit.reduce_by_greedy(method="random", shots=1)

    # Print the dynamic quantum circuit
    if draw:
        converted_circuit.print_circuit()
    
    return converted_circuit



###################### IMPORTANT FUNCTION ###############################
def remove_barrier_from_circuit(circuit):
    """
    This function removes barrier from a quantum circuit
    """
    
    from qiskit.converters import circuit_to_dag, dag_to_circuit
    from qiskit.dagcircuit.dagnode import DAGInNode, DAGNode, DAGOpNode, DAGOutNode
    
    circuit_dag = circuit_to_dag(circuit)
    
    # remove barrier nodes from the new dag to avoid cyclical DAG and enable topological sorting
    barrier_nodes = [node for node in list(circuit_dag.nodes()) if isinstance(node, DAGOpNode)  and node.op.name == "barrier"]
    
    for op_node in barrier_nodes:
        circuit_dag.remove_op_node(op_node)
        
    new_circuit = dag_to_circuit(circuit_dag)
    
    return new_circuit



############### Extract only Edges Algorithm ##########################33
def _greedy_heuristic(
        circuit, candidate_matrix: numpy.ndarray, roots: List[Any], terminals: List[Any], method: str
    ) -> list:

    r"""Apply greedy heuristic algorithm to the candidate matrix of the simplified bipartite graph
    corresponding to a quantum circuit to obtain a list of edges can be added to the graph
    without introducing any directed cycles.

    Args:
        candidate_matrix (numpy.ndarray): the candidate matrix of the simplified bipartite graph
        roots (List[Any]): a list of root vertices in the graph
        terminals (List[Any]): a list of terminal vertices in the graph
        method (str): specific method to identify the local optimum during each iteration of the greedy algorithm

    Returns:
        list: a list of edges can be added to the simplified bipartite graph
        without introducing any directed cycles
    """

    def _add_one_edge_by_greedy(candidate: numpy.ndarray, method=method) -> Tuple[tuple, numpy.ndarray]:
        r"""Identify a candidate edge by greedy heuristic algorithm and update the candidate matrix after
        adding this candidate edge.

        Args:
            candidate (numpy.ndarray): the candidate matrix of the graph
            method (str): specific method to identify the candidate edge during each iteration

        Returns:
            tuple: a tuple containing a candidate edge added to the graph and the updated candidate matrix
            after adding this edge to the graph.
        """
        # If there is no more candidate edge, then return an empty edge
        if numpy.count_nonzero(candidate) == 0:
            return tuple(), candidate

        # Initialize a zero matrix to record scores of candidate edges
        score_matrix = numpy.zeros((candidate.shape[0], candidate.shape[1]), dtype=int)
        # Calculate the scores of all candidate edges
        for i in range(candidate.shape[0]):
            for j in range(candidate.shape[1]):
                if candidate[i][j] != 0:  # non-zero entry indicates a candidate edge
                    # Update the candidate matrix after adding the candidate edge (t_i, r_j)
                    candidate_copy = copy.deepcopy(candidate)
                    candidate_copy = __update_candidate_matrix(candidate_copy, i, j)
                    # Use the number of remaining candidate as the score of candidate edge (t_i, r_j)
                    # Plus one is used to distinguish candidate edges from non-candidate edges
                    score_matrix[i][j] += numpy.sum(candidate_copy) + 1

        # Identify the target edge as the candidate edge with the highest score
        if method == "deterministic":
            # Deterministically select the first one if multiple candidate edges share the highest score
            edge = numpy.unravel_index(numpy.argmax(score_matrix), score_matrix.shape)
            u, v = edge
        elif method == "random":
            # Randomly select one if multiple candidate edges share the highest score
            max_score_candidates = numpy.where(score_matrix == numpy.max(score_matrix))
            index = random.randint(0, len(max_score_candidates[0]) - 1)
            u = max_score_candidates[0][index]
            v = max_score_candidates[1][index]
            edge = (u, v)
        else:
            raise NotImplementedError

        # Update the candidate matrix after adding the target edge
        candidate = __update_candidate_matrix(candidate, u, v)

        return edge, candidate

    added_edges = []
    added_qubits = []

    # Iteratively add candidate edges to the graph until the candidate matrix becomes a zero matrix
    flag = 1
    while flag > 0:
        new_edge, candidate_matrix = _add_one_edge_by_greedy(candidate_matrix, method=method)
        if len(new_edge) != 0:
            added_edge = [[new_edge[0]], [new_edge[1]]]
            added_edges.append(added_edge)
            added_qubits.append([new_edge[0], new_edge[1]])
        flag = len(new_edge)

    return added_edges, added_qubits


def modified_reduce_by_greedy(circuit, method=None, shots=1, draw=False) -> None:
    
    r"""Compile a quantum circuit into an equivalent dynamic circuit with fewer qubits
    by the greedy heuristic algorithm.

    Args:
        method (optional, str): specific method to identify the candidate edge at each iteration
        shots: (optional, int): number of times to run random greedy algorithm
        draw: (optional, bool): whether to draw the modified graph with added edges

    Note:
        If there are multiple local optimal candidate edges during one iteration,
        there are two different methods to identify which one to choose:

        1. Deterministic selection: always select the first one encountered ("deterministic")
        2. Random selection: randomly select one from all local optima ("random")

        If no method is specified, then by default the random greedy algorithm will be executed once.

        If 'shots' is set to a value greater than one, the random greedy algorithm will be run multiple
        times. It will return the dynamic circuit with the minimal circuit width across all runs.

        Multiple runs of the deterministic greedy algorithm produce consistent result.
    """

    # Get the graph representation of the circuit
    graph, roots, terminals = circuit.to_dag()
    # Obtain the initial candidate matrix of the graph
    candidate_matrix = circuit._get_candidate_matrix_by_boolean_matrix()
    # Get a list of edges added to the graph with specified method
    if method == "random" or method is None:
        added_edges = []
        added_qubits = []
        # Run random greedy algorithm multiple times and return the dynamic circuit with the minimal width
        for shot in range(shots):
            candidate_copy = copy.deepcopy(candidate_matrix)
            new_edges, new_qubits = _greedy_heuristic(circuit, candidate_copy, roots, terminals, method="random")
            if len(new_edges) > len(added_edges):
                added_edges = new_edges
                added_qubits = new_qubits
    elif method == "deterministic":
        if shots > 1:
            print("\nThe deterministic greedy algorithm produces consistent results across multiple runs.")
        added_edges, added_qubits = _greedy_heuristic(circuit, candidate_matrix, roots, terminals, method="deterministic")
    else:
        raise NotImplementedError

    return added_qubits

# def modified_apply_qnet_qubit_reuse(qnet_circuit, method="random", shots=1, draw=False):

#     """
#     Given a Qiskit Quantum circuit, this function converts it to
#     QNET circuit format and applies QNET's 'greedy' qubit reuse
#     algorithm to convert the circuit to a dynamic circuit.
#     """

#     # convert circuit to qnet circuit format
#     # converted_circuit = from_qiskit_to_qnet(circ)
#     qubit_reuse_list = modified_reduce_by_greedy(qnet_circuit, method, shots)

#     return qubit_reuse_list


def compute_qnet_qubit_reuse_list_timing(circ, method="random", shots=1, draw=False):
    """
    Computes only the sets of qubits that can be reused together. 
    It does not create a chain or list of more than two qubits.
    This function is primarily used to time the QNET algorithm.
    """
    qubit_reuse_list = modified_reduce_by_greedy(circ, method, shots, draw)
    return qubit_reuse_list



def compute_qnet_qubit_reuse_list(circ, method="random", shots=1, draw=False):
    """
    Computes the full qubit reuse chain (list) for a quantum circuit.
    """
    qubit_reuse_list = compute_qnet_qubit_reuse_list_timing(circ, method, shots, draw)
    qubit_reuse_list = merge_sublists(qubit_reuse_list)
    qubit_reuse_list = finalize_qubit_reuse_list(qubit_reuse_list, circ.width)
    return qubit_reuse_list


############# Helpers ###################
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


