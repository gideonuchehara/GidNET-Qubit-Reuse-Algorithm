import numpy as np
import math
import copy
import logging
from collections import defaultdict
import random

from qiskit import QuantumCircuit
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

from .utils import merge_subsets

import warnings
warnings.simplefilter("ignore", category=DeprecationWarning)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GidNET:
    """
    A class for implementing the GidNET Qubit Reuse Algorithm.

    Attributes:
        circuit (QuantumCircuit): The quantum circuit to be transformed.
        qubit_reuse_sequence (list): Stores the sequence of qubit reuse mappings.
        biadjacency_matrix (np.ndarray): The biadjacency matrix representing connections between input and output qubits.
        candidate_matrix (np.ndarray): The candidate matrix indicating potential qubit reuse.
        dynamic_circuit (QuantumCircuit): The dynamically transformed quantum circuit after applying qubit reuse.
    """
    def __init__(self, circuit: QuantumCircuit):
        """Initializes the GidNET algorithm with a given quantum circuit.
        
        Args:
            circuit (QuantumCircuit): The input quantum circuit to be analyzed and transformed.
        """
        self.circuit = circuit
        self.circuit_dag = circuit_to_dag(self.circuit)  # Convert the circuit to a DAG representation
        
        # compute the candidate matrix
        self.biadjacency_matrix, self.candidate_matrix = self.compute_intial_biadjacency_and_candidate_matrix()
        # self.biadjacency_matrix, self.candidate_matrix = None, None
        # self.candidate_matrix_copy = None
        self.candidate_matrix_copy = np.copy(self.candidate_matrix) # make a copy of the candidate matrix
        self.qubit_reuse_sequences = None
        self.reuse_edges = None
        self.circuit_dag_with_reuse_edges = None
        self.dynamic_circuit = None
        self.dynamic_circuit_dag = None
        self.dynamic_circuit_width = None  # set to the width of the original circuit if no reuse is possible
        self.list_of_computed_reuse_sequences = []  # this keeps track of all the reuse sequences computed so far. It is used to determine the optimal iterations to use



    def compile_to_dynamic_circuit(self, iterations: int = 10, draw: bool = False) -> QuantumCircuit:
        """
        Converts a static quantum circuit into a dynamic quantum circuit.

        This method optimizes qubit reuse by modifying the DAG structure of the circuit. It:
        - Computes the DAG and candidate matrix of the circuit.
        - Computes optimized qubit reuse sequences.
        - Adds edges to the DAG based on these sequences.
        - Removes barriers to enable topological sorting.
        - Reorders qubits based on the reuse mappings.
        - Constructs the final transformed dynamic quantum circuit.

        Args:
            iterations (int, optional): Number of iterations for the GidNET optimization algorithm. Default is 10.
            draw (bool, optional): Whether to display the transformed dynamic circuit. Default is False.

        Returns:
            QuantumCircuit: The transformed dynamic quantum circuit.
        """

        # self.circuit_dag = circuit_to_dag(self.circuit)  # Convert the circuit to a DAG representation
        
        # compute the candidate matrix
        # self.biadjacency_matrix, self.candidate_matrix = self.compute_intial_biadjacency_and_candidate_matrix()

        # self.candidate_matrix_copy = np.copy(self.candidate_matrix) # make a copy of the candidate matrix
        
        # Compute optimized qubit reuse sequences
        self.qubit_reuse_sequences = self.compute_optimized_reuse_sequences(iterations)
        
        # Return the original circuit if no qubit reuse is found
        if self.qubit_reuse_sequences is None:
            return self.circuit

        # Add edges to the DAG using the reuse sequences
        self.add_qubit_reuse_edges()
    
        # Remove barrier nodes from the DAG to prevent cycles and enable topological sorting
        barrier_nodes = [node for node in list(self.circuit_dag_with_reuse_edges.nodes()) 
                         if isinstance(node, DAGOpNode) and node.op.name == "barrier"]
        for op_node in barrier_nodes:
            self.circuit_dag_with_reuse_edges.remove_op_node(op_node)
    
        # Check if there is a cycle in the DAG
        if len(list(self.circuit_dag_with_reuse_edges.nodes())) > len(list(self.circuit_dag_with_reuse_edges.topological_nodes())):
            raise AssertionError("A cycle was detected in the DAG")
    
        # Create a mapping from old qubits to new qubits
        old_to_new_qubit_remap_dict = {}
        for new_qubit in range(len(self.qubit_reuse_sequences)):
            for old_qubit in self.qubit_reuse_sequences[new_qubit]:
                old_to_new_qubit_remap_dict[old_qubit] = new_qubit
    
        # Initialize the new DAG for the dynamic circuit
        self.dynamic_circuit_dag = DAGCircuit()
        num_qubits = len(self.qubit_reuse_sequences)
        qregs = QuantumRegister(num_qubits)
        self.dynamic_circuit_dag.add_qreg(qregs)
    
        qubits_indices = {i: qubit for i, qubit in enumerate(self.dynamic_circuit_dag.qubits)}
    
        # Add classical registers to the dynamic circuit DAG
        num_clbits = self.circuit_dag_with_reuse_edges.num_clbits()
        cregs = ClassicalRegister(num_clbits)
        self.dynamic_circuit_dag.add_creg(cregs)
        clbits_indices = {i: clbit for i, clbit in enumerate(self.dynamic_circuit_dag.clbits)}
    
        measure_reset_nodes = [edge[0].__hash__() for edge in self.reuse_edges]
        remove_input_nodes = [edge[1].__hash__() for edge in self.reuse_edges]
    
        # Iterate through the DAG nodes and apply transformations
        for node in self.circuit_dag_with_reuse_edges.topological_nodes():
            if isinstance(node, (DAGInNode, DAGOutNode)):
                continue  # Skip input and output nodes
    
            old_qubits = [n.index for n in node.qargs]
            clbits = [n.index for n in node.cargs]
    
            if len(old_qubits) > 1:
                new_qubits = [old_to_new_qubit_remap_dict[q] for q in old_qubits]
                new_qargs = tuple(qubits_indices[q] for q in new_qubits)
            else:
                new_qubits = old_to_new_qubit_remap_dict[old_qubits[0]]
                new_qargs = (qubits_indices[new_qubits],)
    
            new_cargs = tuple(clbits_indices[c] for c in clbits)
            new_node = DAGOpNode(op=node.op, qargs=new_qargs, cargs=new_cargs)
            self.dynamic_circuit_dag.apply_operation_back(new_node.op, qargs=new_node.qargs, cargs=new_node.cargs)
    
            if node.__hash__() in measure_reset_nodes:
                reset_node = self._add_reset_op(new_qargs)
                self.dynamic_circuit_dag.apply_operation_back(reset_node.op, qargs=reset_node.qargs)
    
        self.dynamic_circuit = dag_to_circuit(self.dynamic_circuit_dag)
        self.dynamic_circuit_width = self.dynamic_circuit.num_qubits
    
        if draw:
            display(self.dynamic_circuit.draw("mpl"))
    
        return self.dynamic_circuit
    

       
    
    def compute_optimized_reuse_sequences(self, iterations: int) -> Optional[List[List[int]]]:
        """
        Computes optimized qubit reuse sequences using the GidNET Qubit Reuse Algorithm.

        This method iteratively finds and optimizes qubit reuse sequences while maintaining 
        an updated candidate matrix throughout the process.

        Args:
        iterations (int): Number of iterations for optimization of GidNET Algorithm.

        Returns:
            Optional[List[List[int]]]: List of optimized reuse sequences or None if the circuit is irreducible.
        """
        # Reset the candidate matrix to its original state before optimization
        self.candidate_matrix = np.copy(self.candidate_matrix_copy)
        n = self.candidate_matrix.shape[0]
        U = [[i] for i in range(n)]  # Initialize trivial reuse sets (one per qubit)

        if np.all(self.candidate_matrix == 0):
            self.dynamic_circuit_width = self.circuit.num_qubits
            return None  # Irreducible circuit, no reuse possible

        for _ in range(int(iterations)):  # Perform optimization for the given iterations
            self.candidate_matrix = np.copy(self.candidate_matrix_copy)  # Reset the candidate matrix
            U_prime = []

            # While there are still possible qubit reuse options
            while np.sum(self.candidate_matrix) > 0:
                r = np.sum(self.candidate_matrix, axis=1)
                available_terminals = {i for i in range(n) if r[i] > 0}  # Terminals with available reuse

                terminal = random.choice(list(available_terminals))  # Select a random terminal qubit
                F = self.best_reuse_sequence(terminal)  # Compute best reuse sequence

                if len(F) > 1:
                    U_prime.append(F)  # Append only meaningful reuse sequences

            # Merge and finalize reuse sequences to remove redundancy
            U_prime = merge_subsets(U_prime)
            U_prime = self.finalize_reuse(U_prime)
            
            self.list_of_computed_reuse_sequences.append(U_prime) # this keeps track of all the reuse sequences computed so far. It is used to determine the optimal iterations to use

            # Update U only if a better optimization is found
            if len(U_prime) < len(U):
                U = U_prime

        return U
     
            
    def finalize_reuse(self, qubit_reuse_sequence: List[List[int]]) -> List[List[int]]:
        """
        Ensures that all qubits are accounted for in the final reuse sequence.
    
        This method verifies that all qubits are included in the final qubit reuse sequence.
        If a qubit is not part of any reuse sequence, it is added as a standalone sequence.
    
        Args:
            qubit_reuse_sequence (List[List[int]]): The current list of qubit reuse sequences.
    
        Returns:
            List[List[int]]: The finalized qubit reuse sequence, ensuring all qubits are included.
        """
        # Extract all qubits already included in some reuse sequence
        included_qubits = set(qubit for sublist in qubit_reuse_sequence for qubit in sublist)
        
        # Get the total set of qubits in the circuit
        all_qubits = set(range(self.circuit.num_qubits))
        
        # Identify qubits that were not included in any reuse sequence
        missing_qubits = all_qubits - included_qubits
    
        # Append missing qubits as individual sequences
        for qubit in missing_qubits:
            qubit_reuse_sequence.append([qubit])
    
        return qubit_reuse_sequence


    def best_reuse_sequence(self, terminal: int) -> Tuple[List[int], np.ndarray]:
        """
        Determines the optimized reuse sequence for a given terminal qubit.
    
        This method finds the best sequence of qubits for reuse and updates the candidate matrix accordingly.
    
        Args:
            terminal (int): The index of the terminal qubit.
    
        Returns:
            Tuple[List[int], np.ndarray]:
                - The optimized sequence of qubits for reuse.
        """
        # Initialize the reuse sequence, F  and the potential reuse sequence P
        F = [terminal]
        P = {j for j in range(self.candidate_matrix.shape[1]) if self.candidate_matrix[terminal, j] == 1}
    
        while P:
            D = defaultdict(set) # placeholder for common neighbors
    
            # Compute the intersection sets
            for root in P:
                neighbors = set.intersection(*[set(np.where(self.candidate_matrix[k] == 1)[0]) 
                                               for k in (F + [root])])
                
                D[root] = neighbors
    
            if all(len(D[root]) == 0 for root in D):
                if P:
                    root = random.choice(list(P))
                    F.append(root)
                    P.remove(root)
    
                if len(F) > 1:
                    for k in range(len(F) - 1):
                        terminal = F[k]
                        root = F[k + 1]
                        self.update_candidate_matrix(terminal, root)
    
                return F
            else:
                max_size = max(len(D[root]) for root in D)
                M = {root for root in D if len(D[root]) == max_size}
    
                if len(M) == 1:
                    root = next(iter(M))
                else:
                    S = {} # reuse scores
                    for root in M:
                        neighbors_j = D[root]
                        sigma = [len(neighbors_j & D[k]) for k in M if k != root]
                        S[root] = sum(sigma)
    
                    max_intersection = max(S.values())
                    L = [root for root in S if S[root] == max_intersection]
                    root = random.choice(L)
    
                neighbors_j = D[root]
                F.append(root)
                P = neighbors_j
    
        if len(F) > 1:
            for k in range(len(F) - 1):
                terminal = F[k]
                root = F[k + 1]
                self.update_candidate_matrix(terminal, root)
    
        return F


    def update_candidate_matrix(self, terminal: int, root: int) -> np.ndarray:
        """
        Updates the candidate matrix after the selection of specific qubits.
    
        Args:
            C (np.ndarray): The candidate matrix to be updated.
            terminal (int): The selected terminal qubit index.
            root (int): The selected root qubit index.
    
        Returns:
            np.ndarray: The updated candidate matrix.
        """
        n = self.candidate_matrix.shape[0]

        # Identify the sets Q_r and Q_t
        Q_r = {k for k in range(n) if self.candidate_matrix[terminal, k] == 0}
        Q_t = {k for k in range(n) if self.candidate_matrix[k, root] == 0}

        # Update the matrix C based on the cartesian product of Q_t and Q_r
        for k in Q_t:
            for l in Q_r:
                self.candidate_matrix[k, l] = 0

        
        self.candidate_matrix[terminal, :] = 0  # Set all entries in the row corresponding to the terminal qubit to 0
        self.candidate_matrix[:, root] = 0      # Set all entries in the column corresponding to the root qubit to 0


    def compute_intial_biadjacency_and_candidate_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the biadjacency and candidate matrices of the simplified bipartite graph 
        of the quantum circuit by searching for connections between the input qubits and
        the output qubits of the quantum circuit.
    
        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - biadjacency_matrix: A matrix representing direct connections between input and output qubits.
                - candidate_matrix: A matrix representing potential qubit reuse based on missing direct connections.
        """
        # Create a temporary copy of the circuit DAG without final measurements
        tmp_circuit = copy.deepcopy(self.circuit)
        tmp_circuit.remove_final_measurements()
        tmp_circuit_dag = circuit_to_dag(tmp_circuit)
        
        # Convert DAG to NetworkX graph
        graph = self._to_networkx(tmp_circuit_dag)
        
        # Identify input and output qubits
        roots = list(tmp_circuit_dag.input_map.values())
        terminals = list(tmp_circuit_dag.output_map.values())
        
        # Compute the biadjacency matrix
        biadjacency_matrix = np.zeros((len(roots), len(terminals)), dtype=int)
        for i, root in enumerate(roots):
            for j, terminal in enumerate(terminals):
                if nx.has_path(graph, root, terminal):
                    biadjacency_matrix[i][j] = 1
        
        # Compute the candidate matrix (inverse of biadjacency matrix)
        candidate_matrix = np.ones((len(terminals), len(roots)), dtype=int) - biadjacency_matrix.T
        
        return biadjacency_matrix, candidate_matrix


    def _to_networkx(self, circuit_dag: DAGCircuit) -> nx.MultiDiGraph:
        """Converts a Qiskit DAGCircuit into a NetworkX MultiDiGraph.
        
        This function extracts the directed acyclic graph (DAG) structure of a quantum circuit
        and represents it as a MultiDiGraph using NetworkX. This allows for easier manipulation
        and visualization of circuit dependencies.
        
        Args:
            circuit_dag (DAGCircuit): The DAG representation of the quantum circuit.
        
        Returns:
            nx.MultiDiGraph: A NetworkX graph representation of the DAG.
        
        Raises:
            ImportError: If the NetworkX library is not installed.
        """
        try:
            import networkx as nx
        except ImportError as ex:
            raise ImportError("NetworkX is required for to_networkx(). Install it using 'pip install networkx'") from ex
        
        G = nx.MultiDiGraph()
        
        # Add nodes from the Qiskit DAG
        for node in circuit_dag._multi_graph.nodes():
            G.add_node(node)
        
        # Add edges based on dependencies
        for node_id in rx.topological_sort(circuit_dag._multi_graph):
            for source_id, dest_id, edge in circuit_dag._multi_graph.in_edges(node_id):
                G.add_edge(circuit_dag._multi_graph[source_id],
                           circuit_dag._multi_graph[dest_id],
                           wire=edge)
        return G


    

    def add_qubit_reuse_edges(self) -> None:
        """
        Modifies the circuit DAG by adding edges that represent qubit reuse.
    
        This method updates `self.circuit_dag_with_reuse_edges` by adding edges that connect 
        reused qubits to the corresponding qubits that reuse them. This allows for better 
        visualization and analysis of the qubit reuse operations.
        
        Steps:
        - Creates a deep copy of `self.circuit_dag` to preserve the original DAG.
        - Identifies the input (roots) and output (terminals) qubits in the circuit.
        - Iterates over the qubit reuse sequences and forms directed edges.
        - Adds these edges to `self.circuit_dag_with_reuse_edges` to reflect qubit reuse relationships.
        
        """
        # Create a copy of the circuit DAG to modify with reuse edges
        self.circuit_dag_with_reuse_edges = copy.deepcopy(self.circuit_dag)
        total_qubits = len(self.circuit_dag_with_reuse_edges.qubits)
    
        # Identify input and output nodes of the circuit
        roots = list(self.circuit_dag_with_reuse_edges.input_map.values())  # Input qubits (roots)
        terminals = list(self.circuit_dag_with_reuse_edges.output_map.values())  # Output qubits (terminals)
    
        self.reuse_edges = []  # Store reuse edges for tracking
    
        # Iterate through each reuse sequence to establish edges
        for qubit_sequence in self.qubit_reuse_sequences:
            if len(qubit_sequence) > 1:
                # Create directed edges from reuse sequences
                reuse_edges_indx = [(qubit_sequence[i], qubit_sequence[i + 1]) 
                                    for i in range(len(qubit_sequence) - 1)]
                for root, terminal in reuse_edges_indx:
                    # The root was previously a terminal before reuse
                    # The terminal is the qubit that will be reused by the root
                    self.reuse_edges.append((terminals[root], roots[terminal]))
    
        # Add the reuse edges to the DAG
        for root_node, terminal_node in self.reuse_edges:
            self.circuit_dag_with_reuse_edges._multi_graph.add_edge(
                root_node._node_id, terminal_node._node_id, root_node.wire
            )


    def _add_reset_op(self, qargs: List[Qubit]) -> DAGOpNode:
        """
        Creates a reset operation node to be added to a DAGCircuit.

        The reset operation resets the qubit state to |0⟩, ensuring that it can be reused.
        This function constructs a DAGOpNode containing the reset instruction.

        Args:
            qargs (List[Qubit]): List of qubits to be reset.

        Returns:
            DAGOpNode: A DAG node representing the reset operation.
        """
        # Define a reset operation with no classical bits
        reset_op = Instruction(name='reset', num_qubits=1, num_clbits=0, params=[])
        
        # Create and return a DAGOpNode representing the reset operation
        reset_node = DAGOpNode(op=reset_op, qargs=qargs, cargs=())
        return reset_node

    

    

    

if __name__ == "__main__":
    logging.info("GidNET Qubit Reuse Algorithm Initialized")

