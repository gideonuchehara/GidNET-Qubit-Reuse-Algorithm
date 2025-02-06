#### Helper Functions for GidNET

def merge_subsets(list_of_pairs):
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
        pair = list(pair)  # Convert set to list to maintain order
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
            for j in range(i + 1, len(merged_list)):
                if any(elem in merged_list[j] for elem in merged_list[i]):
                    # Merge i-th and j-th sublists and remove the j-th sublist
                    merged_list[i] = merge_and_order(merged_list[i], merged_list[j])
                    del merged_list[j]
                    merge_occurred = True
                    break
            if merge_occurred:
                break

    return merged_list


##************ Qiskit Qubit Reuse ************************###
from qiskit.converters import circuit_to_dag, dag_to_circuit
from benchmarks.qiskit_qubit_reuse import qubit_reuse, qubit_reuse_greedy
def apply_qiskit_qubit_reuse(cirucit):
    qr = qubit_reuse.QubitReuseModified()
    cirucit_dag= circuit_to_dag(cirucit)
    qr_cirucit = dag_to_circuit(qr.run(cirucit_dag))
    return qr_cirucit



########## QUANTUM CUIRCUIT BENCHMARKS #####################################

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



def remove_idle_qubits_from_circuit(qc):
    """
    This function removes idle qubits from a quantum circuit and returns 
    the updated circuit without idle qubits.
    """
    from qiskit.converters import circuit_to_dag, dag_to_circuit
    
    dag_qc = circuit_to_dag(qc)
    idle_wires = list(dag_qc.idle_wires())
    for wire in idle_wires: 
        dag_qc.remove_qubits(wire)

    clean_qc = dag_to_circuit(dag_qc)

    return clean_qc


