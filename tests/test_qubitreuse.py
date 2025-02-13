import pytest
import numpy as np
from qiskit import QuantumCircuit
from gidnet.qubitreuse import GidNET

@pytest.fixture
def simple_circuit():
    """Creates a simple quantum circuit with 3 qubits."""
    qc = QuantumCircuit(3)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.measure_all()
    return qc

@pytest.fixture
def complex_circuit():
    """Creates a more complex quantum circuit with 5 qubits."""
    qc = QuantumCircuit(5)
    qc.cx(1, 2)
    qc.cx(0, 3)
    qc.cx(1, 4)
    qc.cx(2, 4)
    qc.cx(3, 4)
    qc.measure_all()
    return qc

@pytest.fixture
def gidnet_simple(simple_circuit):
    """Fixture for initializing GidNET with a simple circuit."""
    # qc = simple_circuit()
    return GidNET(simple_circuit)

@pytest.fixture
def gidnet_complex(complex_circuit):
    """Fixture for initializing GidNET with a complex circuit."""
    # qc = complex_circuit()
    return GidNET(complex_circuit)

def test_initialization(gidnet_simple):
    """Test if the GidNET instance initializes correctly."""
    assert gidnet_simple.circuit is not None
    assert gidnet_simple.dynamic_circuit is None
    assert gidnet_simple.qubit_reuse_sequences is None


def test_compute_initial_biadjacency_and_candidate_matrix(gidnet_simple, gidnet_complex):
    """Test if biadjacency and candidate matrices are generated properly."""
    intial_biadjacency_matrix, intial_candidate_matrix = gidnet_simple.compute_intial_biadjacency_and_candidate_matrix()
    assert isinstance(intial_biadjacency_matrix, np.ndarray)
    assert isinstance(intial_candidate_matrix, np.ndarray)
    simple_biadjacency_mat = np.array([[1, 1, 1], 
                                       [1, 1, 1], 
                                       [0, 1, 1]])
    simple_candidate_mat = (np.ones((3, 3), dtype=int) - np.array([[1, 1, 1], 
                                                                   [1, 1, 1], 
                                                                   [0, 1, 1]])).T
    assert np.allclose(intial_biadjacency_matrix, simple_biadjacency_mat)
    assert np.allclose(intial_candidate_matrix, simple_candidate_mat)

    # test for complex circuit
    intial_biadjacency_matrix, intial_candidate_matrix = gidnet_complex.compute_intial_biadjacency_and_candidate_matrix()
    assert isinstance(intial_biadjacency_matrix, np.ndarray)
    assert isinstance(intial_candidate_matrix, np.ndarray)
    complex_biadjacency_mat = np.array([[1, 0, 0, 1, 1], 
                                        [0, 1, 1, 1, 1], 
                                        [0, 1, 1, 1, 1], 
                                        [1, 0, 0, 1, 1], 
                                        [0, 1, 1, 1, 1]])
    complex_candidate_mat = (np.ones((5, 5), dtype=int) - np.array([[1, 0, 0, 1, 1], 
                                                                    [0, 1, 1, 1, 1], 
                                                                    [0, 1, 1, 1, 1], 
                                                                    [1, 0, 0, 1, 1], 
                                                                    [0, 1, 1, 1, 1]])).T
    assert np.allclose(intial_biadjacency_matrix, complex_biadjacency_mat)
    assert np.allclose(intial_candidate_matrix, complex_candidate_mat)

@pytest.mark.parametrize("iterations", [10, 20])
def test_compile_to_dynamic_circuit(gidnet_simple, gidnet_complex, iterations):
    """Test if GidNET compiles a valid dynamic circuit."""
    dynamic_circ = gidnet_simple.compile_to_dynamic_circuit(iterations=iterations)
    assert isinstance(dynamic_circ, QuantumCircuit)
    assert dynamic_circ.num_qubits > 0  # Ensuring qubits exist
    assert sorted(gidnet_simple.qubit_reuse_sequences) == sorted([[0, 2], [1]]) # Ensuring correct reuse sequence is returned

    # test for complex circuit
    dynamic_circ = gidnet_complex.compile_to_dynamic_circuit(iterations=iterations)
    assert isinstance(dynamic_circ, QuantumCircuit)
    assert dynamic_circ.num_qubits > 0  # Ensuring qubits exist
    # assert (sorted(gidnet_complex.qubit_reuse_sequences) == sorted([[1, 0], [2, 3], [4]])) | (sorted(gidnet_complex.qubit_reuse_sequences) == sorted([[1, 3], [2, 0], [4]])) # Ensuring correct reuse sequence is returned
    
    
    # Define the set of expected qubit reuse sequences (order of outer list does not matter)
    expected_sequences = [
        [[1, 0], [2, 3], [4]],   # Expected valid output 1
        [[1, 3], [2, 0], [4]]    # Expected valid output 2
    ]
    
    # Check if the actual output matches any expected sequence
    assert any(sorted(gidnet_complex.qubit_reuse_sequences) == sorted(seq) for seq in expected_sequences)
    
    
@pytest.mark.parametrize("row, col", [(0, 1), (1, 2), (2, 3)])
def test_update_candidate_matrix(gidnet_complex, row, col):
    """Test if candidate matrix updates correctly."""
    _, original_matrix = gidnet_complex.compute_intial_biadjacency_and_candidate_matrix()
    gidnet_complex.update_candidate_matrix(row, col)
    assert not np.array_equal(original_matrix, gidnet_complex.candidate_matrix)
    
    
@pytest.mark.parametrize("terminal", [1, 2, 3])
def test_best_reuse_sequence(gidnet_complex, terminal):
    """Test that best_reuse_sequence returns a valid sequence."""
    reuse_sequence = gidnet_complex.best_reuse_sequence(terminal)
    assert isinstance(reuse_sequence, list)
    assert terminal in reuse_sequence
    
    
@pytest.mark.parametrize("iterations", [10, 20])
def test_compute_optimized_reuse_sequences(gidnet_complex, iterations):
    """Test if optimized qubit reuse sequences are computed."""
    reuse_sequences = gidnet_complex.compute_optimized_reuse_sequences(iterations=iterations)
    assert isinstance(reuse_sequences, list)
    assert all(isinstance(seq, list) for seq in reuse_sequences)
    
    
@pytest.mark.parametrize("terminal", [1, 2, 3])
def test_best_reuse_sequence(gidnet_complex, terminal):
    """Test that best_reuse_sequence returns a valid sequence."""
    reuse_sequence = gidnet_complex.best_reuse_sequence(terminal)
    assert isinstance(reuse_sequence, list)
    assert terminal in reuse_sequence
    
    
    
def test_finalize_reuse(gidnet_complex):
    """Test if finalize_reuse correctly processes reuse sequences."""
    test_sequence = [[1, 0], [2, 3]]
    final_sequence = gidnet_complex.finalize_reuse(test_sequence)
    assert isinstance(final_sequence, list)
    assert len(final_sequence) >= len(test_sequence)
    assert sorted(final_sequence) == sorted([[1, 0], [2, 3], [4]])

