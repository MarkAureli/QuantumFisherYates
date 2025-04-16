# This code is part of qufy.
#
# Copyright (c) 2025 Lennart Binkowski
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from math import acos, sqrt
from numpy import binary_repr
from qiskit import QuantumCircuit


def init_perm_reg(num_elements: int) -> QuantumCircuit:
    """Construct quantum circuit for initializing a permutation register.

    :param num_elements: Number of elements on which represented permutation acts.
    :return: Initialization of the permutation register as quantum circuit.
    """
    num_rep_qubits: int = (num_elements - 1).bit_length()
    qc: QuantumCircuit = QuantumCircuit(num_elements * num_rep_qubits)
    for k in range(1, num_elements):
        for i in range(k.bit_length()):
            if (k >> i) & 1:
                qc.x(k * num_rep_qubits + i)
    return qc


def shukla_vedula(num_states: int) -> QuantumCircuit:
    """Construct quantum circuit generating uniform superposition of first computational basis states.

    This implementation is adapted from the construction presented in
    `Shukla and Vedula (2024) <https://dx.doi.org/10.1007/s11128-024-04258-4>` and extended to
    also cover powers of two correctly.

    :param num_states: Number of computational basis states to be superposed uniformly
    :return: Uniform superposition state preparation as quantum circuit.
    """
    qc: QuantumCircuit = QuantumCircuit(max(1, (num_states - 1).bit_length()))
    num_qubits: int = (num_states - 1).bit_length()
    if num_qubits != num_states.bit_length():  # num_states is a power of 2
        qc.h(slice(num_qubits))
        return qc
    bit_pos: list[int] = [index for (index, bit) in enumerate(binary_repr(num_states)[::-1]) if bit == '1']
    qc.x(bit_pos[1:num_states.bit_length()])
    cur_num_states: int = 1 << bit_pos[0]
    theta: float = -2 * acos(sqrt(cur_num_states / num_states))
    if bit_pos[0] > 0:  # num_states is even
        qc.h(slice(bit_pos[0]))
    qc.ry(theta, bit_pos[1])
    qc.ch(bit_pos[1], slice(bit_pos[0], bit_pos[1]), ctrl_state='0')
    for m in range(1, len(bit_pos) - 1):
        theta = -2 * acos(sqrt(2 ** bit_pos[m] / (num_states - cur_num_states)))
        qc.cry(theta, bit_pos[m], bit_pos[m + 1], ctrl_state='0')
        qc.ch(bit_pos[m + 1], slice(bit_pos[m], bit_pos[m + 1]), ctrl_state='0')
        cur_num_states += 1 << bit_pos[m]
    return qc


def one_hot_prep(num_states: int) -> QuantumCircuit:
    """Construct quantum circuit generating uniform superposition of all-zero state and states with Hamming weight one.

    This implementation directly follows the procedure described in
    `Barenco et al. (1997) <https://doi.org/10.1137/S0097539796302452>`.

    :param num_states: Number of computational basis states to be superposed uniformly.
    :return: Uniform superposition state preparation as quantum circuit.
    """
    qc: QuantumCircuit = QuantumCircuit(max(1, num_states - 1))
    if num_states == 1:
        return qc
    qc.ry(2 * acos(sqrt(1 / num_states)), 0)
    for i in range(1, num_states - 1):
        qc.cx(i, i - 1)
        qc.cry(2 * acos(sqrt(1 / (num_states - i))), i - 1, i)
        qc.cx(i, i - 1)
    return qc


def quantum_fisher_yates(num_elements: int, subreg_size: int = 0, rec_perm: bool = True, binary_control: bool = True,
                         disentangling: bool = True) -> QuantumCircuit:
    """Construct quantum circuit for applying the quantum Fisher-Yates shuffle.

    Depending on the input arguments, the quantum Fisher-Yates shuffle may permute the subregisters of some data
    register in superposition (the first num_elements * subreg_size qubits) and record the applied permutations in a
    permutation register (the subsequent num_elements * log2(num_elements) qubits), using an ancilla register (the
    remaining qubits) whose size depends on the chosen control method (binary control vs. one-hot control).

    :param num_elements: Number of elements/subregisters to be permuted.
    :param subreg_size: Number of qubits per subregister.
    :param rec_perm: Whether inverted word representation of permutations should be generated.
    :param binary_control: Whether swaps are controlled on binary encoded integers or on single qubits.
    :param disentangling: Whether data and permutation register should be disentangled from the ancilla register.
    :return: Quantum Fisher-Yates shuffle as quantum circuit.
    """
    if num_elements <= 0:
        raise ValueError("Number of elements/subregisters must be greater than 0.")
    if subreg_size < 0:
        raise ValueError("Subregister size must be greater than are equal to 0.")
    if subreg_size == 0 and not rec_perm:
        raise ValueError("At least one of the data and permutation register must be considered.")
    if not rec_perm and disentangling:
        raise ValueError("Disentangling strategy only works when recording the applied permutations.")

    num_rep_qubits: int = (num_elements - 1).bit_length()
    perm_offset: int = num_elements * subreg_size
    ancilla_offset: int = perm_offset + rec_perm * num_rep_qubits * num_elements
    num_ancilla_qubits: int = num_rep_qubits
    if binary_control:
        if not disentangling:
            num_ancilla_qubits += sum([i.bit_length() for i in range(1, num_elements - 1)])
    else:
        num_ancilla_qubits = num_elements - 1
        if not disentangling:
            num_ancilla_qubits += sum([i for i in range(1, num_elements - 1)])
    num_qubits: int = ancilla_offset + num_ancilla_qubits
    qc: QuantumCircuit = QuantumCircuit(num_qubits)
    if rec_perm:
        qc.compose(other=init_perm_reg(num_elements), qubits=slice(perm_offset, ancilla_offset), inplace=True)
    offset: int = 0
    for i in range(1, num_elements):
        num_ctrl: int = i.bit_length()
        if binary_control:
            qc.compose(other=shukla_vedula(i + 1),
                       qubits=slice(ancilla_offset + offset, ancilla_offset + offset + num_ctrl),
                       inplace=True)
        else:
            qc.compose(other=one_hot_prep(i + 1),
                       qubits=slice(ancilla_offset + offset, ancilla_offset + offset + i),
                       inplace=True)
        for j in range(0, i):
            for q in range(0, subreg_size):
                qc.cx(control_qubit=j * subreg_size + q, target_qubit=i * subreg_size + q, ctrl_state='1')
                if binary_control:
                    qc.mcx(control_qubits=[i * subreg_size + q] + list(range(ancilla_offset + offset,
                                                                             ancilla_offset + offset + num_ctrl)),
                           target_qubit=j * subreg_size + q,
                           ctrl_state=binary_repr(j, num_ctrl) + '1')
                else:
                    qc.ccx(control_qubit1=i * subreg_size + q, control_qubit2=ancilla_offset + offset + j,
                           target_qubit=j * subreg_size + q, ctrl_state='11')
                qc.cx(control_qubit=j * subreg_size + q, target_qubit=i * subreg_size + q, ctrl_state='1')
            if rec_perm:
                for q in range(0, num_rep_qubits):
                    qc.cx(control_qubit=perm_offset + j * num_rep_qubits + q,
                          target_qubit=perm_offset + i * num_rep_qubits + q, ctrl_state='1')
                    if binary_control:
                        qc.mcx(control_qubits=[perm_offset + i * num_rep_qubits + q] + list(
                            range(ancilla_offset + offset, ancilla_offset + offset + num_ctrl)),
                               target_qubit=perm_offset + j * num_rep_qubits + q,
                               ctrl_state=binary_repr(j, num_ctrl) + '1')
                    else:
                        qc.ccx(control_qubit1=perm_offset + i * num_rep_qubits + q,
                               control_qubit2=ancilla_offset + offset + j,
                               target_qubit=perm_offset + j * num_rep_qubits + q, ctrl_state='11')
                    qc.cx(control_qubit=perm_offset + j * num_rep_qubits + q,
                          target_qubit=perm_offset + i * num_rep_qubits + q, ctrl_state='1')
        if disentangling:
            for j in range(1, i + 1):
                if binary_control:
                    for k in range(j.bit_length()):
                        if (j >> k) & 1:
                            qc.mcx(control_qubits=list(
                                range(perm_offset + j * num_rep_qubits, perm_offset + j * num_rep_qubits + num_ctrl)),
                                target_qubit=ancilla_offset + offset + k, ctrl_state=i)
                else:
                    qc.mcx(control_qubits=list(
                        range(perm_offset + (j - 1) * num_qubits, perm_offset + (j - 1) * num_qubits + num_ctrl)),
                        target_qubit=ancilla_offset + offset + (j - 1), ctrl_state=i)
        else:
            offset += num_ctrl if binary_control else i
    return qc
