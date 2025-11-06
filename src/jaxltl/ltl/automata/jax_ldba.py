from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from jaxltl.environments.environment import Environment
from jaxltl.environments.wrappers.wrapper import EnvWrapper
from jaxltl.ltl.automata.ldba import LDBA


class JaxLDBA(NamedTuple):
    """Jax representation of a Limit Deterministic Büchi Automaton (LDBA)."""

    num_states: jax.Array  # int32
    initial_state: jax.Array  # int32
    transitions: jax.Array  # shape: (num_states, num_assignments) -> int32
    accepting: jax.Array  # shape: (num_states, num_assignments) -> bool
    finite: jax.Array  # bool

    def get_next_state(
        self, state: jax.Array, assignment_index: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """Get the next LDBA state given the current state and assignment.

        Args:
            state: Current LDBA state (int32).
            assignment_index: Index of the assignment (int32).

        Returns:
            next_state: Next LDBA state (int32).
            is_accepting: Whether the transition is accepting (bool)."""
        next_state = self.transitions[state, assignment_index]
        accepting = self.accepting[state, assignment_index]
        return next_state, accepting

    @classmethod
    def from_ldba(cls, ldba: LDBA, env: Environment | EnvWrapper) -> "JaxLDBA":
        """Convert an LDBA to its Jax representation."""

        if ldba.initial_state is None:
            raise ValueError("LDBA not initialized.")

        num_states = jax.numpy.array(ldba.num_states, dtype=jax.numpy.int32)
        initial_state = jax.numpy.array(ldba.initial_state, dtype=jax.numpy.int32)

        num_assignments = len(env.assignments)
        # Use numpy arrays and convert at the end for efficiency
        transitions = -np.ones((ldba.num_states, num_assignments), dtype=np.int32)
        accepting = np.zeros((ldba.num_states, num_assignments), dtype=bool)

        for state in range(ldba.num_states):
            for transition in ldba.state_to_transitions[state]:
                for assignment in transition.valid_assignments:
                    index = env.assignments.index(assignment)
                    transitions[state, index] = transition.target
                    if transition.accepting:
                        accepting[state, index] = True

        assert np.all(transitions >= 0), "Incomplete LDBA transitions."

        return cls(
            num_states=num_states,
            initial_state=initial_state,
            transitions=jnp.array(transitions),
            accepting=jnp.array(accepting),
            finite=jnp.array(ldba.is_finite_specification()),
        )
