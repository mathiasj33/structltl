from jaxltl.deep_ltl.reach_avoid.reach_avoid_sequence import (
    EPSILON,
    AssignmentSet,
    EpsilonType,
)
from jaxltl.ltl.logic.boolean_parser import Node


class GraphReachAvoidSequence:
    """A reach-avoid sequence with both assignment sets and boolean formula graphs."""

    def __init__(
        self,
        reach_avoid_assignments: list[
            tuple[AssignmentSet | EpsilonType, AssignmentSet]
        ],
        reach_avoid_graphs: list[tuple[Node | EpsilonType | None, Node | None]],
        repeat_last: int = 0,
    ):
        """
        Params:
            reach_avoid_assignments: A list of pairs of reach and avoid assignments or epsilon.
            reach_avoid_graphs: A list of pairs of reach and avoid boolean formula graphs or epsilon.
            repeat_last: Number of times the last pair should be repeated.
        """
        if len(reach_avoid_assignments) != len(reach_avoid_graphs):
            raise ValueError("Assignments and graphs lists must have the same length.")
        self.reach_avoid_assignments = tuple(reach_avoid_assignments)
        self.reach_avoid_graphs = tuple(reach_avoid_graphs)
        self.repeat_last = repeat_last

    def __hash__(self):
        return hash((self.reach_avoid_assignments, self.repeat_last))

    def __eq__(self, other):
        if not isinstance(other, GraphReachAvoidSequence):
            return False
        return (
            self.reach_avoid_assignments == other.reach_avoid_assignments
            and self.repeat_last == other.repeat_last
        )

    def __len__(self):
        return len(self.reach_avoid_assignments) + self.repeat_last

    def __repr__(self):
        # Representation from ReachAvoidSequence is more readable
        assign_repr = [
            f"{set(r) if not isinstance(r, EpsilonType) else EPSILON} ||| {set(a)}"
            for r, a in self.reach_avoid_assignments
        ]
        return f"{assign_repr} x {self.repeat_last}"
