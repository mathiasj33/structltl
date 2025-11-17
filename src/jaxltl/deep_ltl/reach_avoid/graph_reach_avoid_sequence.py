from jaxltl.deep_ltl.reach_avoid.reach_avoid_sequence import (
    AssignmentSet,
    EpsilonType,
    ReachAvoidSequence,
)
from jaxltl.ltl.logic.boolean_parser import Node


class GraphReachAvoidSequence(ReachAvoidSequence):
    """A reach-avoid sequence with both assignment sets and boolean formula graphs."""

    def __init__(
        self,
        reach_avoid: list[tuple[AssignmentSet | EpsilonType, AssignmentSet]],
        reach_avoid_graphs: list[tuple[Node | EpsilonType | None, Node | None]],
        repeat_last: int = 0,
    ):
        """
        Params:
            reach_avoid: A list of pairs of reach and avoid assignments or epsilon.
            reach_avoid_graphs: A list of pairs of reach and avoid boolean formula graphs or epsilon.
            repeat_last: Number of times the last pair should be repeated.
        """
        super().__init__(reach_avoid, repeat_last)
        if len(reach_avoid) != len(reach_avoid_graphs):
            raise ValueError("Assignments and graphs lists must have the same length.")
        self.reach_avoid_graphs = tuple(reach_avoid_graphs)

    def __hash__(self):
        return hash((self.reach_avoid, self.reach_avoid_graphs, self.repeat_last))

    def __eq__(self, other):
        if not isinstance(other, GraphReachAvoidSequence):
            return False
        return (
            self.reach_avoid == other.reach_avoid
            and self.reach_avoid_graphs == other.reach_avoid_graphs
            and self.repeat_last == other.repeat_last
        )

    def __iter__(self):
        return iter(zip(self.reach_avoid, self.reach_avoid_graphs, strict=False))

    def __getitem__(self, item):
        if isinstance(item, slice):
            if item.start >= len(self.reach_avoid):
                if self.repeat_last <= 0:
                    return []
                return [(self.reach_avoid[-1], self.reach_avoid_graphs[-1])]
            return list(
                zip(self.reach_avoid[item], self.reach_avoid_graphs[item], strict=False)
            )
        if item >= len(self.reach_avoid):
            if self.repeat_last <= 0:
                raise IndexError
            return self.reach_avoid[-1], self.reach_avoid_graphs[-1]
        return self.reach_avoid[item], self.reach_avoid_graphs[item]
