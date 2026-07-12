from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass
class SplitResult:
    """Result of decoding a batch of giant tours.

    costs has shape (batch, pomo).  predecessors, when requested, has shape
    (batch, pomo, problem_size + 1) and stores the previous auxiliary-graph
    vertex of each shortest-path label.
    """

    costs: torch.Tensor
    predecessors: Optional[torch.Tensor] = None


def split_giant_tours(
    depot_xy: torch.Tensor,
    node_xy: torch.Tensor,
    node_demand: torch.Tensor,
    giant_tours: torch.Tensor,
    capacity: float = 1.0,
    return_predecessors: bool = False,
    epsilon: float = 1e-6,
) -> SplitResult:
    """Optimally split fixed customer permutations under a hard capacity.

    This is the classical Bellman shortest-path Split algorithm, vectorized
    over the batch and POMO dimensions.  The customer order is never changed.

    Args:
        depot_xy: (batch, 1, 2) depot coordinates.
        node_xy: (batch, n, 2) customer coordinates.
        node_demand: (batch, n) customer demands.
        giant_tours: (batch, pomo, n) permutations using customer IDs 1..n.
        capacity: Vehicle capacity in the same units as node_demand.
        return_predecessors: Keep shortest-path predecessors for route recovery.
        epsilon: Numerical tolerance for capacity feasibility.
    """

    _validate_inputs(depot_xy, node_xy, node_demand, giant_tours, capacity)

    batch_size, problem_size, _ = node_xy.shape
    pomo_size = giant_tours.size(1)
    route_count = batch_size * pomo_size

    # Convert public customer IDs 1..n to tensor indices 0..n-1.
    gather_index = (giant_tours - 1).unsqueeze(-1).expand(-1, -1, -1, 2)
    expanded_xy = node_xy[:, None, :, :].expand(-1, pomo_size, -1, -1)
    ordered_xy = expanded_xy.gather(2, gather_index).reshape(route_count, problem_size, 2)

    demand_index = giant_tours - 1
    expanded_demand = node_demand[:, None, :].expand(-1, pomo_size, -1)
    ordered_demand = expanded_demand.gather(2, demand_index).reshape(route_count, problem_size)

    depot = depot_xy[:, None, :, :].expand(-1, pomo_size, -1, -1).reshape(route_count, 1, 2)
    depot_distance = (ordered_xy - depot).pow(2).sum(-1).sqrt()

    if problem_size > 1:
        consecutive_distance = (ordered_xy[:, 1:] - ordered_xy[:, :-1]).pow(2).sum(-1).sqrt()
        edge_prefix = torch.cat(
            (torch.zeros(route_count, 1, device=node_xy.device, dtype=node_xy.dtype),
             consecutive_distance.cumsum(dim=1)),
            dim=1,
        )
    else:
        edge_prefix = torch.zeros(route_count, 1, device=node_xy.device, dtype=node_xy.dtype)

    demand_prefix = torch.cat(
        (torch.zeros(route_count, 1, device=node_demand.device, dtype=node_demand.dtype),
         ordered_demand.cumsum(dim=1)),
        dim=1,
    )

    inf = torch.tensor(float("inf"), device=node_xy.device, dtype=node_xy.dtype)
    potential = torch.full(
        (route_count, problem_size + 1), inf, device=node_xy.device, dtype=node_xy.dtype
    )
    potential[:, 0] = 0
    predecessors = None
    if return_predecessors:
        predecessors = torch.full(
            (route_count, problem_size + 1), -1, device=node_xy.device, dtype=torch.long
        )

    # Auxiliary edge (start, end) represents depot -> pi[start:end] -> depot.
    for end in range(1, problem_size + 1):
        segment_load = demand_prefix[:, end, None] - demand_prefix[:, :end]
        internal_distance = edge_prefix[:, end - 1, None] - edge_prefix[:, :end]
        segment_cost = (
            depot_distance[:, :end]
            + internal_distance
            + depot_distance[:, end - 1, None]
        )

        candidate = potential[:, :end] + segment_cost
        candidate = candidate.masked_fill(segment_load > capacity + epsilon, inf)
        best_cost, best_start = candidate.min(dim=1)
        potential[:, end] = best_cost
        if predecessors is not None:
            predecessors[:, end] = best_start

    final_cost = potential[:, problem_size]
    if not torch.isfinite(final_cost).all():
        raise ValueError(
            "Split found no capacity-feasible solution. Check that every customer demand "
            "is no larger than the vehicle capacity."
        )

    pred_out = None
    if predecessors is not None:
        pred_out = predecessors.reshape(batch_size, pomo_size, problem_size + 1)
    return SplitResult(final_cost.reshape(batch_size, pomo_size), pred_out)


def reconstruct_routes(
    giant_tour: torch.Tensor,
    predecessors: torch.Tensor,
) -> List[List[int]]:
    """Recover one decoded solution as customer-ID routes from Split labels.

    Args:
        giant_tour: One permutation with shape (n,), using IDs 1..n.
        predecessors: One predecessor vector with shape (n+1,).
    """

    if giant_tour.dim() != 1 or predecessors.dim() != 1:
        raise ValueError("giant_tour and predecessors must both be rank-1 tensors")
    if predecessors.numel() != giant_tour.numel() + 1:
        raise ValueError("predecessors must contain n+1 auxiliary-graph entries")

    tour = giant_tour.detach().cpu().tolist()
    pred = predecessors.detach().cpu().tolist()
    routes = []
    end = len(tour)
    while end > 0:
        begin = pred[end]
        if begin < 0 or begin >= end:
            raise ValueError("invalid Split predecessor chain")
        routes.append(tour[begin:end])
        end = begin
    routes.reverse()
    return routes


def _validate_inputs(
    depot_xy: torch.Tensor,
    node_xy: torch.Tensor,
    node_demand: torch.Tensor,
    giant_tours: torch.Tensor,
    capacity: float,
) -> None:
    if capacity <= 0:
        raise ValueError("capacity must be positive")
    if depot_xy.dim() != 3 or depot_xy.shape[1:] != (1, 2):
        raise ValueError("depot_xy must have shape (batch, 1, 2)")
    if node_xy.dim() != 3 or node_xy.size(2) != 2:
        raise ValueError("node_xy must have shape (batch, n, 2)")
    if node_demand.shape != node_xy.shape[:2]:
        raise ValueError("node_demand must have shape (batch, n)")
    if giant_tours.dim() != 3:
        raise ValueError("giant_tours must have shape (batch, pomo, n)")
    if giant_tours.size(0) != node_xy.size(0) or giant_tours.size(2) != node_xy.size(1):
        raise ValueError("giant_tours batch/problem dimensions do not match node_xy")
    if depot_xy.size(0) != node_xy.size(0):
        raise ValueError("depot_xy and node_xy batch dimensions do not match")

    n = node_xy.size(1)
    if giant_tours.numel() > 0:
        if giant_tours.min().item() < 1 or giant_tours.max().item() > n:
            raise ValueError("giant_tours must use customer IDs in 1..n")
        sorted_tours = giant_tours.sort(dim=2).values
        expected = torch.arange(1, n + 1, device=giant_tours.device, dtype=giant_tours.dtype)
        if not torch.equal(sorted_tours, expected.view(1, 1, n).expand_as(sorted_tours)):
            raise ValueError("each giant tour must be a permutation of all customer IDs")
    if (node_demand < 0).any().item():
        raise ValueError("customer demands must be non-negative")
    if (node_demand > capacity + 1e-6).any().item():
        raise ValueError("a customer demand exceeds vehicle capacity")
