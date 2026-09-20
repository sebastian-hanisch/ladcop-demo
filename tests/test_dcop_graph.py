from cn_scenario import Instance, Job
from dcop_graph import objective_value, pairwise_cost, unary_cost


def _instance(jobs, n_agents=2, agent_start_positions=None, travel_time_per_unit=1.0):
    if agent_start_positions is None:
        agent_start_positions = tuple(0.0 for _ in range(n_agents))
    return Instance(
        n_jobs=len(jobs), n_agents=n_agents, jobs=jobs,
        agent_start_positions=agent_start_positions, travel_time_per_unit=travel_time_per_unit,
    )


def test_unary_cost_formula():
    jobs = (Job(index=0, position=5.0, duration=3.0),)
    instance = _instance(jobs, n_agents=2, agent_start_positions=(0.0, 10.0), travel_time_per_unit=2.0)
    assert unary_cost(instance, 0, 0) == 5.0 * 2.0 + 3.0
    assert unary_cost(instance, 0, 1) == 5.0 * 2.0 + 3.0  # |10-5|*2 + 3 = 13


def test_pairwise_cost_zero_when_different_agents_regardless_of_position():
    jobs = (Job(index=0, position=0.0, duration=1.0), Job(index=1, position=100.0, duration=1.0))
    instance = _instance(jobs, n_agents=2)
    assert pairwise_cost(instance, 0, 1, 0, 1) == 0.0
    assert pairwise_cost(instance, 0, 1, 1, 0) == 0.0


def test_pairwise_cost_is_distance_when_same_agent():
    jobs = (Job(index=0, position=2.0, duration=1.0), Job(index=1, position=9.0, duration=1.0))
    instance = _instance(jobs, n_agents=2, travel_time_per_unit=1.5)
    assert pairwise_cost(instance, 0, 1, 0, 0) == 7.0 * 1.5
    assert pairwise_cost(instance, 0, 1, 1, 1) == 7.0 * 1.5


def test_objective_value_matches_manual_summation():
    jobs = (
        Job(index=0, position=1.0, duration=2.0),
        Job(index=1, position=9.0, duration=3.0),
        Job(index=2, position=5.0, duration=1.0),
    )
    instance = _instance(jobs, n_agents=2, agent_start_positions=(0.0, 10.0))
    assignment = {0: 0, 1: 1, 2: 0}
    # Handrechnung (siehe test_dcop_dpop.py fuer die volle Herleitung):
    # unary(0,0)=3, unary(1,1)=4, unary(2,0)=6 -> 13
    # pairwise(0,1,0,1)=0 (verschiedene Agenten)
    # pairwise(0,2,0,0)=4 (gleicher Agent, |1-5|)
    # pairwise(1,2,1,0)=0 (verschiedene Agenten)
    expected = 3.0 + 4.0 + 6.0 + 0.0 + 4.0 + 0.0
    assert objective_value(instance, assignment) == expected
    assert expected == 17.0
