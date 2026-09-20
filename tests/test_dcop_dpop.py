from cn_scenario import Instance, Job, generate_instance
from dcop_bruteforce import solve_bruteforce_dcop
from dcop_dpop import solve_dcop, util_table_size


def _instance(jobs, n_agents=2, agent_start_positions=None, travel_time_per_unit=1.0):
    if agent_start_positions is None:
        agent_start_positions = tuple(0.0 for _ in range(n_agents))
    return Instance(
        n_jobs=len(jobs), n_agents=n_agents, jobs=jobs,
        agent_start_positions=agent_start_positions, travel_time_per_unit=travel_time_per_unit,
    )


def _hand_instance():
    # Siehe Plan-Datei fuer die volle, unabhaengig zweimal nachgerechnete Herleitung.
    jobs = (
        Job(index=0, position=1.0, duration=2.0),
        Job(index=1, position=9.0, duration=3.0),
        Job(index=2, position=5.0, duration=1.0),
    )
    return _instance(jobs, n_agents=2, agent_start_positions=(0.0, 10.0))


def test_hand_computed_util_tables_and_assignment():
    instance = _hand_instance()
    result = solve_dcop(instance)

    job0, job1, job2 = result.util_tables
    assert job0.job_index == 0 and job0.separator == ()
    assert job0.entries == {(): (17.0, 0)}

    assert job1.job_index == 1 and job1.separator == (0,)
    assert job1.entries == {(0,): (14.0, 1), (1,): (18.0, 1)}

    assert job2.job_index == 2 and job2.separator == (0, 1)
    # (0,1) und (1,0) sind echte Unentschieden (je zwei x2-Optionen kosten 10.0) -
    # loesen sich zur niedrigeren Agenten-ID (0) auf.
    assert job2.entries == {
        (0, 0): (6.0, 1), (0, 1): (10.0, 0), (1, 0): (10.0, 0), (1, 1): (6.0, 0),
    }

    assert result.assignment == {0: 0, 1: 1, 2: 0}
    assert result.total_cost == 17.0


def test_dpop_matches_bruteforce_on_own_objective_across_random_instances():
    for n_jobs in range(2, 9):
        for n_agents in (2, 3, 4):
            for seed in range(6):
                instance = generate_instance(n_jobs, n_agents, 0.5, 1.0, seed)
                dpop = solve_dcop(instance)
                bf_cost, bf_assignment = solve_bruteforce_dcop(instance)
                assert abs(dpop.total_cost - bf_cost) < 1e-6, f"n_jobs={n_jobs} n_agents={n_agents} seed={seed}"
                assert dpop.assignment == bf_assignment, f"n_jobs={n_jobs} n_agents={n_agents} seed={seed}"


def test_util_table_size_grows_as_predicted():
    for n_jobs in range(1, 8):
        for n_agents in (2, 3, 4):
            instance = generate_instance(n_jobs, n_agents, 0.3, 1.0, 0)
            result = solve_dcop(instance)
            for job_index, table in enumerate(result.util_tables):
                expected = n_agents ** job_index
                assert len(table.entries) == expected, f"n_jobs={n_jobs} n_agents={n_agents} job={job_index}"
                assert util_table_size(n_agents, job_index) == expected


def test_solve_dcop_is_deterministic():
    instance = generate_instance(7, 3, 0.5, 1.2, 3)
    first = solve_dcop(instance)
    second = solve_dcop(instance)
    assert first.assignment == second.assignment
    assert first.total_cost == second.total_cost
    for t1, t2 in zip(first.util_tables, second.util_tables):
        assert t1.entries == t2.entries


def test_single_job_edge_case():
    jobs = (Job(index=0, position=3.0, duration=1.0),)
    instance = _instance(jobs, n_agents=3, agent_start_positions=(0.0, 1.0, 10.0))
    result = solve_dcop(instance)
    assert len(result.util_tables) == 1
    assert result.util_tables[0].separator == ()
    # Agent 1 (Position 1.0) ist am naechsten -> guenstigste Unaerkosten
    assert result.assignment == {0: 1}
