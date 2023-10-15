# mcspy

A Python Library for Mobile Crowdsensing Problems,
with random, Epsilon-Greedy and New_MAB methods to build a MCS system and its group.

## Installation

    pip install mcspy

## Author contact

Han Yaohui

1407210129@csu.edu.cn

## PyPI Address

https://pypi.org/project/mcspy/

## Source Code

see https://github.com/Han-0107/mcspy

## Script example

https://github.com/Han-0107/New_MAB_in_MCS

## Functions

### Basic Functions

    1. constant_produce(num_of_system)
    2. system_postprocess(relation_pre, workers, group_efficiency, times_total, num_of_group)
    3. system_init(num_of_group, num_of_system)
    4. normalization(x, num_of_system)
    5. reselection_judge(workers, num_choice, i, num_of_group)
    6. epsilon_produce(times)
    7. random_unit(p)
    8. list_init(start, stop, length)
    9. result_produce(num_choice, pos_choice, workers, reselection_flag, num_of_group, relation_real, ability_of_workers)

### Matrix Functions

    1. matrix_assign(result_of_system, relation_total, relation_n, workers, num_of_group)
    2. matrix_renewal(relation_n, relation_pre, relation_total, num_of_system)

### Printing Functions

    1. print_basic(relation_real, num_of_system, num_of_group, relation_var, abilities_var, ability_of_workers)
    2. print_result(group_efficiency)
    3. print_result_of_all(sum_of_result, iterations)
    4. print_time(start, end)

### Working Functions

    1. group_work(workers, relation_real, num_of_group, ability_of_workers)
    2. system_work_random(workers, num_of_group, num_of_system, relation_real, ability_of_workers)
    3. system_work_epsilon(workers, min_index, person_efficiency, epsilon, num_of_system, num_of_group, relation_real, ability_of_workers)
    4. system_work_mab(workers, min_index, person_efficiency, person_co, times, epsilon, num_of_system, num_of_group, relation_real, ability_of_workers)