import numpy as np
import random
import math
from numpy import random


class worker:
    def __init__(self, num, ability_of_workers):
        # self.num = num
        self.mu = ability_of_workers[num]
        # self.mu = 1
        self.sigma = 0.01

    def person_work(self):
        worker_result = np.random.normal(self.mu, self.sigma, 1)
        return worker_result


def matrix_assign(result_of_system, relation_total, relation_n, workers, num_of_group):
    for i in range(num_of_group):
        for j in range(num_of_group):
            if i != j:
                relation_total[workers[i]][workers[j]] += result_of_system
                relation_n[workers[i]][workers[j]] += 1
    return relation_total, relation_n


def matrix_renewal(relation_n, relation_pre, relation_total, num_of_system):
    # Random和Epsilon放在最后统一计算，MAB是每次任务都要计算
    for i in range(num_of_system):
        for j in range(num_of_system):
            if relation_n[i][j] == 0:
                relation_pre[i][j] = round(relation_total[i][j], 3)
            else:
                relation_pre[i][j] = round(relation_total[i][j] / relation_n[i][j], 3)
    return relation_pre


# relation_real
def constant_produce(num_of_system):
    relation_real = random.random(size=(num_of_system, num_of_system))
    for i in range(num_of_system):
        relation_real[i] = np.round(relation_real[i], 3)
    relation_var = float(np.var(relation_real))

    # ability_of_workers
    ability_of_workers = abs(np.random.randn(num_of_system))

    for i in range(num_of_system):
        ability_of_workers[i] = np.round(ability_of_workers[i], 3)
    abilities_var = float(np.var(ability_of_workers))

    return relation_real, relation_var, ability_of_workers, abilities_var


def system_postprocess(relation_pre, workers, group_efficiency, times_total, num_of_group):
    contribution_of_ones = np.zeros(num_of_group)
    for i in range(num_of_group):
        contribution_of_ones[i] = sum(relation_pre[workers[i]])
    group_efficiency = group_efficiency / times_total
    return group_efficiency


def system_init(num_of_group, num_of_system):
    relation_total = np.zeros([num_of_system, num_of_system])
    relation_n = np.zeros([num_of_system, num_of_system])
    relation_pre = np.ones([num_of_system, num_of_system])
    workers = list_init(0, num_of_system - 1, num_of_group)
    person_efficiency = np.ones(num_of_system)  # 工人个体效能估计值，用于贪心算法，找到高效能工人的编号
    person_co = sum(relation_pre)
    group_efficiency = 0.0
    min_index = np.zeros(num_of_group)  # 群组里效能的倒序，用于找到低效能工人的位置并替换
    return relation_total, relation_n, relation_pre, workers, person_efficiency, person_co, group_efficiency, min_index


def normalization(x, num_of_system):  # 归一化函数
    for i in range(num_of_system):
        if max(x) - min(x) == 0:
            x = x
        else:
            x[i] = (x[i] - min(x)) / (max(x) - min(x))
    return x


def reselection_judge(workers, num_choice, i, num_of_group):
    reselection_flag = 0
    for j in range(num_of_group):
        if workers[j] == num_choice[i]:
            reselection_flag = 1
        else:
            reselection_flag = 0
    return reselection_flag


def epsilon_produce(times):
    if times == 0:
        epsilon = 1
    else:
        epsilon = 1 / math.sqrt(times)
    return epsilon


def random_unit(p):
    return random.random() < p


def list_init(start, stop, length):
    start, stop = (int(start), int(stop)) if start <= stop else (int(stop), int(start))
    length = int(abs(length)) if length else 0
    random_list = []
    for i in range(length):
        random_list.append(random.randint(start, stop))
    return random_list


def print_basic(relation_real, num_of_system, num_of_group, relation_var, abilities_var, ability_of_workers):
    relation_average = sum(sum(relation_real)) / (num_of_system * num_of_system)
    abilities_average = sum(ability_of_workers) / num_of_system
    print("\033[1;30;47mBasic constants:\033[0m")
    print("   The number of workers in a group is ", num_of_group)
    print("   The number of workers in the system is ", num_of_system)
    print("   The variance of the relation is ", relation_var)
    print("   The variance of the abilities is ", abilities_var)
    print("   The average of the relation is ", relation_average)
    print("   The average of the abilities is ", abilities_average)
    print("\033[1;30;47mThe results of system:\033[0m")


def print_result(group_efficiency):
    print("   The average result of the group is ", group_efficiency)


def print_result_of_all(sum_of_result, iterations):
    print("\033[1;32mThe final result of iterations is \033[0m", sum_of_result / iterations)


def print_time(start, end):
    run_time = end - start
    print("\033[1;32mThe time cost of iterations is \033[0m", run_time)


def group_work(workers, relation_real, num_of_group, ability_of_workers):
    group_result = 0
    person_result = np.zeros(num_of_group)

    for i in range(num_of_group):
        person_result[i] = worker(workers[i], ability_of_workers).person_work()
        for j in range(num_of_group):
            if i != j:
                group_result += person_result[i] * (relation_real[workers[i]][workers[j]])
    group_result = group_result / (num_of_group * (num_of_group - 1))
    return group_result, person_result


def result_produce(num_choice, pos_choice, workers, reselection_flag, num_of_group, relation_real, ability_of_workers):
    if reselection_flag == 1:
        result = group_work(workers, relation_real, num_of_group, ability_of_workers)
    else:
        for k in range(int(num_of_group * 0.2)):
            workers[int(pos_choice[k])] = int(num_choice[k])
        result = group_work(workers, relation_real, num_of_group, ability_of_workers)
    return result


def system_work_random(workers, num_of_group, num_of_system, relation_real, ability_of_workers):
    pos_choice = np.zeros(num_of_system)
    num_choice = np.zeros(num_of_system)
    reselection_flag = 0

    for i in range(int(num_of_group * 0.2)):
        pos_choice[i] = random.randint(0, (num_of_group - 1))
        num_choice[i] = random.randint(0, (num_of_system - 1))
        reselection_flag = reselection_judge(workers, num_choice, i, num_of_group)
    result = result_produce(num_choice, pos_choice, workers, reselection_flag, num_of_group, relation_real, ability_of_workers)
    return result[0]


def system_work_epsilon(workers, min_index, person_efficiency, epsilon, num_of_system, num_of_group, relation_real, ability_of_workers):
    pos_choice = np.zeros(num_of_system)
    num_choice = np.zeros(num_of_system)
    reselection_flag = 0

    for i in range(int(num_of_group * 0.2)):
        pos_choice[i] = min_index[i]
        if random.random() < epsilon:
            num_choice[i] = random.randint(0, (num_of_system - 1))
        else:
            num_choice[i] = person_efficiency[i]
        reselection_flag = reselection_judge(workers, num_choice, i, num_of_group)
    result = result_produce(num_choice, pos_choice, workers, reselection_flag, num_of_group, relation_real, ability_of_workers)
    return result


def system_work_mab(workers, min_index, person_efficiency, person_co, times, epsilon, num_of_system, num_of_group, relation_real, ability_of_workers):
    pos_choice = np.zeros(num_of_system)
    num_choice = np.zeros(num_of_system)
    regulatory_factor = float(1 / (1 + math.exp(-times)))  # 调节因子，当times的原点位置不加以修正的时候，达到了更好的效果
    reselection_flag = 0

    # 对person_efficiency和person_co进行归一化，并组合出一个“综合能力”用于贪心算法的选择
    person_co_n = normalization(person_co, num_of_system)
    person_co_n = np.array(person_co_n)
    person_efficiency_n = normalization(person_efficiency, num_of_system)
    person_efficiency_n = np.array(person_efficiency_n)
    person_comprehensive = (person_co_n * regulatory_factor) + (person_efficiency_n * (1 - regulatory_factor))
    person_comprehensive = sorted(person_comprehensive, reverse=True)

    for i in range(int(num_of_group * 0.5)):
        pos_choice[i] = min_index[i]    # 剔除通过个人能力，选取通过综合能力
        if random.random() < epsilon:
            num_choice[i] = random.randint(0, (num_of_system - 1))
        else:
            num_choice[i] = person_comprehensive[i]
        reselection_flag = reselection_judge(workers, num_choice, i, num_of_group)
    result = result_produce(num_choice, pos_choice, workers, reselection_flag, num_of_group, relation_real, ability_of_workers)
    return result
