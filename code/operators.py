import copy
import random

import numpy as np
from psp import PSP


### Destroy operators ###
def destroy_1(current: PSP, random_state):
    """Random Tasks Removal"""
    post_destroy = current.copy()
    num_tasks_to_remove = random_state.randint(1, 6)
    assigned_tasks = [task for task in post_destroy.tasks if task not in post_destroy.unassigned]
    tasks_to_remove = random_state.choice(assigned_tasks, num_tasks_to_remove)
    for task in tasks_to_remove:
        # find the workers who worked on the task
        for worker in post_destroy.workers:
            if task in worker.tasks_assigned:
                worker.remove_task(task.id)
                post_destroy.unassigned.append(task)
                break
    return post_destroy


def destroy_2(current: PSP, random_state):
    """Overworked Workers' Tasks Removal"""
    post_destroy = current.copy()
    num_workers_to_free = random_state.randint(1, 6)
    num_tasks_to_free = random_state.randint(1, 6)

    for _ in range(num_workers_to_free):
        for worker in post_destroy.workers:
            while (worker.tasks_assigned) and (num_tasks_to_free > 0):
                task = worker.tasks_assigned[0]
                worker.remove_task(task.id)
                post_destroy.unassigned.append(task)
                num_tasks_to_free -= 1
    return post_destroy


def destroy_3(current: PSP, random_state):
    """Most Expensive rate Worker's Tasks Removal"""
    post_destroy = current.copy()
    num_tasks_to_remove = random_state.randint(1, 6)
    task_costs = []
    for worker in post_destroy.workers:
        for task in worker.tasks_assigned:
            task_costs.append((task, worker.rate))

    task_costs.sort(key=lambda x: x[1], reverse=True)
    # remove the first num_tasks_to_remove of the most expensive
    tasks_to_remove = [task for task, _ in task_costs[:num_tasks_to_remove]]
    for task in tasks_to_remove:
        for worker in post_destroy.workers:
            if task in worker.tasks_assigned:
                worker.remove_task(task.id)
                post_destroy.unassigned.append(task)
                break
    return post_destroy


### Repair operators ###
def repair_1(destroyed: PSP, random_state):
    """Cheapest Task Assignment. Try to assign as many as possible"""
    post_repair = destroyed.copy()
    post_repair.workers.sort(key=lambda worker: worker.rate)
    prev_len = np.inf
    current_len = len(post_repair.unassigned)
    while prev_len > current_len:
        for task in post_repair.unassigned:
            for worker in post_repair.workers:
                if worker.can_assign(task):
                    worker.assign_task(task)
                    post_repair.unassigned.remove(task)
                    break

        prev_len = current_len
        current_len = len(post_repair.unassigned)
        # iteration will stop once the list of unassigned tasks don't change
    return post_repair


def repair_2(destroyed: PSP, random_state):
    """Freeest Worker Task Assignment. Try to assign as many as possible"""
    post_repair = destroyed.copy()
    post_repair.workers.sort(key=lambda worker: worker.total_hours)
    prev_len = np.inf
    current_len = len(post_repair.unassigned)
    while prev_len > current_len:
        for task in post_repair.unassigned:
            for worker in post_repair.workers:
                if worker.can_assign(task):
                    worker.assign_task(task)
                    post_repair.unassigned.remove(task)
                    break

        prev_len = current_len
        current_len = len(post_repair.unassigned)
        # iteration will stop once the list of unassigned tasks don't change
    return post_repair

def repair_3(destroyed: PSP, random_state):
    """Most Cost-Effective Task Assignment"""
    post_repair = destroyed.copy()
    prev_len = np.inf
    current_len = len(post_repair.unassigned)
    while prev_len > current_len:
        for task in post_repair.unassigned:
            current_minimum_cost_difference = np.inf
            current_minimum_to_assign_to = None
            for worker in post_repair.workers:
                if worker.can_assign(task):
                    # make a simulated worker that if assigned, count the objective function
                    current_objective = worker.get_objective()
                    current_cost = max(current_objective, 50) if current_objective > 0 else 0

                    simulated_worker = copy.deepcopy(worker)
                    simulated_worker.assign_task(task)

                    simulated_objective = worker.get_objective()
                    simulated_cost = max(simulated_objective, 50) if simulated_objective > 0 else 0

                    cost_difference = simulated_cost - current_cost

                    if cost_difference < current_minimum_cost_difference:
                        current_minimum_to_assign_to = worker
                        current_minimum_cost_difference = cost_difference
            
            # if the best personnel is found, assign
            if current_minimum_to_assign_to is not None:
                current_minimum_to_assign_to.assign_task(task)
                post_repair.unassigned.remove(task)

        prev_len = current_len
        current_len = len(post_repair.unassigned)
        # iteration will stop once the list of unassigned tasks don't change
    return post_repair
