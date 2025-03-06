import copy
import json
import random

from src.alns import State


### Parser to parse instance json file ###
# You should not change this class!
class Parser(object):
    def __init__(self, json_file):
        """initialize the parser, saves the data from the file into the following instance variables:
        -
        Args:
            json_file::str
                the path to the xml file
        """
        self.json_file = json_file
        with open(json_file, "r") as f:
            self.data = json.load(f)

        self.name = self.data["name"]
        self.Alpha = self.data["ALPHA"]
        self.T = self.data["T"]
        self.BMAX = self.data["BMax"]
        self.WMAX = self.data["WMax"]
        self.RMIN = self.data["RMin"]

        self.workers = [
            Worker(worker_data, self.T, self.BMAX, self.WMAX, self.RMIN)
            for worker_data in self.data["Workers"]
        ]
        self.tasks = [Task(task_data) for task_data in self.data["Tasks"]]


class Worker(object):
    def __init__(self, data, T, bmax, wmax, rmin):
        """Initialize the worker
        Attributes:
            id::int
                id of the worker
            skills::[skill]
                a list of skills of the worker
            available::{k: v}
                key is the day, value is the list of two elements,
                the first element in the value is the first available hour for that day,
                the second element in the value is the last available hour for that day, inclusively
            bmax::int
                maximum length constraint
            wmax::int
                maximum working hours
            rmin::int
                minimum rest time
            rate::int
                hourly rate
            tasks_assigned::[task]
                a list of task objects
            blocks::{k: v}
                key is the day where a block is assigned to this worker
                value is the list of two elements
                the first element is the hour of the start of the block
                the second element is the hour of the start of the block
                if a worker is not assigned any tasks for the day, the key is removed from the blocks dictionary:
                        Eg. del self.blocks[D]

            total_hours::int
                total working hours for the worker

        """
        self.id = data["w_id"]
        self.skills = data["skills"]
        self.T = T
        self.available = {int(k): v for k, v in data["available"].items()}
        # the constant number for f2 in the objective function
        self.bmin = 4
        self.bmax = bmax
        self.wmax = wmax
        self.rmin = rmin

        self.rate = data["rate"]
        self.tasks_assigned = []
        self.blocks = {}
        self.total_hours = 0

    def can_assign(self, task):
        # // Implement Code Here
        ## check skill set
        if task.skill not in self.skills:
            return False

        ## check available time slots
        if task.day not in self.available:
            # if not available that day, False
            return False
        
        if not (self.available[task.day][0] <= task.hour <= self.available[task.day][1]):
            # if available that day, but not that hour, False
            return False

        ## cannot do two tasks at the same time
        for assigned_task in self.tasks_assigned:
            if assigned_task.day == task.day and assigned_task.hour == task.hour:
                # if there is a task that has the same day and same hour, False
                return False

        ## If no other tasks assigned in the same day
        if task.day not in self.blocks:
            ## check if task.hour within possible hours for current day
            if not (self.available[task.day][0] <= task.hour <= self.available[task.day][1]):
                return False
            ## check if after total_hours < wmax after adding block
            if self.total_hours + 1 > self.wmax:
                return False
        else:
            ## If there are other tasks assigned in the same day
            block_start, block_end = self.blocks[task.day]
            ## if the task fits within the existing range
            if block_start <= task.hour <= block_end:
                # at this point it's assured that no task overlaps already
                return True
            ## otherwise check if new range after task is assigned is rmin feasible
            if task.hour < block_start: # task is earlier
                if block_start - task.hour > self.rmin:
                    return False
            elif task.hour > block_end: # task is later
                if task.hour - block_end > self.rmin:
                    return False
            # check if new range after task is assigned is within bmax and wmax
            new_block_start = min(block_start, task.hour)
            new_block_end = max(block_end, task.hour)
            if new_block_end - new_block_start + 1 > self.bmax:
                return False
            if self.total_hours + 1 > self.wmax:
                return False

        return True

    def assign_task(self, task):
        # // Implement Code Here
        # assume that the task can be assigned first before calling this function

        self.tasks_assigned.append(task)
        self.total_hours += 1

        if task.day in self.blocks:
            block_start, block_end = self.blocks[task.day]
            self.blocks[task.day] = [min(block_start, task.hour), max(block_end, task.hour)]
        else:
            self.blocks[task.day] = [task.hour, task.hour]


    def remove_task(self, task_id):
         # // Implement Code Here
        task_to_remove = None
        for task in self.tasks_assigned:
            if task.id == task_id:
                task_to_remove = task
                break

        if task_to_remove is None:
            # task doesn't exist
            return False

        self.tasks_assigned.remove(task_to_remove)
        self.total_hours -= 1

        # Update blocks
        if task_to_remove.day in self.blocks:
            block_start, block_end = self.blocks[task_to_remove.day]
            if block_start == task_to_remove.hour or block_end == task_to_remove.hour:
                # Recalculate the block for the day
                hours = [t.hour for t in self.tasks_assigned if t.day == task_to_remove.day]
                if hours:
                    self.blocks[task_to_remove.day] = [min(hours), max(hours)]
                else: # if no blocks left that day, delete the key
                    del self.blocks[task_to_remove.day]

        return True

    def get_objective(self):
        # this is basically just counting how many hours this worker works in
        # for the entirety of the period
        t = sum(x[1] - x[0] + 1 for x in self.blocks.values())
        return t * self.rate

    def __repr__(self):
        if len(self.blocks) == 0:
            return ""
        return "\n".join(
            [
                f"Worker {self.id}: Day {d} Hours {self.blocks[d]} Tasks {sorted([t.id for t in self.tasks_assigned if t.day == d])}"
                for d in sorted(self.blocks.keys())
            ]
        )


class Task(object):
    def __init__(self, data):
        self.id = data["t_id"]
        self.skill = data["skill"]
        self.day = data["day"]
        self.hour = data["hour"]


### PSP state class ###
# PSP state class. You could and should add your own helper functions to the class
# But please keep the rest untouched!
class PSP(State):
    def __init__(self, name, workers, tasks, alpha):
        """Initialize the PSP state
        Args:
            name::str
                name of the instance
            workers::[Worker]
                workers of the instance
            tasks::[Task]
                tasks of the instance
        """
        self.name = name
        self.workers = workers
        self.tasks = tasks
        self.Alpha = alpha
        # the tasks assigned to each worker, eg. [worker1.tasks_assigned, worker2.tasks_assigned, ..., workerN.tasks_assigned]
        self.solution = []
        self.unassigned = list(tasks)

    def random_initialize(self, seed=None):
        """
        Args:
            seed::int
                random seed
        Returns:
            objective::float
                objective value of the state
        """
        if seed is None:
            seed = 606

        random.seed(seed)
        # -----------------------------------------------------------
        # // Implement Code Here
        # // This should contain your construction heuristic for initial solution
        # // Use Worker class methods to check if assignment is valid
        # -----------------------------------------------------------
        
        # Sort tasks by skill requirement. Tasks with rarest available skills should be prioritized
        self.skill_availability = {}
        for worker in self.workers:
            for skill in worker.skills:
                if skill in self.skill_availability:
                    self.skill_availability[skill] += 1
                else:
                    self.skill_availability[skill] = 1

        self.M_big_number = len(self.workers) + 1 # number of appearance should not exceed this big number
        # aside from rarity, task should be ordered by earliest starting time
        self.tasks.sort(key=lambda task: (self.skill_availability.get(task.skill, self.M_big_number), task.day, task.hour))
        # Sort workers based on cheapest rates
        self.workers.sort(key=lambda worker: worker.rate)
        
        for task in self.tasks:
            # greedy construction, for each worker, if can be assigned, assign
            # logically: the first to accept will be the cheapest worker that can get this task
            for worker in self.workers:
                if worker.can_assign(task):
                    worker.assign_task(task)
                    self.unassigned.remove(task)
                    break

        # update solution
        self.solution = [worker.tasks_assigned for worker in self.workers]

    def copy(self):
        return copy.deepcopy(self)

    def objective(self):
        """Calculate the objective value of the state
        Return the total cost of each worker + unassigned cost
        """
        f1 = len(self.unassigned)
        f2 = sum(max(worker.get_objective(), 50) for worker in self.workers if worker.get_objective() > 0)
        return self.Alpha * f1 + f2
