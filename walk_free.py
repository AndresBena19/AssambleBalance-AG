import networkx as nx
from collections import defaultdict
import random
import re
import matplotlib.pyplot as plt
import xlsxwriter
import uuid


class Chromosome:

    def __init__(self, cycle_time, graph_successor, tasks, task_wight, graph_predecessors, chromosome_station=None,
                 chromosome=None):
        self.row = int(str(uuid.uuid4().int)[:5])
        self.dag_structure = nx.DiGraph(graph_successor)
        self.cycle_time = cycle_time
        self.graph_predecessors = graph_predecessors
        self.task_weight = task_wight
        self.task = tasks

        if not chromosome:
            self.chromosome = self.get_random_chromosome()
            self.chromosome_stations = self.get_stations()
        else:
            self.chromosome = chromosome
            self.chromosome_stations = chromosome_station

        self.chromosome_offstring = self.get_offstring()
        self.efficiency, self.total_efficiency, self.efficiency_time = self.get_efficiency()

    def chunks(self, l, n):
        for i in range(0, len(l), n):
            yield l[i:i + n]

    def __str__(self):
        return self.chromosome_offstring

    def get_random_chromosome(self):
        import copy
        possibilities = []
        temporal = copy.deepcopy(self.task)
        while temporal:
            task = random.sample(temporal, k=1)[0]
            if not (self.graph_predecessors.get(task)) or self.validate_precedence(self.graph_predecessors.get(task),
                                                                                   possibilities):
                possibilities.append(task)
                temporal.pop(temporal.index(task))
        return possibilities

    @staticmethod
    def validate_precedence(predecessor, memory_task):
        for task in predecessor:
            if task in memory_task:
                continue
            else:
                return False
        return True

    def get_stations(self):
        stations = defaultdict(list)
        count_station = 1
        for gene in self.chromosome:
            if stations:
                actual_wight = sum([nodes_weigh[gene_by_station] for gene_by_station in stations.get(count_station)])
            else:
                actual_wight = 0
            if random.choice([0, 1]):
                if (actual_wight + nodes_weigh[gene]) > self.cycle_time:
                    count_station += 1
                    stations[count_station].append(gene)
                else:
                    stations[count_station].append(gene)
            else:
                count_station += 1
                stations[count_station].append(gene)
        return list(enumerate(stations.values(), start=1))

    def get_efficiency(self):
        values_weigh = defaultdict(int)
        values_weigh_number = defaultdict(int)
        acum_efficiency = 0
        for station, tasks in self.chromosome_stations:
            time = sum([self.task_weight.get(gene) for gene in tasks])
            efficiency = time / self.cycle_time
            values_weigh[station] = efficiency
            values_weigh_number[station] = time
            acum_efficiency = acum_efficiency + efficiency
        return values_weigh, (acum_efficiency / len(self.chromosome_stations)), values_weigh_number

    def get_possible_candidate(self):
        possible_candidates = []
        for station, time in self.efficiency_time.items():
            if (station + 1) <= len(self.chromosome_stations):
                next_tasks = dict(self.chromosome_stations).get(station + 1)
                actual_task = dict(self.chromosome_stations).get(station)
                for task in next_tasks:
                    task_time = self.task_weight.get(task)
                    if (time + task_time) <= self.cycle_time:
                        validate_predeccesors = [predecessor in actual_task for predecessor in
                                                 self.dag_structure.predecessors(task)]
                        if all(validate_predeccesors):
                            possible_candidates.append([task, station, station + 1])
        return possible_candidates

    def get_offstring(self):
        chromosome_offstring = '{}'
        for station, tasks in self.chromosome_stations:
            stations_offstring = '{}'
            for task in tasks:
                stations_offstring = stations_offstring.format('{}{}{}'.format(station, task, '{}'))
            chromosome_offstring = chromosome_offstring.format('{}'.format(stations_offstring, '{}'))
        return chromosome_offstring[:-2]

    def adjust_stations(self, tasks_station_origin):
        chromosome_stations = dict(self.chromosome_stations)
        chromosome_stations.pop(tasks_station_origin)
        return list(enumerate(chromosome_stations.values(), start=1))

    def mutate(self, task, station_target, station_origin):
        tasks_station_origin = dict(self.chromosome_stations).get(station_origin)
        tasks_station_finish = dict(self.chromosome_stations).get(station_target)
        tasks_station_origin.remove(task)
        tasks_station_finish.append(task)
        if not tasks_station_origin:
            self.chromosome_stations = self.adjust_stations(station_origin)
        self.number_stations = len(self.chromosome_stations)
        self.chromosome_offstring = self.get_offstring()
        self.efficiency, self.total_efficiency, self.efficiency_time = self.get_efficiency()


class Generation:

    def __init__(self, graph_succesor,
                 task_string,
                 nodes_weigh,
                 generation,
                 number_experimental,
                 cycle_time,
                 mutation_probability,
                 graph_predeccesors):

        self.number_experimental = number_experimental
        self.cycle_time = cycle_time
        self.mutation_probability = mutation_probability
        self.generation = generation
        self.best_chromosome = 0
        self.task_string = task_string
        self.nodes_weigh = nodes_weigh
        self.graph_predeccesors = graph_predeccesors
        self.graph_succesor = graph_succesor
        # This is to generate a portable file with the trace of the steps took by the generic algorithm process
        self.workbook = xlsxwriter.Workbook('AGOlga.xlsx')

    def get_parents(self):
        import copy
        generation = copy.deepcopy(self.generation)

        mother = random.choices(
            population=generation,
            weights=[chromosome.total_efficiency for chromosome in generation],
            k=1
        )[0]

        generation.pop(generation.index(mother))
        father = random.choices(
            population=generation,
            weights=[chromosome.total_efficiency for chromosome in generation],
            k=1
        )[0]

        return [mother, father]

    def chunks(self, l, n):
        for i in range(0, len(l), n):
            yield l[i:i + n]

    def valide_existence(self, chromosome):
        if not chromosome:
            return True
        for task in task_string:
            match = re.search(r'\d+{}(\d+|$)'.format(task), chromosome)
            if not match:
                return True
        return False

    def get_descendence(self):
        unique_son_mf = ""
        unique_son_fb = ""
        validator = True
        while validator:
            mother, father = self.get_parents()

            node = random.choice(task_string)
            unique_son_mf = '{}{}'.format(
                re.search(r'(.*\d+{})(\d+|$)'.format(node), mother.chromosome_offstring).groups()[0],
                re.search(r'({})(.*)'.format(node), father.chromosome_offstring).groups()[1])

            unique_son_fb = '{}{}'.format(
                re.search(r'(.*\d+{})(\d+|$)'.format(node), father.chromosome_offstring).groups()[0],
                re.search(r'({})(.*)'.format(node), mother.chromosome_offstring).groups()[1])

            validator_mf = self.valide_existence(unique_son_mf) or self.validate_precedence(unique_son_mf)
            validator_fm = self.valide_existence(unique_son_fb) or self.validate_precedence(unique_son_fb)
            validator = validator_mf and validator_fm

        valid_mf = self.valide_existence(unique_son_mf) or self.validate_precedence(unique_son_mf)
        definitive_child = unique_son_mf if not valid_mf else unique_son_fb
        chromosome_stations, chromosome = self.adjust(definitive_child)
        return definitive_child, [mother, father], node, chromosome, chromosome_stations

    def crossover(self):
        childs = []
        parents_saver = []
        texts_mutation = []
        for _ in range(0, int(self.number_experimental / 2)):
            print(
                "==================================== SELECTING PARENTS {} ===================================".format(
                    _))
            unique_son, parents, node, chromosome, chromosome_stations = self.get_descendence()

            child_chromosome = Chromosome(self.cycle_time,
                                          self.graph_succesor,
                                          self.task_string,
                                          self.nodes_weigh,
                                          self.graph_predeccesors,
                                          chromosome_stations,
                                          chromosome)

            if random.random() <= self.mutation_probability:
                child_chromosome, text_mutation = self.mutation(child_chromosome)
                parents_saver.append([parents, node, text_mutation])
            else:
                parents_saver.append([parents, node, None])
            childs.append(child_chromosome)
        return childs, parents_saver, node, texts_mutation

    def validate_precedence(self, chromosome):
        if not chromosome:
            return True

        temporal_order = []
        station_validate = defaultdict(list)
        chromosome_stations = re.findall(r'\d+\D+', chromosome)

        for gene in chromosome_stations:
            station = re.search(r'\d+', gene).group()
            task = re.search(r'\D+', gene).group()
            station_validate[int(station)].append(task)
            temporal_order.append(task)
            predecessors = self.graph_predeccesors.get(task)
            for predecessor in predecessors:
                if predecessor not in temporal_order:
                    return True

        for stations, tasks in station_validate.items():
            total_time = sum([self.nodes_weigh.get(task) for task in tasks])
            if total_time > self.cycle_time:
                return True
        return False

    def adjust(self, chromosome):
        fixed_chromosome = defaultdict(list)
        chromosome_list = []
        station_registered = []
        chromosome_stations = re.findall(r'\d+\D+', chromosome)
        for gene in chromosome_stations:
            station = int(re.search(r'\d+', gene).group())
            task = re.search(r'\D+', gene).group()
            if task not in chromosome_list:
                if station_registered:
                    end_station = max(station_registered)
                else:
                    end_station = 0
                if station in station_registered and end_station > station:
                    one_step_before = fixed_chromosome.get(end_station)
                    station_weight = sum([self.nodes_weigh.get(task_before) for task_before in one_step_before])
                    actual_weight = self.nodes_weigh.get(task)

                    if station_weight + actual_weight <= self.cycle_time:
                        fixed_chromosome[end_station].append(task)
                    else:
                        fixed_chromosome[end_station + 1].append(task)
                else:
                    fixed_chromosome[station].append(task)
                station_registered.append(station)
                chromosome_list.extend(task)
        return list(enumerate(fixed_chromosome.values(), start=1)), chromosome_list

    def mutation(self, chromosome):
        candidates = chromosome.get_possible_candidate()
        text_mutation = None
        if candidates:
            print("====================================== WOWWW MUTATION ===========================================-")
            candidate = random.choice(candidates)
            text_mutation = 'Task {} change from the statiton {} to {}'.format(candidate[0], candidate[2], candidate[1])
            print(text_mutation)
            chromosome.mutate(*candidate)
            print("MUTATED : ", chromosome.chromosome_offstring)
        return chromosome, text_mutation

    def print_representation(self):
        import pprint
        print("{:<10}| {:<30} | {:<60} | {:<100}".format('row', 'Chromosome ', 'Eficciency', 'Total Efficency'))
        for chromosome in self.generation:
            print("{:130} |{:<30}  | {:<60} |{:<100}".format(str(chromosome.row),
                                                             '-'.join(list(chromosome.chromosome_offstring)),
                                                             str(dict(chromosome.efficiency)),
                                                             chromosome.total_efficiency))

    def generate_file(self, number, number_experimental, parents_p=None, node=None, generation=None):
        max_stations = max([len(value_e.chromosome_stations) for value_e in self.generation])
        worksheet = self.workbook.add_worksheet(name='Generation {}'.format(number))

        parent_vs_individuo = {}
        for _ in range(0, number_experimental + 1):
            worksheet.write('A{}'.format(_), 'INDIVIDUO {}'.format(_))

        row_temp = number_experimental + 3
        begin_write = row_temp
        worksheet.write('A{}'.format(row_temp), 'ESTACION')

        col = 1
        for _ in range(0, number_experimental):
            worksheet.write(row_temp - 1, col, 'INDIVIDUO {}'.format(_ + 1))
            col += 1

        for station_number in range(0, max_stations):
            worksheet.write(row_temp, 0, station_number + 1)
            row_temp += 1

        total_efecty_row = row_temp + 1
        self.total_efecty_row = total_efecty_row
        worksheet.write('A{}'.format(total_efecty_row), 'EFICIENCIA TOTAL')

        col_e = 1
        row = 0

        for chromosome in generation:
            temp_write = begin_write
            col = 0
            # Iterate over the data and write it out row by row.
            for value in re.findall(r'\d+\D+', chromosome.chromosome_offstring):
                worksheet.write(row, col + 1, value)
                col += 1
            row += 1

            for station, efecty in chromosome.efficiency.items():
                worksheet.write(temp_write, col_e, '{0:.2f}%'.format(float(efecty) * 100))
                temp_write += 1
            worksheet.write(total_efecty_row - 1, col_e, '{0:.2f}%'.format(float(chromosome.total_efficiency) * 100))
            col_e += 1

        parent_row = total_efecty_row + 3
        col_p = 0

        step_to_jump = 1
        for chromosome_e in generation:
            parent_vs_individuo[chromosome_e.row] = step_to_jump
            step_to_jump = step_to_jump + 1

        for parents, node, text_mutation in parents_p:
            worksheet.write(parent_row, col_p, 'Individuo {}'.format(parent_vs_individuo.get(parents[0].row)))
            worksheet.write(parent_row, col_p + 1, 'Individuo {}'.format(parent_vs_individuo.get(parents[1].row)))
            worksheet.write(parent_row, col_p + 2, node)
            worksheet.write(parent_row, col_p + 3, text_mutation)
            parent_row += 1


nodes_weigh = {'A': 70,
               'B': 80,
               'C': 40,
               'D': 20,
               'E': 40,
               'F': 30,
               'G': 50,
               'H': 50
               }

task_string = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
graph_successor = [('A', 'B'),
                   ('A', 'C'),
                   ('A', 'D'),
                   ('A', 'E'),
                   ('B', 'F'),
                   ('C', 'F'),
                   ('E', 'H'),
                   ('F', 'H'),
                   ('D', 'H'),
                   ('G', 'H')
                   ]

graph_predeccesors = {'A': [],
                      'B': ['A'],
                      'C': ['A'],
                      'D': ['A'],
                      'E': ['A'],
                      'F': ['B', 'C'],
                      'G': ['C'],
                      'H': ['F', 'G', 'D', 'E']}

if __name__ == '__main__':
    number_generations = 20
    number_experimental = 8
    cycle_time = 90
    mutation_probability = 0.05
    print(
        "*****************************************************************************************************************")
    print(
        "                                                   GENERATION INITIAL                                            ")
    print(
        "*****************************************************************************************************************")
    initial_poblation = [Chromosome(cycle_time, graph_successor, task_string, nodes_weigh, graph_predeccesors) for _ in
                         range(0, number_experimental)]
    generation = Generation(graph_successor,
                            task_string,
                            nodes_weigh,
                            initial_poblation,
                            number_experimental,
                            cycle_time,
                            mutation_probability,
                            graph_predeccesors)

    max_efficiency = [0, 0]
    graph_efficiency = []
    graph_efficiency.append(
        sorted([(chromosome.total_efficiency, chromosome.chromosome_stations, chromosome.chromosome_offstring) for
                chromosome in generation.generation], key=lambda x: x[0])[-1][0])
    generation.print_representation()
    decendence, parents, node, text_mutation = generation.crossover()

    generation.generate_file('INITIAL', generation.number_experimental, parents, node, generation.generation)
    parent_inhered = random.choices(generation.generation, k=int(number_experimental / 2))
    decendence.extend(parent_inhered)
    generation.generation = decendence
    for _ in range(0, number_generations):
        generation_candidates = generation.generation
        max_ = \
        sorted([(chromosome.total_efficiency, chromosome.chromosome_stations, chromosome.chromosome_offstring) for
                chromosome in generation.generation], key=lambda x: x[0])[-1]

        if max_[0] > max_efficiency[0]:
            max_efficiency = max_
        graph_efficiency.append(max_[0])

        decendence, parents, node, text_mutation = generation.crossover()
        generation.generate_file(_ + 1, generation.number_experimental, parents, node, generation.generation)

        parent_inhered = random.choices(generation_candidates, k=int(number_experimental / 2))
        decendence.extend(parent_inhered)
        generation.generation = decendence
        print(
            "********************************************************************************************************")
        print("                                                   GENERATION ", _,
              "                                           ")
        print(
            "********************************************************************************************************")
        generation.print_representation()

    generation.workbook.close()
    plt.plot(graph_efficiency)
    plt.title("Evolution")
    plt.show()
    print("===================================== WINNERR  BEST SOLUTION FOUND =====================================")
    print(max_efficiency)
