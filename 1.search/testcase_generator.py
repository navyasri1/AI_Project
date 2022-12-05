import os
import subprocess

algos = ['dfs', 'bfs', 'ucs', 'astar', 'mm0', 'mm']

layouts = []
flag = 0
for file in os.listdir('layouts'):
    if flag!=0:
        layouts.append(file.split('.')[0])
    flag = 1

results = {}
values = []
for algo in algos:
    temp = []
    for layout in layouts:
        count = 0
        if '_' not in layout:
            continue
        command_str = 'python3 pacman.py -l ' + layout + ' -p SearchAgent -a fn=' + algo

        if 'Corners' in layout:
            command_str = command_str + ',prob=CornersProblem'
            if algo != 'mm0':
                command_str = command_str + ',heuristic=cornersHeuristic'

        elif 'Search' in layout or 'Dotted' in layout:
            command_str = command_str + ',prob=FoodSearchProblem'
            if algo != 'mm0':
                command_str = command_str + ',heuristic=foodSearchHeuristic'

        elif algo == 'astar' or algo == 'mm':
            command_str = command_str + ',heuristic=manhattanHeuristic'

        if 'Classic' in layout:
            continue

        command_str = command_str + ' --frameTime 0.001'
        print(command_str)
        result = subprocess.run(command_str.split(' '), capture_output=True).stdout.decode()

        cost = int(str(result).split('Path found with total cost of')[1].split('in')[0])
        nodes_exp = int(result.split('Search nodes expanded:')[1].split('\n')[0])
        score = int(result.split('Score:')[1].split('\n')[0])

        results[(algo, layout)] = [cost, nodes_exp, score]
        temp.append(nodes_exp)
    values.append(temp)

final_list = []
for key in results.keys():
    print(key, results[key])
    final_list.append([key[0], key[1], results[key][0], results[key][1], results[key][2]])

import csv

header = ['Algo', 'environment', 'Path cost', 'Nodes expanded', 'Score']
header2 = ['T-test Results with MM','DFS','BFS','UCS','ASTAR','MM0']

from scipy import stats

t_value_list = []
p_value_list = []
'''for i in range(len(values)):
    temp_t = []
    temp_p = []
    for j in range(len(values)):
        t_value,p_value = stats.ttest_ind(values[j], values[i])
        t_value_list.append(t_value)
        p_value_list.append(p_value)
        temp_p.append(p_value/2)
        temp_t.append(t_value)
    t_value_list.append(temp_t)
    p_value_list.append(temp_p)'''

t_value_list.append('\t')
p_value_list.append('\t')

for i in range(len(values)):
    print(values[i])
for i in range(len(values)):
    t_value, p_value = stats.ttest_ind(values[i], values[-1])
    t_value_list.append(t_value)
    p_value_list.append(p_value/2)

print(t_value_list)
print(p_value_list)

with open('AI_project_output.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)

    # write the header
    writer.writerow(header)

    # write the data
    for line in final_list:
        writer.writerow(line)
    writer.writerow(header2)
    writer.writerow(t_value_list)
    writer.writerow(p_value_list)
