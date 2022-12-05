import os

algorithms = ['bfs','dfs','ucs','astar','mm']

layouts = ['tinyMaze','smallMaze','mediumMaze','bigMaze','openMaze','contoursMaze','testMaze','tinySearch','mediumSearch','bigSearch','trickySearch']

f = open('generated_commands.txt', 'w+')

for layout in layouts:
    for algorithm in algorithms:
        cmd = 'py pacman.py -l ' + str(layout) + ' -p SearchAgent -a fn=' + str(algorithm)
        if algorithm == 'astar':
            cmd+=',heuristic=manhattanHeuristic'
        elif algorithm == 'mm':
            f.write(cmd+'\n')
            cmd+=',heuristic=manhattanHeuristic'
            f.write(cmd+'\n')
        else:
            f.write(cmd+'\n')

f.flush()
f.close()

f = open('generated_environment_commands.txt','w+')

layout_dirs = os.listdir('./layouts')
layout_dirs = [layout_dir for layout_dir in layout_dirs if '.lay' not in layout_dir]

for layout_dir in layout_dirs:
    layouts = os.listdir('./layouts/' + layout_dir)
    for layout in layouts:
        layout = layout.rstrip('.lay')
        for algorithm in algorithms:
            cmd = 'py pacman.py -l ' + str(layout_dir) + '/' + str(layout) + ' -p SearchAgent -a fn=' + str(algorithm)
            if 'tricky' in layout:
                if algorithm =='mm':
                    cmd+=',prob=FoodSearchProblem'
                    f.write(cmd+'\n')
                    cmd+=',heuristic=foodSearchHeuristic'
                    f.write(cmd+'\n')
                else:
                    cmd+=',prob=FoodSearchProblem,heuristic=foodSearchHeuristic'
                    f.write(cmd+'\n')
            elif algorithm == 'astar':
                cmd+=',heuristic=manhattanHeuristic'
                f.write(cmd+'\n')
            elif algorithm == 'mm':
                if 'Corners' in layout:
                    cmd+=',prob=CornersProblem'
                    f.write(cmd+'\n')
                    cmd+=',heuristic=cornersHeuristic'
                    f.write(cmd+'\n')
                elif 'Maze' in layout:
                    f.write(cmd+'\n')
                    cmd+=',heuristic=manhattanHeuristic'
                    f.write(cmd+'\n')
            else:
                f.write(cmd+'\n')

f.flush()
f.close()