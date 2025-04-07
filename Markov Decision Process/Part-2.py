import random
import numpy as np
import matplotlib.pyplot as plt
import csv
import copy

def csv_writer(csv_file_path, data, actions):
    # Define the fieldnames (column names) for the CSV file
    fieldnames = ["bx", "by", "cx", "cy", "t_bot", "actionx", "actiony"]
    # Write epoch data to the CSV file
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        # Write header (fieldnames)
        writer.writeheader()
        t_bot_data = {}
        # Write data rows
        for state in data:
            bot, crew = state
            t_bot_data["bx"] = bot[0]
            t_bot_data["by"] = bot[1]
            t_bot_data["cx"] = crew[0]
            t_bot_data["cy"] = crew[1]
            t_bot_data["t_bot"] = data[state]
            # print(bot[0], bot[1], actions)
            t_bot_data["actionx"] = actions[state][0]
            t_bot_data["actiony"] = actions[state][1]
            writer.writerow(t_bot_data)

def open_neighbours_crew(ship, D, row, col):
    r = []
    # Checking for neighbors and if they exist storing them in the r
    if row - 1 >= 0 and ship[row - 1][col] in ('1', 'T'):
        r.append((row - 1, col))
    if row + 1 < D and ship[row + 1][col] in ('1', 'T'):
        r.append((row + 1, col))
    if col - 1 >= 0 and ship[row][col - 1] in ('1', 'T'):
        r.append((row, col - 1))
    if col + 1 < D and ship[row][col + 1] in ('1', 'T'):
        r.append((row, col + 1))
    return r

def open_neighbours_bot(ship, D, row, col):
    r = []
    # Checking for neighbors and if they exist storing them in the r
    if row - 1 >= 0 and ship[row - 1][col] in ('1', 'T'):
        r.append((row - 1, col))
    if row + 1 < D and ship[row + 1][col] in ('1', 'T'):
        r.append((row + 1, col))
    if col - 1 >= 0 and ship[row][col - 1] in ('1', 'T'):
        r.append((row, col - 1))
    if col + 1 < D and ship[row][col + 1] in ('1', 'T'):
        r.append((row, col + 1))

    if row - 1 >= 0 and col + 1 < D and ship[row - 1][col + 1] in ('1', 'T'):
        r.append((row - 1, col + 1))
    if row + 1 < D and col + 1 < D and ship[row + 1][col + 1] in ('1', 'T'):
        r.append((row + 1, col + 1))
    if col - 1 >= 0 and row - 1 >= 0 and ship[row - 1][col - 1] in ('1', 'T'):
        r.append((row - 1, col - 1))
    if col - 1 >= 0 and row + 1 < D and ship[row + 1][col - 1] in ('1', 'T'):
        r.append((row + 1, col - 1))
    return r

# Function to get open cells in the ship
def get_open_cells(ship, D):
    result = []
    for i in range(D):
        for j in range(D):
            if ship[i][j] in ('1'):
                result.append((i, j))
    return result

# Function to get crew neighbours
def get_crew_neighbours(ship, D, row, col):
    result = []
    if row - 1 >= 0 and ship[row - 1][col] not in ('0', 'B'):
        result.append((row - 1, col))
    if row + 1 < D and ship[row + 1][col] not in ('0', 'B'):
        result.append((row + 1, col))
    if col - 1 >= 0 and ship[row][col - 1] not in ('0', 'B'):
        result.append((row, col - 1))
    if col + 1 < D and ship[row][col + 1] not in ('0', 'B'):
        result.append((row, col + 1))
    return result

# Function to get bot neighbours
def get_bot_neighbours(ship, D, row, col):
    result = []
    if row - 1 >= 0 and ship[row - 1][col] not in ('0', 'C'):
        result.append((row - 1, col))
    if row + 1 < D and ship[row + 1][col] not in ('0', 'C'):
        result.append((row + 1, col))
    if col - 1 >= 0 and ship[row][col - 1] not in ('0', 'C'):
        result.append((row, col - 1))
    if col + 1 < D and ship[row][col + 1] not in ('0', 'C'):
        result.append((row, col + 1))
    if row - 1 >= 0 and col - 1 >= 0 and ship[row - 1][col - 1] not in ('0', 'C'):
        result.append((row - 1, col - 1))
    if row + 1 < D and col + 1 < D and ship[row + 1][col + 1] not in ('0', 'C'):
        result.append((row + 1, col + 1))
    if col - 1 >= 0 and row + 1 < D and ship[row + 1][col - 1] not in ('0', 'C'):
        result.append((row + 1, col - 1))
    if col + 1 < D and row - 1 >= 0 and ship[row - 1][col + 1] not in ('0', 'C'):
        result.append((row - 1, col + 1))
    return result

def manhattan_distance(x1, y1, x2, y2):
    distance = abs(x2 - x1) + abs(y2 - y1)
    return distance

def compute_T_bot(ship_conf, D, teleport_pos):
    T_bot = {}
    for i in range(D):
        for j in range(D):
            for k in range(D):
                for l in range(D):
                    T_bot[(i, j), (k, l)] = 9999
                    if (k, l) == teleport_pos:
                        T_bot[(i, j), (k, l)] = 0

    status_changed = True
    steps = 0
    # Iterate until convergence
    while status_changed:
        # Create a copy of the value function for comparison
        status_changed = False

        ship = copy.deepcopy(ship_conf)  # --
        # Update the value function for each state using the Bellman update equation
        for bx in range(D):
            for by in range(D):
                for cx in range(D):
                    for cy in range(D):
                        if ((bx, by) != (cx, cy) and ship[bx][by] != '0' and ship[cx][cy] != '0'):
                            ship[bx][by] = 'B'  # --
                            ship[cx][cy] = 'C'  # --
                            if (cx, cy) == teleport_pos:
                                continue
                            current_value = T_bot[(bx, by), (cx, cy)]
                            bot_action = (bx, by)
                            min_value = 9999
                            actions = get_bot_neighbours(ship, D, bx, by)
                            ship[bx][by] = '1'
                            for bot_next in actions:
                                ship[bot_next[0]][bot_next[1]] = 'B'
                                neighbors = get_crew_neighbours(ship, D, cx, cy)
                                sum = 0
                                if (bot_next in neighbors):
                                    neighbors.remove(bot_next)

                                for crew_next in neighbors:
                                    sum += 1 / len(neighbors) * T_bot[bot_next, crew_next]
                                value = 1 + sum

                                if (value < min_value):
                                    min_value = value
                                    T_bot[(bx, by), (cx, cy)] = min_value

                                ship[bot_next[0]][bot_next[1]] = '1'
                            ship[bx][by] = ship_conf[bx][by]  # --
                            ship[cx][cy] = ship_conf[cx][cy]  # --
                            if abs(T_bot[(bx, by), (cx, cy)] - current_value) > 0.01:  # Check if significant change
                                status_changed = True
        steps += 1
    return T_bot

# For No bot simulation
def compute_optimal_policy(ship_conf, D, teleport_pos, T_bot_timestamp):
    T_bot = {}
    T_bot_action = {}
    for i in range(D):
        for j in range(D):
            for k in range(D):
                for l in range(D):
                    if (i,j)==(k,l):
                        continue
                    T_bot[(i, j), (k, l)] = 9999
                    T_bot_action[(i, j), (k, l)] = (i, j)
                    if (k, l) == teleport_pos:
                        T_bot[(i, j), (k, l)] = 0

    status_changed = True
    steps = 0
    # Iterate until convergence
    while status_changed:
        # Create a copy of the value function for comparison
        status_changed = False
        max_value = 0

        ship = copy.deepcopy(ship_conf)  # --
        # Update the value function for each state using the Bellman update equation
        for bx in range(D):
            for by in range(D):
                for cx in range(D):
                    for cy in range(D):

                        if ((bx, by) != (cx, cy) and ship[bx][by] != '0' and ship[cx][cy] != '0'):
                            ship[bx][by] = 'B'  # --
                            ship[cx][cy] = 'C'  # --
                            if (cx, cy) == teleport_pos:
                                continue
                            current_value = T_bot[(bx, by), (cx, cy)]
                            bot_action = (bx, by)
                            max_value = 9999
                            actions = get_bot_neighbours(ship, D, bx, by)

                            ship[bx][by] = '1'
                            for bot_next in actions:
                                ship[bot_next[0]][bot_next[1]] = 'B'

                                reward = T_bot_timestamp[(bot_next[0], bot_next[1]), (cx, cy)]
                                neighbors = get_crew_neighbours(ship, D, cx, cy)
                                neighbors.append((cx, cy))
                                sum = 0
                                for crew_next in neighbors:
                                    sum += 1 / len(neighbors) * T_bot[bot_next, crew_next]
                                value = reward + sum

                                if (value < max_value):
                                    max_value = value
                                    bot_action = bot_next
                                    T_bot[(bx, by), (cx, cy)] = max_value
                                    T_bot_action[(bx, by), (cx, cy)] = bot_action

                                ship[bot_next[0]][bot_next[1]] = '1'
                            ship[bx][by] = ship_conf[bx][by]  # --
                            ship[cx][cy] = ship_conf[cx][cy]  # --
                            if abs(T_bot[(bx, by), (cx, cy)] - current_value) > 0.01:  # Check if significant change
                                status_changed = True
        steps += 1
    return T_bot, T_bot_action

def main():
    ship = [['1', '1', '1', '1', '1', '1', '1', '1', '0', '1', '1'], ['1', '1', '1', '1', '0', '1', '1', '1', '1', '1', '1'], ['0', '0', '1', '1', '1', '1', '1', '1', '1', '1', '1'], ['0', '1', '1', '1', '1', '1', '1', '1', '0', '1', '1'], ['1', '1', '1', '1', '0', '1', '0', '1', '1', '1', '1'], ['1', '1', '1', '1', '1', 'T', '1', '1', '1', '1', '1'], ['1', '1', '1', '1', '0', '1', '0', '1', '1', '1', '1'], ['1', '1', '1', '0', '1', '1', '1', '0', '1', '1', '1'], ['0', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1'], ['1', '1', '1', '1', '1', '1', '1', '1', '1', '0', '1'], ['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1']]
    D = 11
    csv_file_path = "./t_bot_data2.csv"
    teleport_pos = (D // 2, D // 2)

    
    t_bot_timestamp = compute_T_bot(ship, D, teleport_pos)

    T_bot, actions = compute_optimal_policy(ship, D, teleport_pos, t_bot_timestamp)

    csv_writer(csv_file_path, T_bot, actions)

if __name__ == "__main__":
    main()