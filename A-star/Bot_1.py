from itertools import product
import random
import numpy as np
import matplotlib.pyplot as plt
from queue import PriorityQueue
from numpy import inf
class Cell:
    def __init__(self):
        self.parent_i = 0  # Parent cell's row index
        self.parent_j = 0  # Parent cell's column index
        self.f = int('inf')  # Total cost of the cell (g + h)
        self.g = int('inf')  # Cost from start to this cell
        self.h = 0

def print_ship(ship, D):
    for row in ship:
        print(" ".join(row))
    print()

def visualize_ship(ship, G, D):
    # Create a copy of the ship and mark visited cells
    visual_matrix = np.zeros((D, D, 3), dtype=np.uint8)
    for i in range(D):
        for j in range(D):
            if ship[i][j] == '0':
                visual_matrix[i, j] = [0, 0, 0]  # Black for 'X'
            elif ship[i][j] in ['1','C']:
                visual_matrix[i, j] = [255, 255, 255]  # White for '0'
            elif ship[i][j] == 'A':
                visual_matrix[i, j] = [255, 0, 0]  # Red for 'A'
            elif ship[i][j] == 'B':
                visual_matrix[i, j] = [0, 0, 255]  # Blue for 'B'
            
    plt.plot(G[1], G[0], marker='$C$', markersize=10, color='green', markeredgecolor='green') #Green C for Captain
    plt.imshow(visual_matrix, interpolation='nearest')
    plt.title('Matrix Visualization')
    plt.show(block=False)
    
    

def get_open_neighbours_count(ship, D, row, col):
    count = 0
    # Check all neighbors and getting the count 
    if row - 1 >= 0 and ship[row - 1][col] == '1':
        count += 1
    if row + 1 < D and ship[row + 1][col] == '1':
        count += 1
    if col - 1 >= 0 and ship[row][col - 1] == '1':
        count += 1
    if col + 1 < D and ship[row][col + 1] == '1':
        count += 1
    return count

def open_neighbours(ship, D, row, col):
    r = []
    #checking for neighbors and if they exist storing them in the r
    if row - 1 >= 0 and ship[row - 1][col] in ('1', 'B', 'C'):
        r.append((row - 1, col))
    if row + 1 < D and ship[row + 1][col] in ('1', 'B', 'C'):
        r.append((row + 1, col))
    if col - 1 >= 0 and ship[row][col - 1] in ('1', 'B', 'C'):
        r.append((row, col - 1))
    if col + 1 < D and ship[row][col + 1] in ('1', 'B', 'C'):
        r.append((row, col + 1))
    return r

def check_blocked_neighbours(ship, D, row, col):
    result = []
    #Checks all neighbors of a cell that are closed and storing them
    if row - 1 >= 0 and ship[row - 1][col] == '0':
        result.append((row - 1, col))
    if row + 1 < D and ship[row + 1][col] == '0':
        result.append((row + 1, col))
    if col - 1 >= 0 and ship[row][col - 1] == '0':
        result.append((row, col - 1))
    if col + 1 < D and ship[row][col + 1] == '0':
        result.append((row, col + 1))
    return result

def get_blocked_cells_with_one_open_neighbor(ship, D):
    result = []
    #checking for closed cells with only one neighbour and storing them
    for i in range(D):
        for j in range(D):
            if ship[i][j] == '0':
                neighbours = get_open_neighbours_count(ship, D, i, j)
                if neighbours == 1:
                    result.append((i, j))
    return result

def get_open_cells(ship, D):
    result = []
    #Checking the cells to get the open cells 
    for i in range(D):
        for j in range(D):
            if ship[i][j] == '1':
                result.append((i, j))
    return result

def get_dead_ends(ship, D):
    result = []
    #Chekcing for open cells with only one neighbour
    for i in range(D):
        for j in range(D):
            if ship[i][j] == '1':
                neighbours = get_open_neighbours_count(ship, D, i, j)
                if neighbours == 1:
                    result.append((i, j))
    return result

def heuristic(a, b):
    x1, y1 = a
    x2, y2 = b
    #Manhattan Distance
    return abs(x1 - x2) + abs(y1 - y2)

def astar(ship, S, G, D):
    
    fringe = PriorityQueue()
    fringe.put((0, S))
    #Dictionary to store the cost of reaching each cell from start
    dist = {cell: inf for cell in product(range(D), range(D))}
    dist[S] = 0
    #Dictionary to keep track of the parent cell of each cell
    prev = {cell: None for cell in product(range(D), range(D))}

    while not fringe.empty():
        _, curr = fringe.get()
        if curr == G:
            path = []
            while curr in prev:
                #Reconstructing the path from start to goal
                path.append(curr)
                curr = prev[curr]
            return path[::-1]

        x, y = curr
        neighbours = open_neighbours(ship, D, x, y)

        for n in neighbours:
            #Calculate the total cost
            temp_dist = dist[curr] + 1
            if n not in dist or temp_dist < dist[n]:
                dist[n] = temp_dist
                prev[n] = curr

                priority = temp_dist + heuristic(n, G)
                fringe.put((priority, n))

    return None

def generate_ship_layout(D):
    #Creating a matrix with all closed cells
    ship = [['0' for _ in range(D)] for _ in range(D)]
    random.seed()
    #Opening some cells randomly
    start_row = random.randint(0, D - 1)
    start_col = random.randint(0, D - 1)
    ship[start_row][start_col] = '1'

    while True:
        #Getting blocked cells with only one open neighbour and opening them randomly
        blocked_cells = get_blocked_cells_with_one_open_neighbor(ship, D)
        if not blocked_cells:
            break
        index = random.randint(0, len(blocked_cells) - 1)
        new_x, new_y = blocked_cells[index]
        ship[new_x][new_y] = '1'
    
    dead_ends = get_dead_ends(ship, D)
    random.seed()
    #Removing dead ends at random
    for _ in range(len(dead_ends) // 2):
        index = random.randint(0, len(dead_ends) - 1)
        new_x, new_y = dead_ends[index]
        
        if ship[new_x][new_y] == '1':
            blockedNeighbours = check_blocked_neighbours(ship, D, new_x, new_y)
            if len(blockedNeighbours) >= 1:
                index = random.randint(0, len(blockedNeighbours) - 1)
                new_x, new_y = blockedNeighbours[index]
                ship[new_x][new_y] = '1'

    return ship

def movealiens(ship, aliens_pos, D):
    aliens_new_pos = []
    
    for i in range(len(aliens_pos)):
        x, y = aliens_pos[i]
        #Checking for open neighbours of each alien and storing them
        neighbours = open_neighbours(ship, D, x, y)
        
        #If there are neighbours then we select one neighbour at random and move the alien there
        if neighbours:
            
            index = random.randint(0, len(neighbours) - 1)
            D_x, D_y = neighbours[index]
            aliens_new_pos.append((D_x, D_y))
            ship[x][y] = '1'   
            ship[D_x][D_y] = 'A'
        else:
            aliens_new_pos.append((x, y))
    return aliens_new_pos
    

def bot1sim(ship, S, G, aliens_pos, D):
    i = 0
    #getting the shortest path
    path = astar(ship, S, G, D)
    visualize_ship(ship, G, D)
    plt.pause(0.8)

    while True:
        
        if path:
            if i < len(path) - 1:
                x, y = path[i]
                x_new, y_new = path[i + 1]
                #Moving the bot to the next position
                ship[x][y] = '1'
                ship[x_new][y_new] = 'B'
                print("Bot Moved")
            
            #If Bot reaches Captain
            if ((x_new, y_new) == G and (x_new, y_new) not in aliens_pos):
                print("Success")
                visualize_ship(ship, G, D)
                plt.pause(0.8)
                break
            
            #If Bot Goes into Alien Cell 
            elif ((x_new, y_new) in aliens_pos):
                print("Failed")
                visualize_ship(ship, G, D)
                plt.pause(0.8)
                break
            
            aliens_pos=movealiens(ship, aliens_pos, D)
            
            

            visualize_ship(ship, G, D)
            plt.pause(0.8)
            
            #If Aliens entered the Bot cell
            if ((x_new, y_new) in aliens_pos):
                print("Failure!!")
                visualize_ship(ship, G, D)
                plt.pause(0.8)
                break
                
        else:
            print("Path not Found")
            break
        i = i + 1
        

def main():
    D = 10
    K = 10
    aliens_size = K
    #Generating the layout of the ships
    ship = generate_ship_layout(D)
    
    #storing open cells
    open_cells = get_open_cells(ship, D)

    bot1 = 'B'
    alien = 'A'
    captain = 'C'
    
    #Placing the Bot Randomly
    index = random.randint(0, len(open_cells) - 1)
    S_x, S_y = open_cells[index]
    ship[S_x][S_y] = bot1
    S = (S_x, S_y)
    
    #Placing the Captain Randomly
    while True:
        i = random.randint(0, len(open_cells) - 1)
        D_x, D_y = open_cells[i]
        if ship[D_x][D_y] != bot1:
            ship[D_x][D_y] = captain
            break
    G = (D_x, D_y)
    aliens_pos = []
    
    #Placing Aliens randomly
    while True:
        index = random.randint(0, len(open_cells) - 1)
        new_x, new_y = open_cells[index]
        if ship[new_x][new_y] != bot1:
            ship[new_x][new_y] = alien
            aliens_size = aliens_size - 1
            aliens_pos.append((new_x, new_y))
        if aliens_size == 0:
            break

    visualize_ship(ship, G, D)
    plt.pause(0.8)
        
    bot1sim(ship, S, G, aliens_pos, D)




if __name__ == "__main__":
    main()
