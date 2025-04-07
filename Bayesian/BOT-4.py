from itertools import product
from math import exp
import random
import copy
import numpy as np
import matplotlib.pyplot as plt
from queue import PriorityQueue
import seaborn as sns

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
    if row - 1 >= 0 and ship[row - 1][col] not in '0':
        r.append((row - 1, col))
    if row + 1 < D and ship[row + 1][col] not in '0':
        r.append((row + 1, col))
    if col - 1 >= 0 and ship[row][col - 1] not in '0':
        r.append((row, col - 1))
    if col + 1 < D and ship[row][col + 1] not in '0':
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
    dist = {S:0}
    #Dictionary to keep track of the parent cell of each cell
    prev = {}

    while not fringe.empty():
        _, curr = fringe.get()
        if curr == G:
            path = []
            while curr in prev:
                #Reconstructing the path from start to goal
                path.append(curr)
                curr = prev[curr]
            path.append(S)
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

def initialize_knowledge_grid(ship, D, open_cells, sensor_cells,bot_pos):
    crew_knowledge =np.zeros((D, D, D, D), dtype=np.longdouble) 
    crew_knowledge_updated= np.zeros((D, D))

    alien_knowledge = [[0 for _ in range(D)] for i in range(D)]
    
    for i in range(D):
        for j in range(D):
            if ship[i][j] not in ('B','0'):
                sum=0
                for x in range(D):
                    for y in range(D):
                        if ship[x][y] not in ('B','0'):
                            if (i,j) != (x,y):
                                crew_knowledge[i][j][x][y]=1/(len(open_cells))*(len(open_cells)-1)  
                                sum=sum+crew_knowledge[i][j][x][y] 
                crew_knowledge_updated[i][j]=sum

    for i in range(D):
        for j in range(D):
            if (i,j) not in sensor_cells and ship[i][j] not in ('B','0'):
                alien_knowledge[i][j] = float(1/(len(open_cells) - len(sensor_cells)))
    return crew_knowledge,crew_knowledge_updated, alien_knowledge




# Example initialization in the main function
def get_sensor_cells(bot_pos, k, grid_size):

    x, y = bot_pos
    cells = []
    
    # Define the range of rows and columns to iterate over
    start_x = max(x - k, 0)
    end_x = min(x + k+1, grid_size - 1)
    start_y = max(y - k, 0)
    end_y = min(y + k+1, grid_size - 1)
    
    # Nested loop to add each cell within the specified area
    for i in range(start_x, end_x + 1):
        for j in range(start_y, end_y + 1):
            cells.append((i, j))
    
    return cells


def sense_aliens(ship, bot_pos, D, k,sensor_cells,alien_pos):
    # Calculate the bounds of the (2k+1) x (2k+1) area centered on the bot

    
    # Iterate through the area to check for the presence of aliens
    count=0
    for alien in alien_pos:
        if alien in sensor_cells:
            count+=1
    if count>=1:
        return True
    else:
        return False

def get_denom1(alien_knowledge,sensor_cells):
    denom1=0.0
    
    for (i,j) in sensor_cells:
        if alien_knowledge[i][j]!=0.0:
            denom1=denom1+(alien_knowledge[i][j])
    return denom1

def get_denom2(alien_knowledge, sensor_cells,D):

    denom2 = 0.0

    # Iterate through all cells in alien_knowledge
    for i in range(D):
        for j in range(D):
            # If the cell is not in sensor_cells, add its probability to denom2
            if (i, j) not in sensor_cells and alien_knowledge[i][j] != 0.0:
                denom2 = denom2 + alien_knowledge[i][j]

    return denom2

def update_alien_knowledge_before_movement(alien_knowledge,sense,sensor_cells,D):    
    denom1=get_denom1(alien_knowledge,sensor_cells)
    denom2 = get_denom2(alien_knowledge, sensor_cells,D)
    for i in range(D):
        for j in range(D):
            if sense:
                if (i,j) in sensor_cells:
                    alien_knowledge[i][j]=float(alien_knowledge[i][j]/denom1)
                else:
                    alien_knowledge[i][j]=0
            else:
                
                if (i, j) not in sensor_cells:
                    alien_knowledge[i][j] = float(alien_knowledge[i][j] /denom2) 
                else:
                    alien_knowledge[i][j]=0
                
    return alien_knowledge
def update_crew_knowledge_before_movement_1(ship,crew_knowledge,sense,D,bot_pos,alpha):
    x,y= bot_pos
    denom=0
    updated_crew_knowledge = copy.deepcopy(crew_knowledge)

    for i in range(D):
        for j in range(D):
            d=abs(i-x)+abs(j-y)
            prob=exp(-alpha * (d-1))
            if sense:
                denom= denom+(crew_knowledge[i][j]*prob)
            else:
                denom= denom+(crew_knowledge[i][j]*(1-prob))
    
    if sense:
        for i in range(D):
            for j in range(D):
                d=abs(i-x)+abs(j-y)
                prob=exp(-alpha * (d-1))
                
                if ship[i][j] not in ('B','0'):
                    updated_crew_knowledge[i][j]=float((crew_knowledge[i][j]*prob)/denom)
                else:
                    updated_crew_knowledge[i][j]=0
    
    else:
        for i in range(D):
            for j in range(D):
                d=abs(i-x)+abs(j-y)
                prob=exp(-alpha * (d-1))
                
                if ship[i][j] not in ('B','0'):
                    updated_crew_knowledge[i][j]=float((crew_knowledge[i][j]*(1-prob))/denom)
                else:
                    updated_crew_knowledge[i][j]=0
    
    
    return updated_crew_knowledge
    
def update_crew_knowledge_before_movement_2(ship,crew_knowledge,crew_knowledge_updated,sense,D,bot_pos,alpha,crew_pos):
    bpx,bpy= bot_pos
    crew_knowledge_temp = copy.deepcopy(crew_knowledge)
    sum=0
    goal=tuple()
    for i in range(D):
        for j in range(D):
            if crew_knowledge_updated[i][j]!=0:
                for x in range(D):
                    for y in range(D):
                        if crew_knowledge[i][j][x][y]!=0:
                            d1=abs(bpx-i)+abs(bpy-j)
                            prob1=exp(-alpha * (d1-1))
                            prob2=0
                            
                            if len(crew_pos)>1:
                                d2=abs(x-bpx)+abs(y-bpy)
                                prob2=exp(-alpha * (d2-1))
                                
                            beep=prob1+prob2-(prob1*prob2)
                            if sense:
                                sum=sum+ (crew_knowledge[i][j][x][y] * beep)
                            else:
                                sum=sum+ (crew_knowledge[i][j][x][y] *(1-beep))
    prob=sum
    max_sum = 0
    
    for i in range(D):
        for j in range(D):
            beep = 0
            if crew_knowledge_updated[i][j] != 0:
                sum = 0
                for x in range(D):
                    for y in range(D):
                        d1=abs(bpx-i)+abs(bpy-j)
                        prob1=exp(-alpha * (d1-1))
                        prob2 = 0
                        if len(crew_pos)>1:
                            d2=abs(x-bpx)+abs(y-bpy)
                            prob2=exp(-alpha * (d2-1))
                        beep = prob1 + prob2 - (prob1 * prob2)
                        if sense:
                            crew_knowledge_temp[i][j][x][y] = crew_knowledge[i][j][x][y] * beep / prob
                        else:
                            crew_knowledge_temp[i][j][x][y] = crew_knowledge[i][j][x][y] * (1-beep) / prob
                        sum += crew_knowledge_temp[i][j][x][y]

                crew_knowledge_updated[i][j] = sum
                if sum > max_sum and ship[i][j] != 'B':
                    goal = (i, j)
                    max_sum = sum

    return crew_knowledge_temp, goal
    
def normalize_crew_knowledge_after_bot_move(D,crew_knowledge,crew_knowledge_updated,alien_knowledge,bot_pos):

    denominator = 1 - crew_knowledge_updated[bot_pos]
    crew_knowledge_updated[bot_pos[0]] = 0
    for i in range(D):
        for j in range(D):
            denominator -= crew_knowledge[i][j][bot_pos[0]][bot_pos[1]]
            crew_knowledge[i][j][bot_pos[0]][bot_pos[1]] = 0

    for i in range(D):
        for j in range(D):
            if crew_knowledge_updated[i][j] != 0:
                sums = 0
                for k in range(D):
                    for l in range(D):
                        if crew_knowledge[i][j][k][l] != 0:
                            crew_knowledge[i][j][k][l] = (crew_knowledge[i][j][k][l] / denominator)
                            sums += crew_knowledge[i][j][k][l]

                crew_knowledge_updated[i][j] = sums





def sense_crew(ship, bot_pos, crew_pos, alpha):
    x,y=bot_pos
    i,j=crew_pos[0]
    d = abs(i-x) + abs(j-y)
    probability = exp(-alpha * (d - 1))
    index=random.random()
    if(index < probability):
        return True
    
    else:
        return False


def sense_crew1(bot_pos, crew_pos, alpha):
    x,y=bot_pos
    i,j=crew_pos[0]
    
    d1 = abs(i-x) + abs(j-y)
    
    probability1 = exp(-alpha * (d1 - 1))
    probability2=0
    if len(crew_pos)>1:
        k,l=crew_pos[1]
        d2= abs(k-x)+abs(l-y)
        probability2 = exp(-alpha * (d2 - 1))
    index=random.random()
    if(index < probability1) or (index<probability2):
        return True
    
    else:
        return False

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
    
    
def place_crew(ship,open_cells):
    while True:
        i = random.randint(0, len(open_cells) - 1)
        D_x, D_y = open_cells[i]
        if ship[D_x][D_y] != 'B':
            ship[D_x][D_y] = 'C'
            break
    G = (D_x, D_y)
    return G

def place_aliens(ship, D, sensor_cells,open_cells):
    while True:
        index = random.randint(0, len(open_cells) - 1)
        new_x, new_y = open_cells[index]
        if ship[new_x][new_y] != 'B':
            if (new_x, new_y) not in sensor_cells:
                ship[new_x][new_y] = 'A'
                return (new_x,new_y)


def check(ship,bot_pos,goal,D,aliens_pos):

    # If bot reached goal and alien is not present at goal
    if(bot_pos in goal and bot_pos not in aliens_pos):
        if len(goal)<=1:
            print("Success!!!")
            return "SUCCESS",aliens_pos
        else:
            print("Lets find the second crew since there more beeps coming")
            return "Next Crew",aliens_pos
    # If bot reached a cell where alien is present
    elif(bot_pos in aliens_pos):
        print("GAME OVER!!!! Bot Moved to Alien pos")
        return "FAILURE",bot_pos

    # Move the aliens to a random neighboring cell
    #print(aliens_pos)
    aliens_pos = movealiens(ship,aliens_pos,D)

    # If aliens reach the bot
    if bot_pos in aliens_pos:
        print("GAME OVER!!!! Alien moved to bot pos")
        return "FAILURE",bot_pos

    return "NEXT",aliens_pos

def update_alien_knowledge_after_movement(ship,D,alien_knowledge):
    
    updated_alien_knowledge=copy.deepcopy(alien_knowledge)

    for i in range(D):
        for j in range(D):
            sum=0
            if ship[i][j]=='0':
                sum+=0
            
            else:
                neighbours=open_neighbours(ship,D,i,j)

                for (x,y) in neighbours:
                    num_open_neighbours=len(open_neighbours(ship,D,x,y))
                    sum=sum+float(alien_knowledge[x][y]*(1/num_open_neighbours))
                updated_alien_knowledge[i][j]=sum
    return updated_alien_knowledge


def main():
    D =10
    ship = generate_ship_layout(D)
    open_cells = get_open_cells(ship, D)
    k = 2
    alpha=0.1
    bot1 = 'B'
    alien = 'A'
    captain = 'C'
    
    #Placing the Bot Randomly
    index = random.randint(0, len(open_cells) - 1)
    S_x, S_y = open_cells[index]
    ship[S_x][S_y] = bot1
    S = (S_x, S_y)
    

    sensor_cells = get_sensor_cells(S,k,D)
    goal=[]
    # Placing the captain randomly other than bot position in an open cell
    goal1 = place_crew(ship, open_cells)
    goal.append(goal1)
    goal2=place_crew(ship,open_cells)
    goal.append(goal2)
    print(goal)
    #print_ship(pre_computed_distances, D)

    # Placing the alien randomly other than bot position and sensor cells in an open cell
    alien_pos = place_aliens(ship, D, sensor_cells,open_cells)
    aliens_pos = []
    aliens_pos.append(alien_pos)

    #Initiating crew probability
    
    crew_knowledge,crew_knowledge_updated,alien_knowledge=initialize_knowledge_grid(ship,D,open_cells,sensor_cells,S)

    x= bot4_run(ship,goal,S,D,k,alpha, aliens_pos,open_cells,crew_knowledge,crew_knowledge_updated,alien_knowledge)

    print(x)




    


def bot4_run(ship,crew_pos,bot_pos,D,k,alpha,alien_pos,open_cells,crew_knowledge,crew_knowledge_updated,alien_knowledge):
    
    step=1
    path=None
    while step<1000:
        sensor_cells=get_sensor_cells(bot_pos,k,D)
        print("-------------------------\n Step ",step,"\n\n")
        #Sense Crew 
        if(len(crew_pos)>=1):
            crew_sense=sense_crew1(bot_pos,crew_pos,alpha)
            print("Sensed-Crew",crew_sense)
            
            print()
        #update crew knowledge
            crew_knowledge,max_crew_pos=update_crew_knowledge_before_movement_2(ship,crew_knowledge,crew_knowledge_updated,crew_sense,D,bot_pos,alpha,crew_pos)
 
        else:
            return "SUCCESS",step
        
        #Sense Aliens
        alien_sense=sense_aliens(ship,bot_pos,D,k,sensor_cells,alien_pos)
        print("Sensed-Alien",alien_sense)
        print()
        #update Alien knowledge
        alien_knowledge=update_alien_knowledge_before_movement(alien_knowledge,alien_sense,sensor_cells,D)
        

        alien_know_max=alien_knowledge[0][0]
        a_x,a_y=0,0
        for i in range(D):
            for j in range(D):
                    #Getting the position of an alien that has the highest probability of having an alien
                    if alien_knowledge[i][j]>alien_know_max:
                        alien_know_max=alien_knowledge[i][j]
                        a_x=i
                        a_y=j
        
        temp_ship=copy.deepcopy(ship)
        temp_ship[a_x][a_y]='A'
        #Finding path
        path= astar(temp_ship,bot_pos,max_crew_pos,D)
        if path and len(path)>1:
            x, y = bot_pos
            x_new, y_new = path[1]
            bot_pos = x_new, y_new
            #Updating Bot Position
            ship[x][y] = '1' 
            ship[x_new][y_new] = 'B'
            #check the position of bot if it is in an open cell with no aliens or if it entered an alien position or it found the crew
            status,alien_pos= check(ship,bot_pos,crew_pos,D,alien_pos)
            new_goal=[]
            step+=1
            s=(x_new,y_new)
            if status=="NEXT":
                crew_knowledge_updated[x_new][y_new] = 0
                alien_knowledge[x_new][y_new] = 0   
                for x in range(D):
                    for y in range(D):
                        crew_knowledge[x_new][y_new][x][y]=0
                #updating alien knowledge after the alien moves
                alien_knowledge = update_alien_knowledge_after_movement(ship, D, alien_knowledge)

                continue
            #After one crew is found
            elif status == "Next Crew":
                s=(x_new,y_new)
                crew_knowledge_updated[x_new][y_new] = 0
                alien_knowledge[x_new][y_new] = 0
                for x in range(D):
                    for y in range(D):
                        crew_knowledge[x_new][y_new][x][y]=0
                #updating alien knowledge after the alien moves
                alien_knowledge = update_alien_knowledge_after_movement(ship, D, alien_knowledge)
                for (i,j) in crew_pos:
                    p=(i,j)
                    
                    if s!=p:
                        new_goal.append(p)
                crew_pos=new_goal  

                continue
        
            elif status in ("SUCCESS","FAILURE"):

                return status, step
        return "FAILURE", step
    if step==1000:
        return "FAILURE",step
            


if __name__ == "__main__":
    main()
