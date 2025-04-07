from math import exp
import random
import copy
import numpy as np
import matplotlib.pyplot as plt
from queue import PriorityQueue    

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
    #Initializing  the knowledge grid for crew and aliens
    crew_knowledge =np.zeros((D, D, D, D), dtype=np.longdouble) 
    crew_knowledge_updated= np.zeros((D, D))

    alien_knowledge = np.zeros((D, D, D, D))
    alien_knowledge_updated = np.zeros((D, D))
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
            if ship[i][j] not in ('0') and ship[i][j] not in sensor_cells:
                sum=0
                for x in range(D):
                    for y in range(D):
                        if ship[x][y] not in ('0') and ship[x][y] not in sensor_cells:
                            alien_knowledge[i][j][x][y] = float(1/((len(open_cells)-len(sensor_cells)-1)*(len(open_cells)-len(sensor_cells)-2)))
                            sum=sum+alien_knowledge[i][j][x][y]
                alien_knowledge_updated[i][j]=sum
    return crew_knowledge,crew_knowledge_updated, alien_knowledge,alien_knowledge_updated


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


def sense_aliens(aliens_pos,sensor_cells):
    # Iterate through the area to check for the presence of aliens
    for alien in aliens_pos:
        if alien in sensor_cells :
            return True
    return False

def get_denom(alien_knowledge,alien_knowledge_updated,D,sense,sensor_cells):
    denom=0
    #Based on if an alien is sensed or not, we update the denom with prob of sensing or (1-prob of sensing)
    for i in range(D):
        for j in range(D):
                for x in range(D):
                    for y in range(D):
                            if (i,j) in sensor_cells:
                                prob1=1
                            else:
                                prob1=0
                            if (x,y) in sensor_cells:
                                prob2=1
                            else:
                                prob2=0
                            #Finding union to make sure both aliens bepps are being considered
                            prob=prob1+prob2-(prob1*prob2)
                            if(sense):
                                denom += (alien_knowledge[i][j][x][y] * (prob))
                            else:
                                denom += (alien_knowledge[i][j][x][y] * (1-prob))

                  
                            
    return denom

def update_alien_knowledge_before_movement(alien_knowledge,alien_knowledge_updated,sense,sensor_cells,D):    
    denom=get_denom(alien_knowledge,alien_knowledge_updated,D,sense,sensor_cells)
    #Based on if an alien is sensed or not, we update the alien_knowledge and alien_knowledge_updated with prob of sensing or (1-prob of sensing)
    for i in range(D):
        for j in range(D):
                sum=0
                for x in range(D):
                    for y in range(D):
                            if (i,j) in sensor_cells:
                                prob1=1
                            else:
                                prob1=0
                            if (x,y) in sensor_cells:
                                prob2=1
                            else:
                                prob2=0
                            prob=prob1+prob2-(prob1*prob2)
                            if(sense):
                                alien_knowledge[i][j][x][y]= float((alien_knowledge[i][j][x][y]*prob)/denom)
                            
                            else:
                                alien_knowledge[i][j][x][y]= float((alien_knowledge[i][j][x][y]*(1-prob))/denom)
                            
                            sum =sum+alien_knowledge[i][j][x][y]
                            
                alien_knowledge_updated[i][j]=sum
                                
    return alien_knowledge,alien_knowledge_updated

    
def update_crew_knowledge_before_movement_2(ship,crew_knowledge,crew_knowledge_updated,sense,D,bot_pos,alpha,crew_pos):
    bpx,bpy= bot_pos
    crew_knowledge_temp = copy.deepcopy(crew_knowledge)
    #Based on if an crew is sensed or not, we update the sum with prob of sensing or (1-prob of sensing)
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
    #Based on if an crew is sensed or not, we update the crew_knowledge with prob of sensing or (1-prob of sensing)
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

#we sense crews  based on the distance and alpha parameter
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
        #placing crew at a random position
        i = random.randint(0, len(open_cells) - 1)
        D_x, D_y = open_cells[i]
        if ship[D_x][D_y] != 'B':
            ship[D_x][D_y] = 'C'
            break
    G = (D_x, D_y)
    return G

def place_aliens(ship, D, sensor_cells,open_cells):
    while True:
        #placing alien outside the sensor cells        
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

#We update the alien knowledge as if (i,j) has some probability of having alien then we distribute this probability to th neigbouring open_cell
def update_alien_knowledge_after_movement(ship,D,alien_knowledge,alien_knowledge_updated,open_cells):
    temp_alien_knowledge = np.zeros((D, D, D, D))

    for i in range(D):
        for j in range(D):
            if ship[i][j] != 0:
                neighbours1 = open_neighbours(ship, D, i, j)
                n1 = len(neighbours1)
                if n1 > 0:
                    v1 = alien_knowledge_updated[i][j] / n1
                    
                    for x in range(D):
                        for y in range(D):
                            if ship[x][y] != 0 and temp_alien_knowledge[i][j][x][y] == 0:
                                neighbours2 = open_neighbours(ship, D, x, y)
                                n2 = len(neighbours2)
                                if n2 > 0:
                                    v2 = alien_knowledge_updated[x][y] / n2
                                    mv = (v1 * v2)
                                    
                                    temp_alien_knowledge[i][j][x][y] += mv
                                    temp_alien_knowledge[x][y][i][j] += mv

    # Correctly sum and update alien_knowledge_updated
    for i in range(D):
        for j in range(D):
            alien_knowledge_updated[i][j] = np.sum(temp_alien_knowledge[i, j])

    return temp_alien_knowledge, alien_knowledge_updated


def normalize_alien_knowledge_after_bot_move(D,alien_knowledge_updated,alien_knowledge,bot_pos):
    
    sum = np.sum(alien_knowledge_updated)

    # Normalize the probabilities to make sure the total is 1
    if sum != 0:
        alien_knowledge_updated = alien_knowledge_updated/sum
        for i in range(D):
            for j in range(D):
                alien_knowledge[i][j] = alien_knowledge[i][j]/sum

    return alien_knowledge,alien_knowledge_updated      



def main():
    D =35
    ship = generate_ship_layout(D)
    open_cells = get_open_cells(ship, D)
    k = 8
    alpha=0.095
    bot1 = 'B'
    alien = 'A'
    captain = 'C'
    
    #Placing the Bot Randomly
    index = random.randint(0, len(open_cells) - 1)
    S_x, S_y = open_cells[index]
    ship[S_x][S_y] = bot1
    S = (S_x,S_y)
    
    #get the sensor cells
    sensor_cells = get_sensor_cells(S,k,D)

    goal=[]
    # Placing the captain randomly other than bot position in an open cell
    goal1 = place_crew(ship, open_cells)
    goal.append(goal1)
    goal2=place_crew(ship,open_cells)
    goal.append(goal2)


    # Placing the alien randomly other than bot position and sensor cells in an open cell
    alien_pos = place_aliens(ship, D, sensor_cells,open_cells)
    aliens_pos = []
    
    aliens_pos.append(alien_pos)
    alien_pos1=place_aliens(ship,D,sensor_cells,open_cells)
    aliens_pos.append(alien_pos1)

    #Initiating crew knowledge, alien knowledge
    
    crew_knowledge,crew_knowledge_updated,alien_knowledge,alien_knowledge_updated=initialize_knowledge_grid(ship,D,open_cells,sensor_cells,S)

    x= bot7_run(ship,goal,S,D,k,alpha, aliens_pos,open_cells,crew_knowledge,crew_knowledge_updated,alien_knowledge,alien_knowledge_updated)

    print(x)




    


def bot7_run(ship,crew_pos,bot_pos,D,k,alpha,alien_pos,open_cells,crew_knowledge,crew_knowledge_updated,alien_knowledge,alien_knowledge_updated):
    
    step=1
    path=None
    while step<1000:
        sensor_cells=get_sensor_cells(bot_pos,k,D)
        print("-------------------------\n Step ",step,"\n\n")
        if(len(crew_pos)>=1):
            crew_sense=sense_crew1(bot_pos,crew_pos,alpha)
            print("Sensed-Crew",crew_sense)
            
            print()
        #update crew knowledge
            crew_knowledge,max_crew_pos=update_crew_knowledge_before_movement_2(ship,crew_knowledge,crew_knowledge_updated,crew_sense,D,bot_pos,alpha,crew_pos)            
        
        else:
            return "SUCCESS",step
        
        #Sense Aliens
        alien_sense=sense_aliens(alien_pos,sensor_cells)
        print("Sensed-Alien",alien_sense,k,alien_pos)
        print()
        #update Alien knowledge
        alien_knowledge,alien_knowledge_updated=update_alien_knowledge_before_movement(alien_knowledge,alien_knowledge_updated,alien_sense,sensor_cells,D)
        #getting 2 alien positions with max probabilities 
        alien_know_max1=alien_knowledge[0][0][0][0]
        alien_know_max2=alien_knowledge_updated[0][0]
        a_x,a_y=0,0
        a_x1,a_y1=0,0
        for i in range(D):
            for j in range(D):
                    if alien_knowledge_updated[i][j]>alien_know_max2:
                        alien_know_max2=alien_knowledge_updated[i][j]
                        a_x=i
                        a_y=j
                    
        for x in range(D):
            for y in range(D):
                if alien_knowledge[a_x][a_y][x][y]>alien_know_max1:
                    alien_know_max1=alien_knowledge[a_x][a_y][x][y]
                    a_x1=x
                    a_y1=y

        temp_ship=copy.deepcopy(ship)
        temp_ship[a_x][a_y]='A'
        temp_ship[a_x1][a_y1]='A'
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
                alien_knowledge_updated[x_new][y_new] = 0
                for x in range(D):
                    for y in range(D):
                        crew_knowledge[x_new][y_new][x][y]=0
                        alien_knowledge[x_new][y_new][x][y]=0
                #updating alien knowledge after the alien moves                        
                alien_knowledge,alien_knowledge_updated = update_alien_knowledge_after_movement(ship,D,alien_knowledge,alien_knowledge_updated,open_cells)
                alien_knowledge,alien_knowledge_updated =normalize_alien_knowledge_after_bot_move(D,alien_knowledge_updated,alien_knowledge,s)


                continue
            elif status == "Next Crew":
                s=(x_new,y_new)
                crew_knowledge_updated[x_new][y_new] = 0
                alien_knowledge_updated[x_new][y_new] = 0
                for x in range(D):
                    for y in range(D):
                        crew_knowledge[x_new][y_new][x][y]=0
                        alien_knowledge[x_new][y_new][x][y]=0
                #updating alien knowledge after the alien moves
                alien_knowledge,alien_knowledge_updated = update_alien_knowledge_after_movement(ship,D,alien_knowledge,alien_knowledge_updated,open_cells)
                alien_knowledge,alien_knowledge_updated = normalize_alien_knowledge_after_bot_move(D,alien_knowledge_updated,alien_knowledge,bot_pos)
                for (i,j) in crew_pos:
                    p=(i,j)
                    
                    if s!=p:
                        new_goal.append(p)
                crew_pos=new_goal  

                continue
        
            elif status in ("SUCCESS","FAILURE"):
                #visualize_ship(ship,D,goal,status,final_bot_path)
                #plt.pause(5)

                return status, step
        return "FAILURE", step
    if step==1000:
        return "FAILURE",step
            


if __name__ == "__main__":
    main()
