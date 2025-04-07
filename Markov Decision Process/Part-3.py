# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

filename="data.csv"
D=11
action=[[[[0]*D for i in range(D)] for j in range(D)] for k in range(D)]

#read Data to list
def read_csv_to_list(filename):
    df = pd.read_csv(filename)
    dl = df.values.tolist()
    dl = [[int(d[0]), int(d[1]), int(d[2]), int(d[3]), int(d[4]), int(d[5]), int(d[6])] for d in dl[1:]]
    return dl
#Initialize the state to the best action
states = read_csv_to_list(filename)
for state in states[1:]:
    action[state[0]][state[1]][state[2]][state[3]] = [state[5], state[6]]

#Split the data to states as X and actions as y
X = np.array([state[:4] for state in states])  # Bot and crew coordinates
y = np.array([state[5:] for state in states])

if y.shape[1] != 2:
    print("Output data does not have the expected shape. Reshaping...")
    outputs = y.reshape((-1, 2))

#Split the data into train and test for the model
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
if y_train.shape[1] != 2 or y_val.shape[1] != 2:
    print("Output data shapes are incorrect. Reshaping...")
    y_train = y_train.reshape((-1, 2))
    y_val = y_val.reshape((-1, 2))

model = Sequential([
    Dense(64, activation='relu', input_shape=(4,)),
    Dense(64, activation='relu'),
    Dense(128, activation='relu'),
    Dense(128, activation='relu'),
    Dense(2)  # Output layer with 2 units for [e, f]
])

#Compile the model using adam optimizer , Mean Absolute Error , and calculate accuracy metrics
model.compile(optimizer='adam', loss='mae', metrics=['accuracy'])

#Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))

#Model Accuracy
test_loss, test_acc = model.evaluate(X_val, y_val, verbose=2)
print(f"Test accuracy: {test_acc:.4f}")

#Save the model
model.save('trained_bot_model.h5')

import random
def open_neighbour_crew(ship, D, row, col):
  #Get the neighbours of crew which are Up, Down, Left, Right
    result = []
    if row - 1 >= 0 and ship[row - 1][col] not in ('0', 'B'):
        result.append((row - 1, col))
    if row + 1 < D and ship[row + 1][col] not in ('0', 'B'):
        result.append((row + 1, col))
    if col - 1 >= 0 and ship[row][col - 1] not in ('0', 'B'):
        result.append((row, col - 1))
    if col + 1 < D and ship[row][col + 1] not in ('0', 'B'):
        result.append((row, col + 1))
    return list(result)

def move_crew(crew,Ship,bo):
    alien_new_positions=[]
    #print("lawda: ",cap)
    for i in range(1):
        k1=open_neighbour_crew(Ship,D,crew[0],crew[1])
        if len(k1)==1:

            alien_new_positions.append(list(k1[0]))
        elif len(k1)>1:
            while True:
                newa=k1[random.randint(0,len(k1)-1)]
                if bo!=list(newa):
                    break
            alien_new_positions.append(list(newa))
    return alien_new_positions[0]



def get_open_cells(ship, D):
  #Get the open cells in the ship
    result = []
    for i in range(D):
        for j in range(D):
            if ship[i][j] in ('1'):
                result.append((i, j))
    return result

def generate_ship(D):
    ship = [['1' for _ in range(D)] for _ in range(D)]
    teleport_row = D // 2
    teleport_col = D // 2
    ship[teleport_row][teleport_col] = 'T'

    # Block diagonal neighbors of the teleport pad
    for dr in (-1, 1):
        for dc in (-1, 1):
            if 0 <= teleport_row + dr < D and 0 <= teleport_col + dc < D:
                ship[teleport_row + dr][teleport_col + dc] = '0'

    # Randomly place additional obstacles, avoiding the teleport pad and its immediate paths
    obstacles_needed = 10
    placed_obstacles = 0
    while placed_obstacles < obstacles_needed:
        row, col = random.randint(0, D - 1), random.randint(0, D - 1)

        # Ensure obstacles are not placed on the teleport pad, or replace existing obstacles
        if (row, col) != (teleport_row, teleport_col) and ship[row][col] == '1':
            # Avoid blocking direct paths to the teleport pad by ensuring no obstacles
            # are placed in the same row or column
            if row != teleport_row and col != teleport_col:
                ship[row][col] = '0'
                placed_obstacles += 1

    return ship

ship_length=11


teleport=[ship_length//2,ship_length//2]


D=ship_length
td=D//2

ship = [['1', '1', '1', '1', '1', '1', '1', '1', '0', '1', '1'], ['1', '1', '1', '1', '0', '1', '1', '1', '1', '1', '1'], ['0', '0', '1', '1', '1', '1', '1', '1', '1', '1', '1'], ['0', '1', '1', '1', '1', '1', '1', '1', '0', '1', '1'], ['1', '1', '1', '1', '0', '1', '0', '1', '1', '1', '1'], ['1', '1', '1', '1', '1', 'T', '1', '1', '1', '1', '1'], ['1', '1', '1', '1', '0', '1', '0', '1', '1', '1', '1'], ['1', '1', '1', '0', '1', '1', '1', '0', '1', '1', '1'], ['0', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1'], ['1', '1', '1', '1', '1', '1', '1', '1', '1', '0', '1'], ['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1']]

crew=[4,2]
bot=[7,2]



steps=0
iterations=0

def round(num):
    if num-int(num) >= 0.5:
        return int(num)+1
    else:
        return int(num)


for game in range(30):
    bot1 = [bot[0], bot[1]]
    crew1 = [crew[0], crew[1]]
    inner_loop_steps=0
    while True:
        if crew1 != teleport:
            cas = [bot1[0], bot1[1], crew1[0], crew1[1]]
            count=0

            while True:

                a = model.predict(np.array([cas]))[0]
                print("predict: ", a,", Captain: ",crew1)
                count+=1
                s = []
                if count==2:
                  s=bot1
                  break
                w1 = [round(a[0]), round(a[1])]
                w2 = [int(a[0]), int(a[1])]
                if 0 <= w1[0] < D and 0 <= w1[1] < D:
                    if ship[w1[0]][w1[1]] != '0' and w1 != crew1:
                        s = w1
                        break
                    elif ship[w2[0]][w2[1]] != '0' and w2 != crew1:
                        s = w2
                        break
            bot1 = s
            crew1 =(move_crew(crew1, ship, bot1))
            print(crew1,bot1)
            inner_loop_steps += 1
        else:
            steps += inner_loop_steps
            iterations += 1
            print(inner_loop_steps, end=" ")
            break


print(f"\nAverage steps: {steps / (iterations)}")