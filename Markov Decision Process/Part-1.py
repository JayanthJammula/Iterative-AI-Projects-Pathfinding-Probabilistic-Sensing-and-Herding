import random
import numpy as np
import matplotlib.pyplot as plt
import copy


def open_neighbours_crew(ship, D, row, col):
    r = []
    # Checking for neighbors and if they exist storing them in the r
    if row - 1 >= 0 and ship[row - 1][col] in ('1','T'):
        r.append((row - 1, col))
    if row + 1 < D and ship[row + 1][col] in ('1','T'):
        r.append((row + 1, col))
    if col - 1 >= 0 and ship[row][col - 1] in ('1','T'):
        r.append((row, col - 1))
    if col + 1 < D and ship[row][col + 1] in ('1','T'):
        r.append((row, col + 1))
    return r
    
def print_matrix_with_full_grid(T):
    np.set_printoptions(precision=2, suppress=True, formatter={'float': '{: 0.2f}'.format})
    print("Grid-view of T Matrix:")
    print("+" + "-----------+" * T.shape[1])  # Header row grid
    for row in T:
        row_str = "|".join(f"{num:5.2f}" for num in row)
        print(f"|{row_str}|")  # Print row with vertical separators
        print("+" + "------------+" * T.shape[1])  # End of row grid

    
def main():
    ship = [['1', '1', '1', '1', '1', '1', '1', '1', '0', '1', '1'], ['1', '1', '1', '1', '0', '1', '1', '1', '1', '1', '1'], ['0', '0', '1', '1', '1', '1', '1', '1', '1', '1', '1'], ['0', '1', '1', '1', '1', '1', '1', '1', '0', '1', '1'], ['1', '1', '1', '1', '0', '1', '0', '1', '1', '1', '1'], ['1', '1', '1', '1', '1', 'T', '1', '1', '1', '1', '1'], ['1', '1', '1', '1', '0', '1', '0', '1', '1', '1', '1'], ['1', '1', '1', '0', '1', '1', '1', '0', '1', '1', '1'], ['0', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1'], ['1', '1', '1', '1', '1', '1', '1', '1', '1', '0', '1'], ['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1']]

    D = 11
    teleport = (5, 5)  # Teleport position
    T_no_bot = compute_T_no_bot(ship, D)
    print("T_no_Bot Matrix:")
    print()
    print_matrix_with_full_grid(T_no_bot)
    
    

def compute_T_no_bot(ship, D):
    T = np.full((D, D), np.inf)  # Start with NaN for clarity
    teleport_pos = (D // 2, D // 2)
    T[teleport_pos] = 0  # No steps needed from the teleport pad to itself

    # Initialize other cells
    for i in range(D):
        for j in range(D):
            if ship[i][j] == '1' or ship[i][j] == '0':  # Accessible cells
                T[i, j] = np.inf  #Large number np.inf for convergence
            elif ship[i][j] == 'T':
                T[i,j]=0#
    changed = True
    iterations = 0
    while changed and iterations < 10000:  # Limit iterations to prevent infinite loops
        changed = False
        max_change = 0
        for i in range(D):
            for j in range(D):
                if (i, j) == teleport_pos or ship[i][j] == '0':
                    continue
                current_value = T[i, j]
                # Compute the minimum expected time using the Bellman equation
                neighbor_values = open_neighbours_crew(ship,D,i,j)
                if neighbor_values:
                    neighbor=[]
                    for neighbors in neighbor_values:
                        neighbor.append(T[neighbors])
                    new_value = 1 + sum(neighbor) / len(neighbor_values)
                    T[i, j] = new_value
                    if abs(new_value - current_value) > 0:  # Check if significant change
                        changed = True
                        max_change = max(max_change, abs(new_value - current_value))
        iterations += 1
    
    return T


if __name__ == "__main__":
    main()
