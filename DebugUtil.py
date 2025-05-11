import numpy as np
import torch

def printAverageDirectionChange(landmark_vector):
    x = 0
    y = 0
    z = 0
    for i in range(0, 63, 3):
        x += landmark_vector[i]
        y += landmark_vector[i + 1]
        z += landmark_vector[i + 2]

    x /= 21
    y /= 21
    z /= 21
    print("Coordinate change x: " + str(x))
    print("Coordinate change y: " + str(y))
    print("Coordinate change z: " + str(z))


def printLandMarkCoordinate(landmark_vector, landmark_index):
    landmark_x = landmark_vector[1 + landmark_index * 3]
    landmark_y = landmark_vector[(landmark_index * 3) + 2]
    landmark_z = landmark_vector[(landmark_index * 3) + 3]

    print("Coordinate change x: " + str(landmark_x))
    print("Coordinate change y: " + str(landmark_y))
    print("Coordinate change z: " + str(landmark_z))


def printTensor(landmark_vector):
    print("Tensor: " + str(torch.tensor(landmark_vector)))

def printNPArray(arr):
    print("[", end='')

    for row in range(len(arr)):
        print("[", end='')

        for col in range(len(arr[row])):
            print(str(arr[row][col]), end='')
            if col < len(arr[row]) - 1:
                print(", ", end='')

        print("]",end = '')
        if row < len(arr) - 1:
            print(", ")

    print("]")