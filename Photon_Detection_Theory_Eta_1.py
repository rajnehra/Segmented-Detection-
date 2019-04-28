""" This script is to model the photo detection theory and POVM analysis for Quantum Detector Tomography
    Author : Nehra, Rajveer
    Project: Darpa Detect
    Date : July 4th, 2018
    """
import numpy as np
import sys
#import pylab as pl
#from matplotlib import*
#import matplotlib.pyplot as plt
import os
import random
from math import *
from mpl_toolkits.mplot3d import Axes3D
from ortools.constraint_solver import pywrapcp
import matplotlib.pyplot as plt
from matplotlib import*


""" Some functions"""
def Shuffle(A):
    return random.shuffle(A)
def pow(x, y):
    return 1 if y == 0 else x * pow(x, y - 1)
def f(m,n):
    return factorial(m)/((factorial(m-n))*factorial(n))
def swap(i,j):
    i,j = j,i

def Randomizer(A, i, ones):
    if ones == 0:
        K_config.append(A.copy())
        return
    if i >= len(A):
        return
    A[i] = 1
    Randomizer(A, i+1, ones-1)
    A[i] = 0
    Randomizer(A, i+1, ones)

def Binomial_sum(N, a ,b):
    Sum = 0;
    for i in range(N):
        Sum+= f(N,i)*pow(a,i)*pow(b,N-i)
    return Sum


def CP_solver(num_input, num_mode, k_clicks):
    global  K, eta
    '''
    , eta, num_input, num_mode, k_clicks
    #print("Please enter the number of clicks : ")
    #k_clicks = int(input())
    #print("Input the number of photons : ")
    #num_input = int(input())
    #print("Input the number of modes/detectors : ")
    #num_mode = int(input())'''
    
    solver = pywrapcp.Solver("Photon_Detection")
    A = [solver.IntVar(0, num_input, "n%i" % i) for i in range(num_mode)]
    solver.Add(np.sum(A) == num_input)
    for i in range(k_clicks):
        solver.Add(A[i]!=0)
    for i in range(k_clicks, num_mode,1):
        solver.Add(A[i] == 0)
    
    db = solver.Phase(A, solver.CHOOSE_FIRST_UNBOUND, solver.ASSIGN_MIN_VALUE)
    solution = solver.Assignment()
    solution.Add(A)
    collector = solver.AllSolutionCollector(solution)
    solver.Solve(db, [collector])
#print("Solutions found:", collector.SolutionCount())
#print("Time:", solver.WallTime(), "ms")
#print()
    K = np.zeros((collector.SolutionCount(),num_mode))
    for sol in range(collector.SolutionCount()):
        #print("Solution number" , sol, '\n')
        for i in range(num_mode):
            K[sol][i] = (collector.Value(sol, A[i]))
    return K
"""All the configurations have been stored in an num_solutions * n_modes array, finally time to do the real things """

if __name__ == "__main__":
    print("Input the number of photons : ")
    num_input = int(input())
    print("Input the number of max_photon")
    num_input_max = int(input())
    print("Input the number of clicks : ")
    k_clicks = int(input())
    print("The number of Detectors : ")
    num_mode = int(input())
    Final_prob_fig = []
    N_ind = []
    '''The prog. is modified to calculate the purity vs m while keeping eta fixed and k_clicks changes '''
        #for n in range(num_input,num_input_max+1, 1):
        #print(n)
    CP_solver(n,num_mode,k_clicks)
    k_click_config = f(num_mode,k_clicks)
    P_config = []
    Eff_P_config = []
    Eff_Prod_fact_n_s_config = []
    Prod_fact_n_s_config = []
    Effective_sum_mult = 0
            #for i in range(len(K)):
            #Prod_eff = 1
            #print("The config is : ", K[i])
            
            ##   Prod_eff *= (-1 + (1.0/pow((1-eta), K[i][j])))
            #P_config.append(Prod_eff)

    for i in range(len(K)):
        print(K[i])
        Prod_fact_n_s = 1
        for j in range(num_mode):
            Prod_fact_n_s *=(1/factorial(K[i][j]))
        print(Prod_fact_n_s)
        Prod_fact_n_s_config.append(Prod_fact_n_s)

    for i in range(len(K)):
            Effective_sum_mult += Prod_fact_n_s_config[i]
    P_k_n = Effective_sum_mult*pow((1/num_mode), num_input)*factorial(n)*k_click_config
    Final_prob_fig.append(P_k_n)
#N_ind.append(int(n))

#f= open("Data_1_clicks_eta_1.txt","w+")
#   for i in range(len(N_ind)):
#       f.write(str(N_ind[i]) + str(',') + str(Final_prob_fig[i]) + "\n")
#   f.close()
#M_ind, Final_prob_fig = np.meshgrid(M_ind, Final_prob_fig)
#   plt.plot(N_ind, Final_prob_fig, '-')
#   plt.title("Eta = 1, k_click = 1, number of detectors = 50")
##   plt.ylabel("Prob. of 1 click")
#  plt.grid()
#   plt.show()
#   plt.savefig("1_click_photons.png")

    print("Ho gya poora")



# X, Y = np.meshgrid(M_ind, P_k_n)
#
##
#ax.plot_surface(x, y, z, cmap=plt.cm.jet, rstride=1, cstride=1, linewidth=0)
#
#ax = fig.gca(projection='3d')
#B = f(100,3)*factorial(3)*pow(1/100,3)
#   ax.plot_trisurf(N_ind, M_ind, P_ind, color = "yellow", linewidth=0)

#   ax.plot(N_ind, M_ind, P_ind, "3", color = "blue" )
#surf = ax.plot_surface(X, Y, P_ind, cmap=cm.coolwarm,
#linewidth=0, antialiased=False)
#wframe = ax.plot_wireframe(X, Y, P_ind, rstride=2, cstride=2)
#ax.set_title("Segmented detector with gradient reflectance")
#   ax.set_xlabel("Number of photons (n)")
#   ax.set_ylabel("Number of detectors (m)")
#   plt.show()
#   fig.savefig("PNR_1.png")







