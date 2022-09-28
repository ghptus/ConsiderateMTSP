# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 16:23:47 2022

@author: Admin
"""
#Libraries

import pandas as pd
import numpy as np
import random, math
import matplotlib.pyplot as plt
import warnings
from datetime import datetime
warnings.simplefilter(action='ignore', category=FutureWarning)




#Global Variables
environmentsize = 80

depot = [0,0]

points = 500
salesmanno = 20
difficultyrange = 100

generations = 1500    #The Number of Generations for the genetic algorithm

populationsize = 100    #The Population Size for each geenration
   
offspring = 25 #Determines how many offspring should be produced


mutationrating = 0.2 #The "level" of mutation performed/ how much is changed
mutationcutoff = 0.1
crossovervariation = 0.2
crossoverpattern = [5,5] #what rate are elements taken from each parent [5,5] would rotate 5 from parent A then 5 from parent B
startparent = 'a'

fitnesssuccess = 0

def generatesalesman():   
    salesmandf = pd.DataFrame(columns = ['Difficulty'])
    for i in range(0,salesmanno):
        salesmandf = salesmandf.append({'Difficulty' : (random.randint(1, difficultyrange))}, ignore_index = True)  
    salesmandf.Difficulty = salesmandf.Difficulty.astype(float)
    salesmandf = salesmandf.sort_values(by=["Difficulty"])
    salesmandf.reset_index(inplace=True,drop=True)
   # print("Generated Salesman...")
    return salesmandf

#generatedataset function
def generatedataset():
    graphdf = pd.DataFrame(columns = ['Difficulty','Latitude','Longitude','DepotDistance'])
    for i in range (0, points):
        Long = random.randint(0, environmentsize)
        Lat = random.randint(0, environmentsize)
        Dist = math.sqrt((depot[0]-Long)**2 + (depot[1]-Lat)**2)
        graphdf = graphdf.append({'Difficulty' : (random.randint(1, difficultyrange)), 'Latitude' : (Long), 'Longitude' : (Lat), 'DepotDistance' : (Dist)}, ignore_index = True)
    graphdf = graphdf.sort_values(by=["DepotDistance"])
    graphdf.reset_index(inplace=True,drop=True)
    #print("Generated Dataset...")
    return graphdf
    #distance matrix
    

def createdistancematrix(graphdf):
    C = np.zeros((2,points,points)) 
    for i in range(0,points):
        for j in range(0,points):
            a = (graphdf.loc[i].Longitude, graphdf.loc[i].Latitude)
            b = (graphdf.loc[j].Longitude, graphdf.loc[j].Latitude)
            #print (str(a) + " \ " + str(b))
            #print(math.sqrt((a[0]-b[0])**2 +(a[1]-b[1])**2))
            C[0,i,j] = math.sqrt((a[0]-b[0])**2 +(a[1]-b[1])**2)
            C[1,i,j] = ((graphdf.loc[i].Difficulty + graphdf.loc[j].Difficulty)/2)
    #print("Produced Distance Matrix...")
    return C
#solution generator -- generate initial population of solututions

def randomsolution(dataset):
    route = pd.DataFrame(columns = ['salesman','Stops','Distance','Difficulty','TotalDistance','DifficultyDiff'])
    for i in range(0, salesmanno):
        route = route.append({'salesman' : (str(i)) , 'Stops' : ""}, ignore_index = True)
    for i in range(0, points):
        x = random.randint(0, salesmanno-1)
        route.loc[x].Stops = str(route.loc[x].Stops) + " " + str(dataset.loc[i].name)
    for i in range(0, salesmanno):
        route.loc[i].Stops = list(route.loc[i].Stops.split(" "))
        route.loc[i].Stops.pop(0)
    
    return route

#create an initial population

def createpopulation(dataset):
    population = pd.DataFrame()
    
    for i in range(0, populationsize):
        population = population.append(randomsolution(dataset))
    
    population = np.array_split(population,populationsize)
    #print("Generated Random Solutions...")
    return population
#fitness Function

def calculatefitness(population, matrix, dataset, salesman):
    ii = 0
    for i in population:
        for y in range(0, salesmanno):
            distance = 0
            difficulty = 0
            lastpoint=-1
            counter = 0
            for z in i.loc[y].Stops:
                t1 = counter
                t2 = len(i.loc[y].Stops)-1
                if(lastpoint == -1 and len(i.loc[y].Stops) != 1):
                    #Calculate Distance
                    a = depot
                    b = [dataset.loc[int(z)].Latitude,dataset.loc[int(z)].Longitude]
                    distance = distance + math.sqrt((a[0]-b[0])**2 +(a[1]-b[1])**2)
                    lastpoint = z
                    
                    #Calculate Difficulty
                    difficulty = difficulty + dataset.loc[int(z)].Difficulty
                    
                elif (lastpoint == -1 and len(i.loc[y].Stops) != 1):
                    #Calculate Distance
                    a = depot
                    b = [dataset.loc[int(z)].Latitude,dataset.loc[int(z)].Longitude]
                    distance = distance + math.sqrt((a[0]-b[0])**2 +(a[1]-b[1])**2)
                    population[ii].loc[y].Distance = distance
                    
                    #Calculate Difficulty
                    difficulty = difficulty + dataset.loc[int(z)].Difficulty
                    population[ii].loc[y].Difficulty = difficulty/len(i.loc[y].Stops)
                    
                elif t1 == t2:
                    #Calculate Distance
                    dist = matrix[0,int(lastpoint),int(z)]
                    distance = distance + dist
                    population[ii].loc[y].Distance = distance
                    
                    #Calculate Difficulty
                    difficulty = difficulty + dataset.loc[int(z)].Difficulty
                    population[ii].loc[y].Difficulty = difficulty/len(i.loc[y].Stops)
                    
                    distance = 0
                    lastpoint=-1
                    counter = 0
                    difficulty= 0
                    
                    
                else:
                    #Calculate Distance
                    dist = matrix[0,int(lastpoint),int(z)]
                    distance = distance + dist
                    
                    #Calculate Difficulty
                    difficulty = difficulty + dataset.loc[int(z)].Difficulty
                   
                counter += 1
                #print(distance)
        ii += 1
    xx = 0
    iii = 0
    for i in population:
        population[iii] = i.sort_values(by=["Difficulty"])
        population[iii].reset_index(inplace=True,drop=True)
        iii += 1
    for i in population:
        distancetotal = 0
        difficultytotal = 0
        yy= 0
        for y in range(0, salesmanno):
            #print(i.loc[y])
            if i.loc[y].Distance > 0.001:
                distancetotal = distancetotal + i.loc[y].Distance
                difficultytotal = difficultytotal + math.sqrt((i.loc[y].Difficulty-salesman.loc[y].Difficulty)**2)
            
            population[xx].loc[y].DifficultyDiff = difficultytotal
            population[xx].loc[y].TotalDistance = distancetotal
            yy += 1   
        xx += 1
   #print("Produced fitness Scores...")    
    return(population)

def plotdataset(dataset):
    x = []
    y = []
    l = []
    for index, row in dataset.iterrows():
        x.append(row.Longitude)
        y.append(row.Latitude)
        l.append(row.name)
    plt.scatter(x, y)
    for i, label in enumerate(l):
        plt.annotate(l[i], (x[i], y[i]))
    plt.show()
 
def darwin(population):
    #Remove mutually non-dominating solutions
    ii = 0
    dellist = []
    for i in population:
            dominated = 0
            length = len(i) - 1
            #print(population[ii])
            #print(i)
            
            for y in population:
                ivar = [(i.loc[length].TotalDistance),(i.loc[length].DifficultyDiff)]
                yvar = [(y.loc[length].TotalDistance),(y.loc[length].DifficultyDiff)]
                if (ivar[0] >= yvar[0]) and (ivar[1] > yvar[1]):
                    dominated = 1
                #print(ivar)
                #print(yvar)
            if dominated == 1:
                dellist.append(ii)
                #print("DELETE")
            ii += 1
    #print(dellist)
    fitnesssuccess == len(dellist) - offspring
    for index in sorted(dellist,reverse=True):
        del population[index]
    
    return(population)
    
    
#Mutator/ Crossover
def MutatorCrossover(dataset):
    datasetcopy = dataset
    poplen = len(population)-1
    for i in range(offspring):
        parents = [random.randint(0, poplen),random.randint(0, poplen)]
        #print(parents)
        child = pd.DataFrame(columns = ['salesman','Stops','Distance','Difficulty','TotalDistance','DifficultyDiff'])
        parenta = datasetcopy[parents[0]]
        parentb = datasetcopy[parents[1]]
        adjustedmutationchance = mutationrating/(g*mutationcutoff)
        for i in range(0, salesmanno):
            child = child.append({'salesman' : (str(i)) , 'Stops' : ""}, ignore_index = True)
        crosscount = 0
        parenttocross = startparent
        adjustedcrossoverpattern = crossoverpattern
        if fitnesssuccess <= 0:
            adjustedcrossoverpattern = [crossoverpattern[0]*random.random()*crossovervariation,crossoverpattern[1]*random.random()*crossovervariation]
        for x in range(points):
            if parenttocross == 'a':
                location = findlocation(parenta, str(x))
                crosscount = crosscount + 1
                if crosscount == adjustedcrossoverpattern[0]:
                    parenttocross = 'b'
                    crosscount = 0
                mutatechance = random.random()
                if mutatechance < adjustedmutationchance:
                    location = random.randint(0, salesmanno-1)
                child.loc[location].Stops = child.loc[location].Stops + str(x) + " "
                
                #print("a" + str(x))
            elif parenttocross == 'b':
                location = findlocation(parentb, str(x))
                crosscount = crosscount + 1
                if crosscount == adjustedcrossoverpattern[1]:
                    parenttocross = 'a'
                    crosscount = 0
                
                mutatechance = random.random()
                if mutatechance < adjustedmutationchance:
                    location = random.randint(0, salesmanno-1)
                child.loc[location].Stops = child.loc[location].Stops + str(x) + " "
                #print(child.loc[location].Stops)
                #child.Stops.loc[location] = child.Stops.loc[location] + " " + str(x)
                #print("b" + str(x))
        count = 0
        for x in child.Stops:
            child.loc[count].Stops = list(child.loc[count].Stops.split(" "))
            child.loc[count].Stops = child.loc[count].Stops[:-1]
            count = count + 1
        
        #print(dataset)
        dataset.append(child)
        #print(dataset)
        #dataset = dataset.append(child)
        #print(child.Stops)
    #print("produced offspring...")
    return(dataset)
        #print(parenta.Stops)
        #print(location)
    #Parents to combinable form
    
    #combine parents to offspring
def findlocation(route, target):
    count = 0
    for i in route.Stops:
        try:
            i.index(target)
            #print(count)
            return(count)
        except:
            pass
        count += 1

def evalpopulation(pop):
    x = []
    y = []
    l = []
    c = 0
    for index in pop:
        x.append(index.loc[salesmanno-1].DifficultyDiff)
        y.append(index.loc[salesmanno-1].TotalDistance)
        l.append(str(c))
        c = c + 1
    plt.scatter(x, y)
    plt.xlabel('Difficulty Difference')
    plt.ylabel('Distance')
    for i, label in enumerate(l):
        plt.annotate(l[i], (x[i], y[i]))
    plt.show()
    
    #Produce Ratings:
    xy=[]
    c = 0
    for i in x:
        xy.append([x[c],y[c]])
        c = c + 1
    #print(xy)
    #print("")
#Main

salesman = generatesalesman()

dataset = generatedataset()

plotdataset(dataset)

distancematrix = createdistancematrix(dataset)

population = createpopulation(dataset)

population = calculatefitness(population, distancematrix, dataset, salesman)

population = darwin(population)

now = datetime.now()

current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)

#Main Loop
for g in range (1, generations):
    #print("")
    #print("Generation: "+ str(g))
    population = calculatefitness(population, distancematrix, dataset, salesman) 
    population = MutatorCrossover(population)
    population = darwin(population)
    
    #print("Mutually Non-dominated Solutions Removed...   " + str(len(population)-offspring) + " Remaining Solutions")
    
evalpopulation(population)
print("Completed")

now = datetime.now()

current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)


