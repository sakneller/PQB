# Internship Background:
This is the code I created during my internship at Polaris Quantum Biotech, where I learned how to program on DWave’s Quantum Annealers. The Internship lasted five weeks from 7/24 to 8/26 2023. We set out trying to make a program that could determine the best path in a network of points. We started by putting in small maps that we could solve by hand. The first map only had four nodes and featured one mode of transportation: walking, show in green. The only thing that determined the best path between two points in this map was the length of the path.

![Map_1](https://github.com/sakneller/PQB/assets/72467352/6949d6ed-fa2b-4d58-a636-100dfc1c19db)


The second map featured nine nodes and introduced buses, shown in blue. These acted as an express service between points in the network, but they had a slightly longer length due to having to conform to the imaginary road network. The buses also required the introduction of two new variables: cost and time. The cost of buses was higher than that of walking, which is free. However, the time it took for buses to get to nodes was much shorter than walking. These factors resulted in buses becoming the optimal mode of transportation to take for longer journeys, as walking would take too long. However, walking became the optimal mode for shorter journeys, since the cost of taking the bus would be unjustified. Notably, Not every node on the network had a bus stop, since buses predominantly run on large roads, meaning that walking was still sometimes necessary. 

![Map_2](https://github.com/sakneller/PQB/assets/72467352/a949fc79-c7f6-44d0-888e-b213ed34bf58)


The Third Map features five nodes and introduces two new modes of transportation: driving and the train, and one new variable: environmental cost. Driving was marginally faster than taking the bus, and their paths were both the same length, since they both have to drive on roads. The driving and bus paths also stopped at the same nodes as the buses, which I attributed to traffic lights on the main road. However, driving’s slight time benefit is overshadowed by its high environmental cost, which I measured in pounds of CO2 released per mile. The train is naturally the fastest option of the four outlined, but only stops at two nodes on the opposite side of the network, one less stop than both the bus and the car. The length for the train is the longest, which I attributed to the fact that riders must sometimes go out of their way to get to a train station, and trains themselves sometimes take detours to avoid densely populated areas.

![Map_3](https://github.com/sakneller/PQB/assets/72467352/38ff60f8-0d0f-4183-9a69-5dd1825b5607)


Link to all maps: https://tennessine.co.uk/metro/99d23155cbadb6f



# Quantum Computers & Annealers:
A quantum computer is a type of computer that uses quantum mechanics in order to perform calculations faster than a regular computer. One of the reasons they can do this is due to a phenomenon in quantum mechanics known as superposition. This means that a qubit, the equivalent of a transistor for a quantum computer, can be either a one or zero at any given moment or a combination of all variations of states at the same time. An example of this is the Schrodinger’s Cat thought experiment, in which a cat is considered both alive and dead at the same time. Qubits allow much more information to be processed than a traditional computer, meaning they can solve problems at a much faster speed than traditional computers. Therefore, quantum computers are excellent at solving multivariable problems. An annealer is a special type of quantum computer that is best at solving optimization problems, such as the multivariable transportation network optimization problem we used it to solve. 



# Constraints & Objectives:
The objective is the column set to be minimized or maximized in the CQM. In this problem, my objective column was the time of the overall trip, and it was minimized. Constraints are other parameters that the CQM must follow. In my code, the constraints are that the total environmental cost, in terms of pounds of CO2 released into the atmosphere, must be less than or equal to one hundred pounds, and that the total cost must be less than or equal to one hundred dollars. A one-hot constraint is a special type of constraint that states that only one item in each category may be chosen. In the previously mentioned white paper, the one hot constraint was used to ensure that 1 menu item from each of the categories was chosen. In my code, I used it to ensure that only one leg for each trip was chosen. A general format of how the CQM formats these objectives and constraints can be seen below. X1, X2, X3, etc. refer to binary variables that will be either one or zero depending on the solution chosen. E1, E2, E3, etc. all refer to the environmental cost column’s value for that specific row, so E1 would refer to the environmental cost for the first row, and so on. T1, T2, T3, etc. refer to the time column’s value for that specific row, so T1 similarly refers to the time value for the first row. Interestingly, one-hot constraints only use binary variables, so if X1 is chosen it will be one, since one squared is one and if X2 is not chosen it will be zero, since zero squared is zero.


Objective: Minimize Env. Cost
Obj=i(Xi x Ei)= (X1 x E1) + (X2 x E2) + (X3 x E3) + (X4 x E4) + (X5 x E5) + (X6 x E6) + (X7 x E7) + (X8 x E8) 

Constraint: Time less than 1 hour:
(X1 x T1) + (X2 x T2) + (X3 x T3) + (X4 x T4) + (X5 x T5) + (X6 x T6) + (X7 x T7) + (X8 x T8) <= 60.


One-Hot Constraint: One leg chosen for each trip
(X1 x X1) + (X2 x X2) + (X3 x X3) + (X4 x X4) = 1.



# Code:
The beginning portion of the my program, titled Solving_Transportation_Network.py, was taken from the Distance Matrix program in the Google Maps API repository. This program contains the functions to calculate the distance between two points on google maps, called origins and destinations in the code, using each of the four modes of transportation available on google maps: walking, bicycling, driving, and transit.

The second part of the code was taken from the Chicken & Waffle problem from the White Paper made by Polaris Quantum Biotech regarding the usage of their quantum computing algorithm to solve a multivariable optimization problem which minimized both the calories and the price of a meal from a menu that they created. The constraints on this problem were that the CQM had to choose 1 item from each of the 5 categories: waffle, smear, chicken, drizzle, and a side. The CQM also had to keep the total number of calories below 700 while minimizing the price. 

Link to the white paper: https://arxiv.org/pdf/2303.15419.pdf

Link to white paper’s github repository: https://github.com/pqb-mb/pqb-cqm-example

Link to Google API: https://developers.google.com/maps/documentation/distance-matrix/overview

Link to the code that exemplify sending requests to the Google Maps Distance Matrix API: https://github.com/googlemaps/google-maps-services-python/blob/645e07de5a27c4c858b2c0673f0dd6f23ca62d28/googlemaps/distance_matrix.py#L23



# Functionality: 
This code takes a set of origins and destinations that the user inputs into the POI_TABLE.csv file and computes the optimal method of transportation to take between each origin and destination. The code also computes the exact distance between the origin and destination using the Google Maps API, which comprises the first portion of the code, as well as rough estimates of the cost and time that the journey will take. 

For guidance on how exactly to format inputs into the POI_TABLE.csv file, check the FORMAT_GUIDE.txt file.



# Statistics:
During the course of my internship, I submitted 132 problems to DWave’s Quantum Annealers. These problems took approximately 10.33 minutes, or 10 minutes and 20 seconds, for the Annealers to solve.

![Dwave_data](https://github.com/sakneller/PQB/assets/72467352/f5725a17-e3d4-4846-81de-facc4aea7945)




In the final 4 days of the internship that I had access to the Google Maps Distance Matrix code I submitted approximately 400 API requests per day for the distance between an origin and destination.

![Google_Maps_API_Data](https://github.com/sakneller/PQB/assets/72467352/0233ae2d-2c3a-4503-b228-ff54c6c1e2cf)



# Potential Improvements & Reflections:
In the future, a user could modify the code to allow a custom input file using the correct format to be defined, and a custom output file and location to be defined as well. 
