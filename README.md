# PQB

Background:
Most of this code has been taken from the example Chicken & Waffle problem from the White Paper made by 
Polaris Quantum Biotech  regarding the usage of Quantum Computing  to solve a multivariable optimization to minimize 
both the calories and the price of a meal from a menu that they created. 

The constraints on this problem were that the CQM had to choose 1 item from each of the 5 categories: waffle, smear,
chicken, drizzle, and a side. The CQM also had to keep the total number of calories below 700 while minizming the price.

LINK TO PAPER: https://arxiv.org/pdf/2303.15419.pdf



Functionality: 
This code takes a set of origins and destinations that the user inputs into the POI_TABLE.csv file and computes the optimal method of transportation to take between each origin and destination. The code also computes the exact distance between the origin and destination using the Google Maps API, which comprises the first portion of the code, as well as rough estimates of the cost and time that the journey will take.
