Astar_Dijkstras

Author: Jing Li

Compare Dijkstra's algorithm and A star algorithm in grid map which conatins different typies of terrain.
Python version: 3.5.0
This script contain 4 sections:
   
    section 1. class: PriorityQueue, Graph, gridMap
    section 2. function: SearchforPath, reconstruct_path, heuristic_cost_estimate, plotresult
    section 3. Test: TestgridMap, TestOptimalPath
    section 4. Use classes, functions to set up, solve problem and plot results
   
How to set up a grid map and use the two algorithm:
    
1. Set up.
    Generate a gird map which contain information about vertices, weights, edges
    coordinates, and types of terrain by class gridMap
         
          Augments passed to gridMap:
          mrow: number of row in grid map
          ncol: number of column in grid map
          matA: a matrix which contains value 0, 1, 3, 5. 
                where 0 indicates obstacle, 1 indicates flat land, 3 indicates jungle, 5 indicates mountains.
          direction4: True indicates that in grid map it can move toward 4 dirctions(east, west, north and south).
                      False indicates that in grid map it can move toward 8 directions(east, west, 
                      north, south, southest, southwest, northeast, northeast).
                      
  
      
2. Given a start point and a end point, search for shortest path by the two algorithm.function SearchforPath is 
     used to search for optimal path with Dijkstra's or A star.
         
         
         Augments passed to this function: 
         gridMap: a map built by class gridMap 
         start: start point 
         goal: target point 
         D, D2: if D=D2=0, this method use Manhantan distance for heuristic function (suitable for 4 direcitions). 
                if D=D2=1, this method use chebyshev distance for heuristic function (suitable for 8 direcitions). 
                if D=1, D2=sqrt(2), this method use octile distance for heuristic fuction (suitable for 8 direcitions). 
         direction4: True indicates that in grid map it can move toward 4 dirctions(east, west, north and south). 
                     False indicates that in grid map it can move toward 8 directions(east, west, north, south, southest, southwest,                                          northeast, northeast).
                      
                      
3. visualize the results. function plotresults is used to plot the results.
