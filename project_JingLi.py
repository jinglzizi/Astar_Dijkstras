'''
Compare Dijkstra's algorithm and A star algorithm in grid map which conatins different typies of terrain.
Author: Jing Li
Python version: 3.5.0
This script contain 4 sections:
   section 1. class: PriorityQueue, Graph, gridMap
   section 2. function: SearchforPath, reconstruct_path, heuristic_cost_estimate, plotresult
   section 3. Test: TestgridMap, TestOptimalPath
   section 4. Use classes, functions to set up, solve problem and plot results

'''

'''
Section 1: Classes

'''
class PriorityQueue():
    '''
    The arguments passed to a PriorityQueue must consist of objects that can be compared using <.
    Use a tuple (priority, item).
    This priority queue and binary heap is used in implementing Dijkstra's and A star algorithm
    '''

    def __init__(self):
        self._pq = []

    def _parent(self,n):
        return (n-1)//2

    def _leftchild(self,n):
        return 2*n + 1

    def _rightchild(self,n):
        return 2*n + 2

    def push(self, obj):
        # append at end and bubble up
        self._pq.append( obj )
        n = len(self._pq)
        self._bubble_up(n-1)

    def _bubble_up(self, index):
        while index>0:
            cur_item = self._pq[index]
            parent_idx = self._parent(index)
            parent_item = self._pq[parent_idx]
            
            if cur_item < parent_item:
                # swap with parent
                self._pq[parent_idx] = cur_item
                self._pq[index] = parent_item
                index = parent_idx
            else:
                break

    def pop(self):
        n = len(self._pq)
        if n==0:
            return None
        if n==1:
            return self._pq.pop()
        
        # replace with last item and sift down:
        obj = self._pq[0]
        self._pq[0] = self._pq.pop()
        self._sift_down(0)
        return obj

    def heapify(self, items):
        """ you can assume that the PQ is empty! """
        self._pq = items
        n=int(len(items)/2)
        for i in range(n+1):
            self._sift_down(n-i)
      
    
    def _sift_down(self,index):
        n = len(self._pq)
        
        while index<n:           
            cur_item = self._pq[index]
            lc = self._leftchild(index)
            if n <= lc:
                break

            # first set small child to left child:
            small_child_item = self._pq[lc]
            small_child_idx = lc
            
            # right exists and is smaller?
            rc = self._rightchild(index)
            if rc < n:
                r_item = self._pq[rc]
                if r_item < small_child_item:
                    # right child is smaller than left child:
                    small_child_item = r_item
                    small_child_idx = rc
            
            # done: we are smaller than both children:
            if cur_item <= small_child_item:
                break
            
            # swap with smallest child:
            self._pq[index] = small_child_item
            self._pq[small_child_idx] = cur_item
            
            # continue with smallest child:
            index = small_child_idx
        
    def size(self):
        return len(self._pq)
    
    def is_empty(self):
        return len(self._pq) == 0
        
    
    def decrease_priority(self, old, new):
        assert(new <= old)
        for i in range(self.size()):
            if self._pq[i]==old:
                self._pq[i]=new
                self._bubble_up(i)
                break
    

class Graph(object):
    '''Represents a graph'''

    def __init__(self, vertices, edges):
        '''A Graph is defined by its set of vertices
           and its set of edges.'''
        self.V = set(vertices) # The set of vertices
        self.E = set([])       # The set of edges
        self.neighbors = {}    # A dictionary that will hold the list
                               # of adjacent vertices for each vertex.
        self.vertex_coordinates = {}  # An optional dictionary that can hold coordinates
                               # for the vertices in grid map.
        self.vertex_value = {}  # a dictionary of value associated with vertex, which indicates
                                #different types of terrain.

        self.add_edges(edges)  # Note the call to add_edges will also
                               # update the neighbors dictionary
        print ('(Initializing a graph with %d vertices and %d edges)' % (len(self.V),len(self.E)))


    def add_vertices(self,vertex_list):
        ''' This method will add the vertices in the vertex_list
            to the set of vertices for this graph. Since V is a
            set, duplicate vertices will not be added to V. '''
        for v in vertex_list:
            self.V.add(v)
        self.build_neighbors()


    def add_edges(self,edge_list):
        ''' This method will add a list of edges to the graph
            It will insure that the vertices of each edge are
            included in the set of vertices (and not duplicated).
            It will also insure that edges are added to the
            list of edges and not duplicated. '''
        for s,t in edge_list:
            if (s,t) not in self.E and (t,s) not in self.E:
                self.V.add(s)
                self.V.add(t)
                self.E.add((s,t))
        self.build_neighbors()


    def build_neighbors(self):
        self.neighbors = {}
        for v in self.V:
            self.neighbors[v] = []
        for e in self.E:
            s,t = e
            self.neighbors[s].append(t)
            self.neighbors[t].append(s)





class gridMap(Graph):
    '''
    Generate a gird map which contain information about vertices, weights, edges
    coordinates, and types of terrain.


    '''
    def __init__(self,mrow,ncol, matA, direction4=True):   
        '''
        augments passed to gridMap:
          mrow: number of row in grid map
          ncol: number of column in grid map
          matA: a matrix which contains value 0, 1, 3, 5. 
                where 0 indicates obstacle, 1 indicates flat land, 3 indicates jungle, 5 indicates mountains.
          direction4: True indicates that in grid map it can move toward 4 dirctions(east, west, north and south).
                      False indicates that in grid map it can move toward 8 directions(east, west, 
                      north, south, southest, southwest, northeast, northeast)

        '''
        V=set() # a set of vertices extracted from matA
        W={} # a dictionary of weights for edges.
        value={} # the value associated with vertex indicative of type of terrain
        coord={} # a dictionary of coordinates in grip map
        visited={}

        gap=1
    
        for m in range(mrow):
            for n in range(ncol):
                v='m{0}n{1}'.format(m,n)
                V.add(v)
                value[v]=matA[m][n]
                coord[v]=(gap/2+n, gap/2+m)
        for v in V:
            visited[v]=False


        for m in range(mrow):
            for n in range(ncol):
                v='m{0}n{1}'.format(m,n)
                if visited[v]==False:
                    visited[v]=True

                    if value[v]!=0:
                        if m>0 and value['m{0}n{1}'.format(m-1,n)]!=0 and visited['m{0}n{1}'.format(m-1,n)]==False:
                            W[(v,'m{0}n{1}'.format(m-1,n))]=0.5*(value['m{0}n{1}'.format(m-1,n)]+value['m{0}n{1}'.format(m,n)])

                        if m<mrow-1 and value['m{0}n{1}'.format(m+1,n)]!=0 and visited['m{0}n{1}'.format(m+1,n)]==False:
                            W[(v,'m{0}n{1}'.format(m+1,n))]=0.5*(value['m{0}n{1}'.format(m+1,n)]+value['m{0}n{1}'.format(m,n)])


                        if n>0 and value['m{0}n{1}'.format(m,n-1)]!=0 and visited['m{0}n{1}'.format(m,n-1)]==False:
                            W[(v,'m{0}n{1}'.format(m,n-1))]=0.5*(value['m{0}n{1}'.format(m,n-1)]+value['m{0}n{1}'.format(m,n)])


                        if n<ncol-1 and value['m{0}n{1}'.format(m,n+1)]!=0 and visited['m{0}n{1}'.format(m,n+1)]==False:
                            W[(v,'m{0}n{1}'.format(m,n+1))]=0.5*(value['m{0}n{1}'.format(m,n+1)]+value['m{0}n{1}'.format(m,n)])

    
                        if not direction4==True:
                            if m>0 and n>0:
                                if value['m{0}n{1}'.format(m-1,n-1)]!=0 and visited['m{0}n{1}'.format(m-1,n-1)]==False:
                                    W[(v,'m{0}n{1}'.format(m-1,n-1))]=(value['m{0}n{1}'.format(m-1,n-1)]**2+value['m{0}n{1}'.format(m,n)]**2)**0.5

                            if m<mrow-1 and n<ncol-1:
                                if value['m{0}n{1}'.format(m+1,n+1)]!=0 and visited['m{0}n{1}'.format(m+1,n+1)]==False:
                                    W[(v,'m{0}n{1}'.format(m+1,n+1))]=(value['m{0}n{1}'.format(m+1,n+1)]**2+value['m{0}n{1}'.format(m,n)]**2)**0.5

                            if m>0 and n<ncol-1:
                                if value['m{0}n{1}'.format(m-1,n+1)]!=0 and visited['m{0}n{1}'.format(m-1,n+1)]==False:
                                    W[(v,'m{0}n{1}'.format(m-1,n+1))]=(value['m{0}n{1}'.format(m-1,n+1)]**2+value['m{0}n{1}'.format(m,n)]**2)**0.5

                            if m<mrow-1 and n>0:
                                if value['m{0}n{1}'.format(m+1,n-1)]!=0 and visited['m{0}n{1}'.format(m+1,n-1)]==False:
                                    W[(v,'m{0}n{1}'.format(m+1,n-1))]=(value['m{0}n{1}'.format(m+1,n-1)]**2+value['m{0}n{1}'.format(m,n)]**2)**0.5

        edges = []
        for e1,e2 in W:
            edges.append((e1,e2))
        
        Graph.__init__(self, V, edges)
        self.weights = {}
        for e1,e2 in W:
            weight = W[(e1,e2)]
            self.weights[(e1,e2)] = weight
            self.weights[(e2,e1)] = weight
        self.vertex_value = value
        self.vertex_coordinates=coord


'''
Step 2: functions

'''
def SearchforPath(gridMap, start, goal, D, D2, direction4=True):
    '''
    This function is used to search for optimal path with Dijkstra's or A star.
    Augments passed to this function:
       gridMap: a map built by class gridMap
       start: start point
       goal: target point
       D, D2: if D=D2=0, this method use Manhantan distance for heuristic function (suitable for 4 direcitions).
              if D=D2=1, this method use chebyshev distance for heuristic function (suitable for 8 direcitions).
              if D=1, D2=sqrt(2), this method use octile distance for heuristic fuction (suitable for 8 direcitions).
       direction4: True indicates that in grid map it can move toward 4 dirctions(east, west, north and south).
                   False indicates that in grid map it can move toward 8 directions(east, west, 
                      north, south, southest, southwest, northeast, northeast)

    '''

    from collections import deque 
    closedSet=set()
    Q=PriorityQueue()
    prev={}
    gScore={}
    fScore={}
    
    
    for node in gridMap.V:
        gScore[node]=float('inf')
        fScore[node]=float('inf')
    
    gScore[start]=0
    fScore[start]=heuristic_cost_estimate(start,goal,gridMap,D,D2,direction4)


    Q.heapify([(v, k) for k, v in fScore.items()])

    while not Q.is_empty():
        value, current=Q.pop()
        
        if current==goal:
            print("steps:", len(closedSet))
            return reconstruct_path(prev, current), closedSet
        
        closedSet.add(current)
        for node in gridMap.neighbors[current]:
            if node in closedSet:
                continue
                   
            new_gScore=gridMap.weights[(node,current)]+value-heuristic_cost_estimate(current,goal,gridMap,D,D2,direction4)

            if  new_gScore>=gScore[node]:
                continue
           
            prev[node]=current
            gScore[node] = new_gScore
            oldfScore=fScore[node]
            fScore[node] = gScore[node] + heuristic_cost_estimate(node, goal,gridMap,D,D2, direction4)
            Q.decrease_priority((oldfScore,node),(fScore[node],node))


    return -1

def reconstruct_path(prev, current):
    '''
     This function is used to construct the optimal path based on results from function SearchforPath.

    '''
    total_path = [current]
    while current in prev.keys():
        current = prev[current]
        total_path.insert(0,current)
    return total_path


def heuristic_cost_estimate(node, goal, gridMap, D, D2, direction4=True):
    '''
     This function is used to estimate the distance between the current node and target point.
      Augments passed to this function:
       
       gridMap: a map built by class gridMap
       
       node: current node
       
       goal: target point
       
       D, D2: If the user want to implement Dijkstra’s algorithm, set D=D2=0 (no heuristic function used).
              If the user want to implement A star algorithm, set D=D2=1 (That is, set Chebyshev distance 
              for heuristic function for 8 directions, or set Mahantan distance for 4 directions), 
              or set D=1 and D2=sqrt(2) (That is, set octile distance for heuristic function for 8 directions).
       
       direction4: True indicates that in grid map it can move toward 4 dirctions(east, west, north and south).
                   False indicates that in grid map it can move toward 8 directions(east, west, 
                      north, south, southest, southwest, northeast, northeast)
    '''
    if D==0 and D2==0:
        return 0


    nodex,nodey=gridMap.vertex_coordinates[node]
    goalx,goaly=gridMap.vertex_coordinates[goal]
    dx = abs(nodex - goalx)
    dy = abs(nodey - goaly)


    if direction4==True:
        dist=D*(dx+dy)
    else:
        dist=D*(dx+dy)+(D2-2*D)*min(dx,dy)
    return dist
    


def plotresult(ncol,mrow, map, A_path, A_search, Dij_path, Dij_search):
    '''
    This function is used to plot the different types of terrain, start point, target point,
    optimal pathes, closed set got by A star and Dijkstra's, 

    '''
    import numpy as np
    from matplotlib import pyplot as plt
    from matplotlib import animation
    from matplotlib.ticker import MultipleLocator


    fig = plt.figure(1)
    ax = plt.axes(xlim=(0, ncol), ylim=(0, mrow))

    spacing = 1 # This can be your user specified spacing. 
    minorLocator = MultipleLocator(spacing)
    ax.yaxis.set_minor_locator(minorLocator)
    ax.xaxis.set_minor_locator(minorLocator)                                                    
  
    ax.grid(which="minor",color='black', linestyle='-', linewidth=2)

    
    for v in map.V:
        if map.vertex_value[v]==0:
            x,y=map.vertex_coordinates[v]
            ax.axvspan(xmin=x-1/2,xmax=x+1/2,ymin=(y-1/2)/mrow, ymax=(y+1/2)/mrow, facecolor='black', alpha=1)

        if map.vertex_value[v]==3:
            x,y=map.vertex_coordinates[v]
            ax.scatter(x, y, color="green",marker='|', alpha=1, linewidths=15)

        if map.vertex_value[v]==5:
            x,y=map.vertex_coordinates[v]
            ax.scatter(x, y, color="green",marker='1', alpha=1, linewidths=15)


        x_start,y_start=map.vertex_coordinates[start]
        ax.axvspan(xmin=x_start-1/2,xmax=x_start+1/2,ymin=(y_start-1/2)/mrow, ymax=(y_start+1/2)/mrow, facecolor='red', alpha=1)

        x_goal,y_goal=map.vertex_coordinates[goal]
        ax.axvspan(xmin=x_goal-1/2,xmax=x_goal+1/2,ymin=(y_goal-1/2)/mrow, ymax=(y_goal+1/2)/mrow, facecolor='green', alpha=1)


    for item in Dij_search:
        x,y=map.vertex_coordinates[item]
        ax.axvspan(xmin=x-1/2,xmax=x+1/2,ymin=(y-1/2)/mrow, ymax=(y+1/2)/mrow, facecolor='yellow', alpha=0.3)
    

    for item in Dij_path:
        x_path,y_path=map.vertex_coordinates[item]
        ax.scatter(x_path, y_path, color="blue",marker='o', alpha=1, linewidths=10)

    fig2 = plt.figure(2)

    ax = plt.axes(xlim=(0, ncol), ylim=(0, mrow))

    spacing = 1 # This can be your user specified spacing. 
    minorLocator = MultipleLocator(spacing)
    ax.yaxis.set_minor_locator(minorLocator)
    ax.xaxis.set_minor_locator(minorLocator)                                                    
  
    ax.grid(which="minor",color='black', linestyle='-', linewidth=2)

    
    for v in map.V:
        if map.vertex_value[v]==0:
            x,y=map.vertex_coordinates[v]
            ax.axvspan(xmin=x-1/2,xmax=x+1/2,ymin=(y-1/2)/mrow, ymax=(y+1/2)/mrow, facecolor='black', alpha=1)

        if map.vertex_value[v]==3:
            x,y=map.vertex_coordinates[v]
            ax.scatter(x, y, color="green",marker='|', alpha=1, linewidths=15)

        if map.vertex_value[v]==5:
            x,y=map.vertex_coordinates[v]
            ax.scatter(x, y, color="green",marker='1', alpha=1, linewidths=15)


        x_start,y_start=map.vertex_coordinates[start]
        ax.axvspan(xmin=x_start-1/2,xmax=x_start+1/2,ymin=(y_start-1/2)/mrow, ymax=(y_start+1/2)/mrow, facecolor='red', alpha=1)

        x_goal,y_goal=map.vertex_coordinates[goal]
        ax.axvspan(xmin=x_goal-1/2,xmax=x_goal+1/2,ymin=(y_goal-1/2)/mrow, ymax=(y_goal+1/2)/mrow, facecolor='green', alpha=1)


    for item in A_search:
        x,y=map.vertex_coordinates[item]
        ax.axvspan(xmin=x-1/2,xmax=x+1/2,ymin=(y-1/2)/mrow, ymax=(y+1/2)/mrow, facecolor='yellow', alpha=0.3)
    

    for item in A_path:
        x_path,y_path=map.vertex_coordinates[item]
        ax.scatter(x_path, y_path, color="blue",marker='o', alpha=1, linewidths=10)    
    
    plt.show()

    
'''
Section 3: Tests

'''

import unittest, sys
import numpy as np


class TestgridMap(unittest.TestCase):
    '''
    This test is used to confirm that the map generated by class gridMap has the following features:
         1. the types of terrain in gridMap mactches the values(0,1,3,5) associated with vertrices
         2. gridMap can establish the right weights associated with edges for 4 directions and 8 driections.

    '''
   
    def test_SimpleMap4(self):
        matA=np.ones((3,3))
        G1 = gridMap(3, 3, matA)

        V=set()
        for m in range(3):
            for n in range(3):
                V.add("m{0}n{1}".format(m,n))

        self.assertEqual(G1.V, V)
        self.assertEqual(len(G1.E), 12)

    def test_SimpleMap8(self): 
        matA=np.ones((3,3))
        
        G2 = gridMap(3, 3, matA, False)
        self.assertEqual(len(G2.V), 9)
        self.assertEqual(len(G2.E), 20)
        self.assertEqual(len(G2.weights), 40)


    def test_Map4(self): 
        matA=np.ones((3,3))
        matA[0,0]=3
        matA[1,1]=0
        matA[2,2]=5
        
        G3 = gridMap(3, 3, matA)
        self.assertEqual(len(G3.V), 9)
        self.assertEqual(len(G3.E), 8)
        self.assertEqual(len(G3.weights), 16)
        self.assertEqual(G3.weights[('m2n1','m2n2')], 3)
        self.assertEqual(G3.weights[('m0n0','m0n1')], 2)

    
    def test_Map8(self): 
        matA=np.ones((3,3))
        matA[0,0]=3
        matA[1,1]=0
        matA[1,2]=5
        
        G4 = gridMap(3, 3, matA, False)
        self.assertEqual(len(G4.V), 9)
        self.assertEqual(len(G4.E), 12)
        self.assertEqual(len(G4.weights), 24)
        self.assertEqual(G4.weights[('m1n2','m2n2')], 3)
        self.assertEqual(G4.weights[('m0n0','m0n1')], 2)
        self.assertEqual(G4.weights[('m0n0','m0n1')], 2)
        self.assertEqual(G4.weights[('m0n1','m1n2')], 26**0.5)

suite = unittest.TestLoader().loadTestsFromTestCase(TestgridMap)
unittest.TextTestRunner().run(suite)


        

class TestOptimalPath(unittest.TestCase):
    '''
    This method is to confirm that the A start with appropriate heuristic function can find optimal
    path as Dijkstra's does.

    '''
    def Test_DistMatch4(self):
        matA=np.ones((10,10))
        matA[5:8,5]=0
        G = gridMap(10, 10, matA)
        A_path, A_search=SearchforPath(G, start, goal, 1, 0)
        Dij_path, Dij_search=SearchforPath(G, start, goal, 0,0)

        n=len(A_path)
        for i in range(n-1):
            A_distance+=G.weights[(A_path[i],path[i+1])]

        m=len(Dij_path)
        for i in range(m-1):
            Dij_distance+=G.weights[(A_path[i],path[i+1])]

        self.assertEqual(A_distance, Dij_distance)

    def Test_DistMatch8(self):
        matA=np.ones((10,10))
        matA[5:8,5]=0
        G = gridMap(10, 10, matA, False)
        A_path, A_search=SearchforPath(G, start, goal, 1, 1, False)
        Dij_path, Dij_search=SearchforPath(G, start, goal, 0,0, False)

        n=len(A_path)
        for i in range(n-1):
            A_distance+=G.weights[(A_path[i],path[i+1])]

        m=len(Dij_path)
        for i in range(m-1):
            Dij_distance+=G.weights[(A_path[i],path[i+1])]

        self.assertEqual(A_distance, Dij_distance)
    
suite2 = unittest.TestLoader().loadTestsFromTestCase(TestOptimalPath)
unittest.TextTestRunner().run(suite2)




'''
section 4: set up, solve problems, and plot results
'''

##  Four direction examples

## Map 1.1: No obstacle, horizontal position
mrow=15
ncol=15
matA=np.ones((mrow,ncol)) 

start="m7n3" #start position
goal="m7n12" #end position

G1=gridMap(mrow, ncol, matA, True)
A_path, A_search=SearchforPath(G1, start, goal, 1,0, True)
Dij_path, Dij_search=SearchforPath(G1, start, goal, 0,0, True)
plotresult(ncol,mrow, G1, A_path, A_search, Dij_path, Dij_search)

## Map 1.2: No obstacle, diagonal position
mrow=15
ncol=15
matA=np.ones((mrow,ncol))

start="m9n9"
goal="m14n14"

G2=gridMap(mrow, ncol, matA, True)
A_path, A_search=SearchforPath(G2, start, goal, 1,0, True)
Dij_path, Dij_search=SearchforPath(G2, start, goal, 0,0, True)
plotresult(ncol,mrow, G2, A_path, A_search, Dij_path, Dij_search)

## Map 2: With obstacle, obstacles around goal position
mrow=30
ncol=30
matA=np.ones((mrow,ncol))

start="m9n22"
goal="m24n6"
matA[26,4:9]=0 #set obstacles
matA[22,4:9]=0
matA[23:26,8]=0

G3=gridMap(mrow, ncol, matA, True)
A_path, A_search=SearchforPath(G3, start, goal, 1,0, True)
Dij_path, Dij_search=SearchforPath(G3, start, goal, 0,0, True)
plotresult(ncol,mrow, G3, A_path, A_search, Dij_path, Dij_search)

## Map 3: Maze
mrow=13
ncol=13
matA=np.ones((mrow,ncol))
matA[5,5:8]=0
matA[6,5]=0
matA[7,5:10]=0
matA[4:7,9]=0
matA[3,4:10]=0
matA[3:10,3]=0
matA[9,3:12]=0
matA[1:10,11]=0
matA[1,2:11]=0
matA[1:12,1]=0
matA[11,2:12]=0
matA[11,12]=0

start="m6n6"
goal="m12n12"

G31=gridMap(mrow, ncol, matA, True)
A_path, A_search=SearchforPath(G31, start, goal, 1,0, True)
Dij_path, Dij_search=SearchforPath(G31, start, goal, 0,0, True)
plotresult(ncol,mrow, G31, A_path, A_search, Dij_path, Dij_search)

## Eight Directions examples

## Map 1.2: No obstacle, horizontal level
mrow=15
ncol=15
matA=np.ones((mrow,ncol))

start="m9n9"
goal="m14n14"

G4=gridMap(mrow, ncol, matA, False)
A_path, A_search=SearchforPath(G4, start, goal, 1,1, False)
#A_path, A_search=SearchforPath(G4, start, goal, 1,2**0.05, False)
Dij_path, Dij_search=SearchforPath(G4, start, goal, 0,0, False)
plotresult(ncol,mrow, G4, A_path, A_search, Dij_path, Dij_search)

## Map 2: With obstacle, bounded around the end position
mrow=30
ncol=30
matA=np.ones((mrow,ncol))

start="m9n22"
goal="m24n6"
matA[26,5:9]=0
matA[22,5:9]=0
matA[23:26,8]=0

G5=gridMap(mrow, ncol, matA, False)
#A_path, A_search=SearchforPath(G5, start, goal, 1,1, False) #chebyshev direction
A_path, A_search=SearchforPath(G5, start, goal, 1,2**0.05, False) #octile position
Dij_path, Dij_search=SearchforPath(G5, start, goal, 0,0, False)
plotresult(ncol,mrow, G5, A_path, A_search, Dij_path, Dij_search)

## Map 3: maze
mrow=13
ncol=13
## walls
matA=np.ones((mrow,ncol))
matA[5,5:8]=0
matA[6,5]=0
matA[7,5:10]=0
matA[4:7,9]=0
matA[3,4:10]=0
matA[3:10,3]=0
matA[9,3:12]=0
matA[1:10,11]=0
matA[1,2:11]=0
matA[1:12,1]=0
matA[11,2:12]=0
matA[11,12]=0

start="m6n6"
goal="m12n12"

G6=gridMap(mrow, ncol, matA, False)
A_path, A_search=SearchforPath(G6, start, goal, 1,1, False)
#A_path, A_search=SearchforPath(G6, start, goal, 1,2**0.05, False) 
Dij_path, Dij_search=SearchforPath(G6, start, goal, 0,0, False)
plotresult(ncol,mrow, G6, A_path, A_search, Dij_path, Dij_search)

## More discussion: consider jungles and mountain
## four directions
mrow=30
ncol=30
matA=np.ones((mrow,ncol))
matA[9:13,4:7]=0
matA[0:5,0:4]=5
matA[24:30,0:3]=5
matA[4:28,8:16]=5
matA[6:26,7]=5
matA[6:26,16]=5
matA[5:25,6]=3
matA[5:25,17:20]=3

start="m20n4"
goal="m13n20"

G7=gridMap(mrow, ncol, matA, True)
A_path, A_search=SearchforPath(G7, start, goal, 10,0, True)
Dij_path, Dij_search=SearchforPath(G7, start, goal, 0,0, True)
plotresult(ncol,mrow, G7, A_path, A_search, Dij_path, Dij_search)

## eight directions
mrow=30
ncol=30
matA=np.ones((mrow,ncol))
matA[9:13,4:7]=0
matA[0:5,0:4]=5
matA[24:30,0:3]=5
matA[4:28,8:16]=5
matA[6:26,7]=5
matA[6:26,16]=5
matA[5:25,6]=3
matA[5:25,17:20]=3

start="m20n4"
goal="m13n20"

G8=gridMap(mrow, ncol, matA, False)
A_path, A_search=SearchforPath(G8, start, goal, 10,1, False)
#A_path, A_search=SearchforPath(G8, start, goal, 1,2**0.05, False)
Dij_path, Dij_search=SearchforPath(G8, start, goal, 0,0, False)
plotresult(ncol,mrow, G8, A_path, A_search, Dij_path, Dij_search)


