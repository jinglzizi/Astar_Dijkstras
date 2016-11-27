
'''Python version: 3.5.0
Author: Jing Li
Problem: Compare A* algorithm and Dijkstra's algorithm in sovling path-finding problems in grid map.
Note: use binary heap when implement the two algorithm.
Steps: 1. build PriortyQueue, Graph, Network.
       2. build function to implement A* algorithm and Dijkstra's algorithm.
       3. build function to generate grid map according to different problems.
       4. Plot results.'''

'''
Step 1: build classes for Priority Queue, Graph, Network
'''
class PriorityQueue():

    def __init__(self):
        self._pq = []

    def _parent(self,n):
        return (n-1)//2

    def _leftchild(self,n):
        return 2*n + 1

    def _rightchild(self,n):
        return 2*n + 2

    def push(self, obj):
        self._pq.append( obj )
        n = len(self._pq)
        self._bubble_up(n-1)

    def _bubble_up(self, index):
        while index>0:
            cur_item = self._pq[index]
            parent_idx = self._parent(index)
            parent_item = self._pq[parent_idx]
            
            if cur_item < parent_item:
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
        
        obj = self._pq[0]
        self._pq[0] = self._pq.pop()
        self._sift_down(0)
        return obj

    def heapify(self, items):
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

            small_child_item = self._pq[lc]
            small_child_idx = lc
            
            rc = self._rightchild(index)
            if rc < n:
                r_item = self._pq[rc]
                if r_item < small_child_item:
                    # right child is smaller than left child:
                    small_child_item = r_item
                    small_child_idx = rc
            
            if cur_item <= small_child_item:
                break
            
            self._pq[index] = small_child_item
            self._pq[small_child_idx] = cur_item
            
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
        self.neighbors = {}          # A dictionary that will hold the list
                               # of adjacent vertices for each vertex.
        self.vertex_coordinates = {}       # An optional dictionary that can hold coordinates
                               # for the vertices.
        self.edge_labels = {}  # a dictionary of labels for edges

        self.add_edges(edges)  

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




class Network(Graph):    
    def __init__(self, vertices, edge_weights):
        ''' Initialize the network with a list of vertices
        and weights (a dictionary with keys (E1, E2) and values are the weights)'''

        edges = []
        for e1,e2 in edge_weights:
            edges.append((e1,e2))
        
        Graph.__init__(self, vertices, edges)
        self.weights = {}
        for e1,e2 in edge_weights:
            weight = edge_weights[(e1,e2)]
            self.weights[(e1,e2)] = weight
            self.weights[(e2,e1)] = weight
        self.edge_labels = self.weights


'''
Step 2: Build functions for Dijstra's and A*stars

'''

def SearchforPath(network, start, goal, D, direction4=True):
    from collections import deque 
    closedSet=set()
    Q=PriorityQueue()
    prev={}
    gScore={}
    fScore={}
    
    
    for node in network.V:
        gScore[node]=float('inf')
        fScore[node]=float('inf')
    
    gScore[start]=0
    fScore[start]=heuristic_cost_estimate(start,goal,network,D,direction4)


    Q.heapify([(v, k) for k, v in fScore.items()])

    while not Q.is_empty():
        value, current=Q.pop()
        
        if current==goal:
           
            return reconstruct_path(prev, current), closedSet
        
        closedSet.add(current)
        for node in network.neighbors[current]:
            if node in closedSet:
                continue

            new_gScore=network.weights[(node,current)]+value-heuristic_cost_estimate(current,goal,network,D,direction4)

            if  new_gScore>=gScore[node]:
                continue
           
            prev[node]=current
            gScore[node] = new_gScore
            oldfScore=fScore[node]
            fScore[node] = gScore[node] + heuristic_cost_estimate(node, goal,network,D, direction4)
            Q.decrease_priority((oldfScore,node),(fScore[node],node))


    return -1

def reconstruct_path(prev, current):
    total_path = [current]
    while current in prev.keys():
        current = prev[current]
        total_path.insert(0,current)
    return total_path


def heuristic_cost_estimate(node, goal, network, D, direction4=True):
    nodex,nodey=network.vertex_coordinates[node]
    goalx,goaly=network.vertex_coordinates[goal]
    dx = abs(nodex - goalx)
    dy = abs(nodey - goaly)
    if direction4==True:
        dist=D*(dx+dy)
    else:
        dist=D*(dx**2+dy**2)**0.5
    return dist
    
    
'''
Step 3: Setup a grid map

'''
def generateMap(mrow, ncol, matA, direction4=True):
    V=set()
    W={}
    value={}
    coord={}
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



    return V, W, coord, value



'''
 example

'''
import numpy as np

mrow=30
ncol=30
matA=np.ones((mrow,ncol))
matA[2:5,0:25]=2
matA[15:25,15:25]=3
matA[12:18,25:30]=3


start="m0n0" #Start Point
goal="m20n25" #Target Point



V,W,coord,value=generateMap(mrow, ncol, matA, False)
G=Network(V,W)
G.vertex_coordinates=coord



A_path, A_search=SearchforPath(G, start, goal, 1, False)
Dij_path, Dij_search=SearchforPath(G, start, goal, 0, False)


def computeDist(path, network):
    n=len(path)
    if n<2:
        return 0

    distance=0
    for i in range(n-1):
        distance+=network.weights[(path[i],path[i+1])]
    return distance

#print(computeDist(A_path, G))
#print(computeDist(Dij_path,G))


'''
Plot

'''

from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.ticker import MultipleLocator

fig = plt.figure()

ax = plt.axes(xlim=(0, ncol), ylim=(0, mrow))

spacing = 1 
minorLocator = MultipleLocator(spacing)
ax.yaxis.set_minor_locator(minorLocator)
ax.xaxis.set_minor_locator(minorLocator)                                                    
  
ax.grid(which="minor",color='black', linestyle='-', linewidth=2)

gap=1
for v in V:
    if value[v]==0:
        x,y=coord[v]
        ax.axvspan(xmin=x-gap/2,xmax=x+gap/2,ymin=(y-gap/2)/mrow, ymax=(y+gap/2)/mrow, facecolor='black', alpha=1)

    if value[v]==2:
        x,y=coord[v]
        ax.scatter(x, y, color="green",marker='|', alpha=1, linewidths=30)


    if value[v]==3:
        x,y=coord[v]
        ax.scatter(x, y, color="green",marker='1', alpha=1, linewidths=30)


x_start,y_start=coord[start]
ax.scatter(x_start, y_start, color="black",marker='x', alpha=1, linewidths=30)

x_goal,y_goal=coord[goal]
ax.scatter(x_goal, y_goal, color="black",marker='o', alpha=1, linewidths=30)


for item in Dij_search:
    x,y=coord[item]
    ax.axvspan(xmin=x-gap/2,xmax=x+gap/2,ymin=(y-gap/2)/mrow, ymax=(y+gap/2)/mrow, facecolor='grey', alpha=0.3)
    

for item in Dij_path:
    x_path,y_path=coord[item]
    ax.scatter(x_path, y_path, color="red",marker='o', alpha=1, linewidths=10)


for item in A_search:
    x,y=coord[item]
    ax.axvspan(xmin=x-gap/2,xmax=x+gap/2,ymin=(y-gap/2)/mrow, ymax=(y+gap/2)/mrow, facecolor='yellow', alpha=0.3)
    

for item in A_path:
    x_path,y_path=coord[item]
    ax.scatter(x_path, y_path, color="blue",marker='o', alpha=1, linewidths=10)    
plt.show()









   

        




        



    











    
    
    
