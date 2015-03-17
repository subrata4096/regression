#!/usr/bin/python

from networkx import *
import sys

#n=10 # 10 nodes
#m=20 # 20 edges
if(len(sys.argv) < 3):
	print "./erdos_renyi.py <number of nodes> <density>"
	exit(0)
 
m = int(sys.argv[1])
density = float(sys.argv[2])
n = int(m * density)
print "m= " + str(m) + " n= " + str(n)

G=gnm_random_graph(m,n)

# some properties
#print("node degree clustering")
#for v in nodes(G):
#    print('%s %d %f' % (v,degree(G,v),clustering(G,v)))

# print the adjacency list to terminal
outfilename = "testGraph_" + str(m) + "_" + str(density) + ".txt"
try:
    #write_adjlist(G,sys.stdout)
    write_edgelist(G, outfilename, data=False)
except TypeError: # Python 3.x
    write_adjlist(G,sys.stdout.buffer)


