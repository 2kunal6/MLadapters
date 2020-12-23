# adapted from Neelam Yadav's code in https://www.geeksforgeeks.org/breadth-first-search-or-bfs-for-a-graph/
# Python3 Program to print BFS traversal
# from a given source vertex. BFS(int s)
# traverses vertices reachable from s.
from collections import defaultdict


# This class represents a directed graph
# using adjacency list representation
class Graph:

    # Constructor
    def __init__(self):

        # default dictionary to store graph
        self.graph = defaultdict(list)

    # function to add an edge to graph
    def addEdge(self, u, v):
        self.graph[u].append(v)

    # Function to print a BFS of graph
    def BFS(self, s):

        file_structures = []
        # Mark all the vertices as not visited
        visited = {}
        for i in self.graph:
            visited[i] = False
            for j in self.graph[i]:
                visited[j] = False

        # Create a queue for BFS
        queue = []

        # Mark the source node as
        # visited and enqueue it
        queue.append([s, s])
        visited[s] = True

        while queue:

            # Dequeue a vertex from
            # queue and print it
            s = queue.pop(0)
            file_structures.append(s[1])
            #print(s, end=" ")

            # Get all adjacent vertices of the
            # dequeued vertex s. If a adjacent
            # has not been visited, then mark it
            # visited and enqueue it
            for i in self.graph[s[0]]:
                if visited[i] == False:
                    queue.append([i, s[1]+ '/' + i])
                    visited[i] = True
        return file_structures