from collections import deque

import numpy as np


def find_factors_with_minimal_sum(number):
    if number == 1:
        return (1, 1)

    # Initialize variables to keep track of the factors with the minimal sum
    min_sum = float("inf")
    min_factors = None

    # Iterate through potential factors from 1 to half of the number
    for factor1 in range(1, number // 2 + 1):
        factor2 = number // factor1

        # Check if factor1 * factor2 is equal to the original number
        if factor1 * factor2 == number:
            current_sum = factor1 + factor2

            # Update the minimum sum and factors if the current sum is smaller
            if current_sum < min_sum:
                min_sum = current_sum
                min_factors = (factor1, factor2)

    return min_factors


class FindCycles:
    """
    Example:
        adjacency_matrix = np.array([
            [0, 1, 0, 0, 1, 1, 0],
            [1, 0, 1, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0],
            [0, 1, 1, 0, 1, 0, 0],
            [1, 0, 0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 1, 0],
        ])

        finder = FindCycles(adjacency_matrix)
        cycles = finder.find_cycles()
    """

    def __init__(self, adj_matrix):
        self.graph = adj_matrix
        self.num_vertices = len(adj_matrix)
        self.visited = [False] * self.num_vertices
        self.cycles = set()  # Set to store unique cycles

    def find_cycles(self) -> list:
        # Remove nodes with degree 1
        self.remove_degree_one_nodes()

        # Perform DFS for remaining nodes
        for vertex in range(self.num_vertices):
            if not self.visited[vertex]:
                self.dfs(vertex, vertex, [])

        # Deduplicate
        self.deduplicate_cycles()

        return list(self.cycles)

    def remove_degree_one_nodes(self):
        # Use a deque to efficiently process nodes with degree 1
        queue = deque([i for i in range(self.num_vertices) if np.sum(self.graph[i]) == 1])

        while queue:
            node = queue.popleft()
            self.graph[node, node] = 0  # Mark the node as removed

            # Decrease the degree of its neighbor
            neighbor = np.argmax(self.graph[node])
            self.graph[node, neighbor] = 0
            self.graph[neighbor, node] = 0

            # If the neighbor now has degree 1, add it to the queue
            if np.sum(self.graph[neighbor]) == 1:
                queue.append(neighbor)

    def dfs(self, start, current, path):
        self.visited[current] = True
        path.append(current)

        for neighbor in range(self.num_vertices):
            if self.graph[current, neighbor] == 1:
                if neighbor == start and len(path) > 2:
                    # Found a cycle with at least 3 vertices
                    self.cycles.add(tuple(path))
                elif not self.visited[neighbor] and neighbor > start:
                    # Continue DFS only if the neighbor has not been removed
                    self.dfs(start, neighbor, path)

        path.pop()
        self.visited[current] = False

    def deduplicate_cycles(self):
        all_cycles = self.cycles
        record_set = set()

        self.cycles = []
        for cycle in all_cycles:
            sorted_cycle = tuple(sorted(cycle))
            if sorted_cycle not in record_set:
                record_set.add(sorted_cycle)
                self.cycles.append(cycle)
