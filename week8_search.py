from collections import deque

graph = {
    "A": ["B", "C"],
    "B": ["D", "E"],
    "C": ["F"],
    "D": ["G"],
    "E": ["G"],
    "F": ["G"],
    "G": []
}

def bfs(start, goal):
    queue = deque([(start, [start])])
    visited = set()
    order = []

    while queue:
        node, path = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        order.append(node)

        if node == goal:
            return path, order

        for n in graph[node]:
            queue.append((n, path + [n]))

def dfs(start, goal):
    stack = [(start, [start])]
    visited = set()

    while stack:
        node, path = stack.pop()
        if node in visited:
            continue
        visited.add(node)

        if node == goal:
            return path

        for n in reversed(graph[node]):
            stack.append((n, path + [n]))

def iddfs(start, goal):
    def dls(node, goal, depth, path):
        if node == goal:
            return path
        if depth == 0:
            return None
        for n in graph[node]:
            if n not in path:
                result = dls(n, goal, depth - 1, path + [n])
                if result:
                    return result
        return None

    for depth in range(20):
        result = dls(start, goal, depth, [start])
        if result:
            return result

print("BFS path:", bfs("A", "G")[0])
print("BFS expansion order:", bfs("A", "G")[1])
print("DFS path:", dfs("A", "G"))
print("IDDFS path:", iddfs("A", "G"))

start = (1,2,3,4,0,6,7,5,8)
goal = (1,2,3,4,5,6,7,8,0)

def moves(state):
    result = []
    zero = state.index(0)
    row, col = divmod(zero, 3)

    for dr, dc, name in [(-1,0,"Up"),(1,0,"Down"),(0,-1,"Left"),(0,1,"Right")]:
        nr, nc = row + dr, col + dc
        if 0 <= nr < 3 and 0 <= nc < 3:
            p = nr * 3 + nc
            s = list(state)
            s[zero], s[p] = s[p], s[zero]
            result.append((tuple(s), name))
    return result

def puzzle_bfs(start, goal):
    queue = deque([(start, [])])
    visited = {start}

    while queue:
        state, path = queue.popleft()
        if state == goal:
            return path

        for new_state, move in moves(state):
            if new_state not in visited:
                visited.add(new_state)
                queue.append((new_state, path + [move]))

print("8-Puzzle BFS moves:", puzzle_bfs(start, goal))
