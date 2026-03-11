import random

def makeEmptyBoard():
    return [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]

def addRandomTile(board):
    empty = []
    for i in range(4):
        for j in range(4):
            if not board[i][j]:
                empty.append([i, j])
    if not empty:
        return
    i, j = random.choice(empty)
    board[i][j] = 2 if random.random() < 0.9 else 4

def isGameOver(board):
    if any(0 in r for r in board):
        return False
    for i in range(4):
        for j in range(4):
            v = board[i][j]
            if j < 3 and v == board[i][j + 1]:
                return False
            if i < 3 and v == board[i + 1][j]:
                return False
    return True

def copy(b):
    return [r[:] for r in b]

def simulateMove(board, dir):
    N = 4
    old = copy(board)
    res = makeEmptyBoard()
    actions = []
    merges = []
    scoreGain = 0

    def slide(line):
        nz = []
        for k in range(N):
            if line[k]:
                nz.append({"v": line[k], "idx": k})
        
        out = [0, 0, 0, 0]
        moveMap = []
        mergeInfo = []
        w = 0
        skip = False
        gain = 0
        
        i = 0
        while i < len(nz):
            if skip:
                skip = False
                i += 1
                continue
            
            if i + 1 < len(nz) and nz[i]["v"] == nz[i + 1]["v"]:
                m = nz[i]["v"] * 2
                out[w] = m
                gain += m
                moveMap.append({"from": nz[i]["idx"], "to": w, "v": nz[i]["v"]})
                moveMap.append({"from": nz[i + 1]["idx"], "to": w, "v": nz[i + 1]["v"]})
                mergeInfo.append({"at": w, "v": m})
                w += 1
                i += 2  # 다음 요소는 합쳐졌으므로 건너뜀
            else:
                out[w] = nz[i]["v"]
                moveMap.append({"from": nz[i]["idx"], "to": w, "v": nz[i]["v"]})
                w += 1
                i += 1
                
        return {"out": out, "moveMap": moveMap, "mergeInfo": mergeInfo, "gain": gain}

    if dir == "left":
        for r in range(N):
            slide_res = slide(old[r])
            out, moveMap, mergeInfo, gain = slide_res["out"], slide_res["moveMap"], slide_res["mergeInfo"], slide_res["gain"]
            scoreGain += gain
            res[r] = out
            for m in moveMap:
                actions.append({
                    "from": {"row": r, "col": m["from"]},
                    "to": {"row": r, "col": m["to"]},
                    "value": m["v"],
                })
            for m in mergeInfo:
                merges.append({"row": r, "col": m["at"], "value": m["v"]})
                
    elif dir == "right":
        for r in range(N):
            rev = list(reversed(old[r]))
            slide_res = slide(rev)
            out, moveMap, mergeInfo, gain = slide_res["out"], slide_res["moveMap"], slide_res["mergeInfo"], slide_res["gain"]
            scoreGain += gain
            restored = list(reversed(out))
            res[r] = restored
            for m in moveMap:
                actions.append({
                    "from": {"row": r, "col": N - 1 - m["from"]},
                    "to": {"row": r, "col": N - 1 - m["to"]},
                    "value": m["v"],
                })
            for m in mergeInfo:
                merges.append({"row": r, "col": N - 1 - m["at"], "value": m["v"]})
                
    elif dir == "up":
        for c in range(N):
            col = [old[0][c], old[1][c], old[2][c], old[3][c]]
            slide_res = slide(col)
            out, moveMap, mergeInfo, gain = slide_res["out"], slide_res["moveMap"], slide_res["mergeInfo"], slide_res["gain"]
            scoreGain += gain
            for r in range(N):
                res[r][c] = out[r]
            for m in moveMap:
                actions.append({
                    "from": {"row": m["from"], "col": c},
                    "to": {"row": m["to"], "col": c},
                    "value": m["v"],
                })
            for m in mergeInfo:
                merges.append({"row": m["at"], "col": c, "value": m["v"]})
                
    elif dir == "down":
        for c in range(N):
            col = list(reversed([old[0][c], old[1][c], old[2][c], old[3][c]]))
            slide_res = slide(col)
            out, moveMap, mergeInfo, gain = slide_res["out"], slide_res["moveMap"], slide_res["mergeInfo"], slide_res["gain"]
            scoreGain += gain
            restored = list(reversed(out))
            for r in range(N):
                res[r][c] = restored[r]
            for m in moveMap:
                actions.append({
                    "from": {"row": N - 1 - m["from"], "col": c},
                    "to": {"row": N - 1 - m["to"], "col": c},
                    "value": m["v"],
                })
            for m in mergeInfo:
                merges.append({"row": N - 1 - m["at"], "col": c, "value": m["v"]})

    changed = old != res
    return {"changed": changed, "result": res, "actions": actions, "merges": merges, "scoreGain": scoreGain}