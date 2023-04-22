import TicTacToeMaster as generator
import json

# desired player is either "X" or "O"
# board states are 1 for curernt player's board slot, 0 for empty board slot and -1 for ememie's board slot
# next best moves are from 0-8 - indexed board slots from the top-left corner in horizontal snake-like indexing

# RETURNS A UNIQUE IDENTIFIER FOR A BOARD STATE AND NEXT MOVE 
def getMoveIdentifier(boardState, nextMovePosition):
    identifier = str(boardState) + ";" + str(nextMovePosition)
    return [identifier, boardState, nextMovePosition]

# JUST CREATES A TUPLE OF BOARD STATE AND NEXT BEST MOVE
def getMove(boardState, nextMovePosition):
    return (boardState, nextMovePosition)

# MAPS BOARD FROM AN ARRAY OF ['X', ' ', 'O'] TO [1, 0, -1] OR [-1, 0, 1] DEPENDING ON DESIRED PLAYER
def mapBoardState(boardState, desiredPlayer):
    mappedBoardState = []
    for oldState in boardState:
        if oldState == desiredPlayer:
            mappedBoardState.append(1)
        elif oldState == ' ':
            mappedBoardState.append(0)
        else:
            mappedBoardState.append(-1)
    return mappedBoardState

# MAIN GENERATING LOOP
def generateBestMoveSet(numberOfMoves, desiredPlayer):
    generatedMoveIdentifiers = []
    moves = []
    
    board = []
    next = ""
    keepGenerating = True
    for i in range(numberOfMoves):
        while(keepGenerating):
            board, next = generator.getRandomBoard()
            aiMoveIndex = generator.getAIMove(board = board, nextMove = next, aiPlayer= next)
            
            boardState = mapBoardState(board, desiredPlayer)
            nextBestMove = aiMoveIndex[0]
            moveIdentifier = getMoveIdentifier(boardState, nextBestMove)

            if(next == desiredPlayer 
            and (not generator.checkTie(board)) 
            and (not generator.checkLose(board, desiredPlayer)) 
            and (not generator.checkWin(board, desiredPlayer))
            and (moveIdentifier not in generatedMoveIdentifiers)):
                keepGenerating = False
        
        newMove = getMove(boardState, nextBestMove)

        generatedMoveIdentifiers.append(moveIdentifier)
        moves.append(newMove)

        keepGenerating = True
        print("Moves generated:", (i+1))
    return moves


#SAVING MOVES TO A JSON
def saveMovesToJson(moves, fileName):
    with open (f'{fileName}.json', "w") as outputJson:
        movesToSave = []
        for move in moves:
            objectToSave = {
                "move": move[1],
                "board": move[0]
            }
            movesToSave.append(objectToSave)
        json.dump(movesToSave, outputJson)
