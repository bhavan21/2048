import sys
import numpy
import random
import copy
import nn
import nn2
import time

# (0,0,0,0,...)

terminalState = [0]*16
indicesList = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
actionList = [0,1,2,3]

epsilon = 0.1
gamma = 1

replaymemory = []
memSize = 50000
batchSize = 1000

trainingStarted = False


def getNextPiece(A):
	output = [0]*4
	arr = []
	for i in range(0,4):
		if A[i]!=0: arr.append(A[i])
	out_i = 0
	i = 0
	while(i<len(arr)-1):
		if(arr[i]==arr[i+1]):
			output[out_i] = arr[i]+1
			i = i+2
		else:
			output[out_i] = arr[i]
			i = i+1
		out_i = out_i + 1
	if(i == len(arr)-1):
		output[out_i] = arr[-1]

	return output

def getPieceReward(A):
	reward = 0
	arr = []
	for i in range(0,4):
		if A[i]!=0: arr.append(A[i])
	i = 0
	while(i<len(arr)-1):
		if(arr[i]==arr[i+1]):
			reward += pow(2,arr[i]+1)
			i = i+2
		else:
			i = i+1
	return reward



def getNextState(s,a):
	if a==-1: return terminalState
	nextState = [0]*16

	if a==0:
		for i in range(0,4):
			nextState[i],nextState[i+4],nextState[i+8],nextState[i+12] = getNextPiece([s[i],s[i+4],s[i+8],s[i+12]])
	elif a==1:
		for i in range(0,4):
			nextState[4*i+3],nextState[4*i+2],nextState[4*i+1],nextState[4*i] = getNextPiece([s[4*i+3],s[4*i+2],s[4*i+1],s[4*i]])
	elif a==2:
		for i in range(0,4):
			nextState[i+12],nextState[i+8],nextState[i+4],nextState[i] = getNextPiece([s[i+12],s[i+8],s[i+4],s[i]])
	elif a==3:
		for i in range(0,4):
			nextState[4*i],nextState[4*i+1],nextState[4*i+2],nextState[4*i+3] = getNextPiece([s[4*i],s[4*i+1],s[4*i+2],s[4*i+3]])

	empty_cell_list = []
	for i in range(0,16):
		if(nextState[i]==0):
			empty_cell_list.append(i)

	if(len(empty_cell_list)==0): return nextState

	p = random.uniform(0,1)
	if p<0.1: nextState[random.choice(empty_cell_list)] = 2
	else: nextState[random.choice(empty_cell_list)] = 1


	return nextState


def initializeBoard():
	s = [0]*16
	TempindicesList = copy.copy(indicesList)

	p = random.uniform(0,1)
	firstindex = random.choice(TempindicesList)
	if p<0.1: s[firstindex] = 2
	else: s[firstindex] = 1

	del TempindicesList[firstindex]

	p = random.uniform(0,1)
	secondindex = random.choice(TempindicesList)
	if p<0.1: s[secondindex] = 2
	else: s[secondindex] = 1

	return s

def printBoard(s):
	for i in range(0,4):
		j = 4*i
		print(s[j],s[j+1],s[j+2],s[j+3])
	print("")


def isValidMove(s,a):
	if(getNextState(s,a)!=s): return True
	else: return False

def getRandomAction(s):
	valid_actions = []
	for a in actionList:
		if(isValidMove(s,a)):
			valid_actions.append(a)
	if len(valid_actions)==0: return -1
	return random.choice(valid_actions)


def encodeInput(s,a):
	result = [0]*768
	j = a*192
	# 12 actions
	for x in range(0,16):
		result[j+12*x+s[x]] = 1 
	return result

def getQ(s,a):
	x = encodeInput(s,a)
	y = nn.getQ(model,x)
	# print(y)
	return y

def bestQ(s):
	bestQ = float("-INF")
	for a in range(0,4):
		currentQ = getQ(s,a)
		if isValidMove(s,a) and currentQ>bestQ:
			bestQ = currentQ
	if bestQ == float("-INF"):
		bestQ = 0
	return bestQ


def addToReplayMemory(state,action,nextState,reward):
	global replaymemory
	global trainingStarted

	replaymemory.append([state,action,nextState,reward])
	if len(replaymemory) == memSize+1:
		if not trainingStarted:
			print("Training started")
			trainingStarted = True
		del replaymemory[0] 
	
def updateQ():
	global trainingStarted
	if (trainingStarted):
		X = []
		Y = []
		newlist = random.sample(replaymemory,batchSize)

		for i in range(0,len(newlist)):
			state,action,nextState,reward = newlist[i]
			y = reward + gamma * bestQ(nextState)
			X.append(encodeInput(state,action))
			Y.append(y)
		nn.train(model,X,Y)
		


	# nn.train(model,x,y)

def getAction(s):
	bestAction = -1
	bestQ = float("-INF")
	for a in range(0,4):
		currentQ = getQ(s,a)
		# print(currentQ)
		if isValidMove(s,a) and currentQ>bestQ:
			bestQ = currentQ
			bestAction = a
	action = bestAction
	e = random.uniform(0,1)
	if e<epsilon:
		return getRandomAction(s)
	else:
		return bestAction


def printAction(a):
	A = ['U','R','D','L']
	if a==-1: print('T')
	else: print(A[a])
	print("")

def getReward(s,a):
	totalReward = 0
	temp= []
	for i in range(0,4):
		temp1 = getPieceReward([s[i],s[i+4],s[i+8],s[i+12]])
		totalReward += temp1
		temp.append(temp1)
	for i in range(0,4):
		temp1 = getPieceReward([s[4*i+3],s[4*i+2],s[4*i+1],s[4*i]])
		totalReward += temp1
		temp.append(temp1)
	for i in range(0,4):
		temp1 = getPieceReward([s[i+12],s[i+8],s[i+4],s[i]])
		totalReward += temp1
		temp.append(temp1)
	for i in range(0,4):
		temp1 = getPieceReward([s[4*i],s[4*i+1],s[4*i+2],s[4*i+3]])
		totalReward += temp1
		temp.append(temp1)
	totalReward  = (4*temp[a]-totalReward)/5000
	return totalReward



def playGame():
	currentstate = initializeBoard()
	previousState = -2
	previousAction = -2
	iters = 1
	while(currentstate!=terminalState):
		# if(iters==40):break
		iters+=1
		printBoard(currentstate)
		action = getAction(currentstate)
		if previousAction != -2:
			reward = getReward(previousState,previousAction)
			# print(bestQ)
			addToReplayMemory(previousState,previousAction,currentstate,reward)
		printAction(action)
		nextState = getNextState(currentstate,action)
		previousState = currentstate
		previousAction = action
		currentstate = nextState
	# print("iters: "iters)


if __name__ == "__main__":
	global model
	model = nn.loadModel()
	for i in range(0,1):
		start = time.time()
		print(i+1)
		sys.stdout.flush()
		playGame()
		updateQ()
		nn.saveModel(model)
		end = time.time()
		hours, rem = divmod(end-start, 3600)
		minutes, seconds = divmod(rem, 60)
		print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
# print(getNextPiece([1,0,0,1]))






	





