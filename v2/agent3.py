import sys
import numpy
import random
import copy
import nn2
import time
from collections import deque


terminalState = [0]*16
indicesList = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
actionList = [0,1,2,3]

totalEpisodes = 100

#fourprob = 0.1

epsilon = 0.9
gamma = 1

replaymemory = deque()
memSize = 5000
batchSize = 32

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

	nextState[random.choice(empty_cell_list)] = 1


	return nextState


def initializeBoard():
	s = [0]*16
	TempindicesList = copy.copy(indicesList)
	firstindex = random.choice(TempindicesList)
	s[firstindex] = 1
	del TempindicesList[firstindex]
	secondindex = random.choice(TempindicesList)
	s[secondindex] = 1

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


def encodeInput(s):
	result = [0]*192
	for x in range(0,16):
		result[12*x+s[x]] = 1 
	return result

def getQ(s):
	x = encodeInput(s)
	y = nn2.getQ(model,x)
	return y


def addToReplayMemory(state,action,nextState,reward):
	global replaymemory
	global trainingStarted

	replaymemory.append([state,action,nextState,reward])
	if len(replaymemory) > memSize:
		if not trainingStarted:
			print("Training started")
			trainingStarted = True
		replaymemory.popleft()

def Zeroes(state):
	temp = 0
	for i in range(16):
		if s[i] == 0:
			temp+=1
	return temp


def getNextAllPossibleState(s,a):
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

	temp = []

	if(len(empty_cell_list)==0): 
		return [nextState]
	else:
		for i in empty_cell_list:
			tempstate = nextState
			tempstate[i] = 2
			temp.append(tempstate[:])
		return temp
		
	
def updateQ():
	global trainingStarted
	global epsilon
	if (trainingStarted):
		X = []
		Y = []
		newlist = random.sample(replaymemory,batchSize)

		for i in range(0,len(newlist)):
			state,action,nextState,reward = newlist[i]
			y = getQ(state)
			twoStates = getNextAllPossibleState(state,action)
			s = 0.0
			for j in twoStates:
				s += max(getQ(j)) * gamma
			s = s/len(twoStates)
			s += reward
			X.append(encodeInput(state))
			y[action] = s
			if s == float("INF") or s == float("-INF"):
				print(s,state)
			Y.append(y)
		if totalEpisodes != 1:
			nn2.train(model,X,Y)
		
	# nn.train(model,x,y)

def getAction(s):
	global totalEpisodes
	bestAction = -1
	bestQ = float("-INF")
	Qlist = getQ(s)
	if totalEpisodes == 1:
		print(Qlist)
	for a in range(0,4):
		currentQ = Qlist[a]
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

def getReward(s,a,ns):
	totalReward = 0.0
	temp= []
	if a == 0:
		for i in range(0,4):
			temp1 = getPieceReward([s[i],s[i+4],s[i+8],s[i+12]])
			totalReward += temp1
		return totalReward/5000 
	elif a == 1:
		for i in range(0,4):
			temp1 = getPieceReward([s[4*i+3],s[4*i+2],s[4*i+1],s[4*i]])
			totalReward += temp1
		return totalReward/5000 
	elif a == 2:	
		for i in range(0,4):
			temp1 = getPieceReward([s[i+12],s[i+8],s[i+4],s[i]])
			totalReward += temp1
		return totalReward/5000 
	elif a == 3:
		for i in range(0,4):
			temp1 = getPieceReward([s[4*i],s[4*i+1],s[4*i+2],s[4*i+3]])
			totalReward += temp1
			temp.append(temp1)
		return totalReward/5000


def playGame():
	currentstate = initializeBoard()
	previousState = -2
	previousAction = -2
	while(currentstate!=terminalState):
		if totalEpisodes == 1:
			printBoard(currentstate)
		action = getAction(currentstate)
		if previousAction != -2:
			reward = getReward(previousState,previousAction,currentstate)

			addToReplayMemory(previousState,previousAction,currentstate,reward)
			updateQ()
		if totalEpisodes == 1:
			printAction(action)
		nextState = getNextState(currentstate,action)
		previousState = currentstate
		previousAction = action
		currentstate = nextState

if __name__ == "__main__":
	global model
	istrain = int(sys.argv[1])
	if istrain == 1:
		totalEpisodes = 1
		epsilon = 0
	model = nn2.loadModel()
	Totalsteps = 0
	for i in range(0,totalEpisodes):
		start = time.time()
		print(i+1)
		sys.stdout.flush()
		playGame()
		if trainingStarted:
			epsilon = epsilon/1.0025
		if Totalsteps == 10:
			nn2.saveModel(model)
			Totalsteps = 0
		Totalsteps+=1
		end = time.time()
		hours, rem = divmod(end-start, 3600)
		minutes, seconds = divmod(rem, 60)
		print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))






	





