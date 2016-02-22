import numpy as np
import numpy.random as rng
import matplotlib.pyplot as plt

iterations = 100000
radix = 7

class pure:

	def rand(self):

		return rng.randint(0,radix)



class nes:

    def __init__(self):
    	self.h_size = 1
        self.history = np.zeros([self.h_size], dtype=np.int64)

    def rand(self):

		# select next piece
		piece = rng.randint(0,radix+1)
		if piece == self.history[0] or piece == radix:
			piece = rng.randint(0,radix)
		
		# update history
		self.history[0] = piece

		return piece



class gboy:

    def __init__(self):
    	self.h_size = 2
        self.history = np.zeros([self.h_size], dtype=np.int64)

    def rand(self):

		# select next piece
		for rolls in range(3):
			piece = rng.randint(0,radix)
			# real game bug -- bitwise or, used to incorrectly test "3-in-a-row"
			if piece == (piece | self.history[0] | self.history[1]):
				continue
			else:
				break
		
		# update history
		self.history[1] = self.history[0]
		self.history[0] = piece

		return piece



class tgm1:

    def __init__(self):
    	self.h_size = 4
        self.history = np.zeros([self.h_size], dtype=np.int64)

    def rand(self):

		# select next piece
		for rolls in range(4):
			piece = rng.randint(0,radix)
			for h in range(self.h_size):
				if piece == self.history[h]:
					break
			else:
				break
		
		# update history
		for h in range(self.h_size-1, 0, -1):
			self.history[h] = self.history[h-1]
		self.history[0] = piece

		return piece



class tgm2:

    def __init__(self):
    	self.h_size = 4
        self.history = np.zeros([self.h_size], dtype=np.int64)

    def rand(self):

		# select next piece
		for rolls in range(6):
			piece = rng.randint(0,radix)
			for h in range(self.h_size):
				if piece == self.history[h]:
					break
			else:
				break
		
		# update history
		for h in range(self.h_size-1, 0, -1):
			self.history[h] = self.history[h-1]
		self.history[0] = piece

		return piece



class tgm3:

    def __init__(self):
    	self.h_size = 4
        self.history = np.zeros([self.h_size], dtype=np.int64)
        self.drought = np.zeros([radix], dtype=np.int64)
        self.droughtest = 0
        self.pool = np.zeros([radix*5], dtype=np.int64)
        for i in range(radix):
        	self.pool[i+(0*radix)] = i
        	self.pool[i+(1*radix)] = i
        	self.pool[i+(2*radix)] = i
        	self.pool[i+(3*radix)] = i
        	self.pool[i+(4*radix)] = i

    def rand(self):
		# select next piece
		for rolls in range(6):
			index = rng.randint(0,35)
			piece = self.pool[index]
			self.pool[index] = self.droughtest
			for h in range(self.h_size):
				if piece == self.history[h]:
					break
			else:
				break
		
		# update history
		for h in range(self.h_size-1, 0, -1):
			self.history[h] = self.history[h-1]
		self.history[0] = piece

		# update droughts
		for p in range(radix):
			self.drought[p] += 1
		self.drought[piece] = 0
		self.droughtest = np.argmax(self.drought)

		return piece



class ccs:

    def __init__(self):
    	self.h_size = 6
        self.history = np.zeros([self.h_size], dtype=np.int64)

    def rand(self):

		# select next piece
		for rolls in range(4):
			piece = rng.randint(0,radix)
			for h in range(self.h_size):
				if piece == self.history[h]:
					break
			else:
				break
		else: # weird bonus 5th roll
			if rng.randint(0,2) == 1:
				piece = self.history[1]
			else:
				piece = self.history[5]

		# update history
		for h in range(self.h_size-1, 0, -1):
			self.history[h] = self.history[h-1]
		self.history[0] = piece

		return piece



class srs:

    def __init__(self):
    	self.pool = np.arange(0, radix, 1, dtype=np.int64)
    	rng.shuffle(self.pool)
        self.index = 0

    def rand(self):

		# select next piece
		piece = self.pool[self.index]
		self.index += 1
		if self.index == radix:
			self.__init__()
		
		return piece



class toj:

    def __init__(self):
    	self.pool = np.arange(0, radix+1, 1, dtype=np.int64)
    	self.pool[radix] = rng.randint(0,radix)
    	rng.shuffle(self.pool)
        self.index = 0

    def rand(self):

		# select next piece
		piece = self.pool[self.index]
		self.index += 1
		if self.index == (radix+1):
			self.__init__()
		
		return piece



class bag2x:

    def __init__(self):
    	self.pool = np.arange(0, radix*2, 1, dtype=np.int64)
    	for i in range(radix, radix*2):
    		self.pool[i] = self.pool[i] % radix
    	rng.shuffle(self.pool)
        self.index = 0

    def rand(self):

		# select next piece
		piece = self.pool[self.index]
		self.index += 1
		if self.index == (radix*2):
			self.__init__()
		
		return piece



class tnt64:

    def __init__(self):
    	self.pool = np.arange(0, radix*9, 1, dtype=np.int64)
    	for i in range(radix, radix*9):
    		self.pool[i] = self.pool[i] % radix
    	rng.shuffle(self.pool)
        self.index = 0

    def rand(self):

		# select next piece
		piece = self.pool[self.index]
		self.index += 1
		if self.index == (radix*9):
			self.__init__()
		
		return piece

def intervals(randomizer):
	intervals = np.zeros([1000], dtype=np.int64)
	last_seen = np.zeros([radix], dtype=np.int64)
	for i in range(iterations):
		piece = randomizer()
		if last_seen[piece]:
			intervals[i - last_seen[piece]] += 1
		last_seen[piece] = i
		
	intervals = intervals / 1.0 / (iterations-radix)
	return intervals

# calculate the intervals for the various randomizers
pure_random = intervals(pure().rand)
nes = intervals(nes().rand)
gboy = intervals(gboy().rand)
tgm1 = intervals(tgm1().rand)
tgm2 = intervals(tgm2().rand)
tgm3 = intervals(tgm3().rand)
ccs = intervals(ccs().rand)
srs = intervals(srs().rand)
toj = intervals(toj().rand)
bag2x = intervals(bag2x().rand)
tnt64 = intervals(tnt64().rand)

# create plots
plt.plot(pure_random, 'k.-', color='#000000', label='pure_random', linewidth=2)
plt.plot(nes, 'k.-', label='nes', color='#2277EE', linewidth=2)
plt.plot(gboy, 'k.-', label='gboy', color='#113311', linewidth=2)
plt.plot(tgm1, 'k.-', label='tgm1', color='#CC6666', linewidth=2)
plt.plot(tgm2, 'k.-', label='tgm2', color='#EE6666', linewidth=2)
plt.plot(tgm3, 'k.-', label='tgm3', color='#FF0000', linewidth=2)
plt.plot(ccs, 'k.-', label='ccs', color='#FF00FF', linewidth=2)
plt.plot(srs, 'k.-', label='srs', color='#00FFFF', linewidth=2)
plt.plot(toj, 'k.-', label='toj', color='#00FF88', linewidth=2)
plt.plot(bag2x, 'k.-', label='bag2x', color='#0000FF', linewidth=2)
plt.plot(tnt64, '.-', label='tnt64', color='#FFFF00', linewidth=2)
plt.title('Repeat intervals of various Tetris randomization algorithms')
plt.xlabel('interval')
plt.xlim(xmin=1, xmax=15)
plt.ylabel('probability')
plt.legend()

plt.show()