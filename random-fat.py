from __future__ import division
import itertools
import numpy as np
import numpy.random as rng
import matplotlib.pyplot as plt


# Name: Random Function Analysis Tool
# Author: Eric Taylor
#
# Generates a random sequence, calculating the probability
# mass function of the intervals and the entropy of the
# output given a history.
#
# Currently 11 algorithms used in various Tetris games are
# implemented.


iterations = 100000 # random sequence length
radix = 7 # distinct category types
depth = 8 # conditional entropy history depth


# Atari / Sega / etc Tetris
class pure:

	def rand(self):

		return rng.randint(0,radix)


# NES Tetris
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


# GameBoy Tetris
class gboy:

    def __init__(self): 
            self.h_size = 2 
            self.history = np.zeros([self.h_size], dtype=np.int64) 

    def rand(self): 

            # select next piece 
            cycles = rng.randint(0,0x4000) # to-do: model this distribution... it's unlikely to be random
            for rolls in range(3): 
                    div = cycles // 0x40 # convert to 8 bit counter 
                    piece = div % 7 
                    # real game bug -- bitwise or, used to incorrectly test "3-in-a-row" 
                    if piece == (piece | self.history[0] | self.history[1]): 
                            # deterministic cycle advance for the "rerolls" 
                            cycles += 100 # constant 
                            cycles += (388 * (div // 7)) # full loop of 7 
                            cycles += (56 * (div % 7)) # cycles for remainder 
                            cycles &= 0x3FFF # 6 bit cycle counter (not 8 bits because every instruction is a multiple of 4 cycles) 
                            continue 
                    else: 
                            break 
            
            # update history 
            self.history[1] = self.history[0] 
            self.history[0] = piece 

            return piece 


# GameBoy Tetris (hypothetical bugfixed)
class gboy_fixed:

    def __init__(self): 
            self.h_size = 2 
            self.history = np.zeros([self.h_size], dtype=np.int64) 

    def rand(self): 

            # select next piece 
            for rolls in range(3): 
                    piece = rng.randint(0,radix) 
                    if ((piece == self.history[0]) and (self.history[0] == self.history[1])): 
                            continue 
                    else: 
                            break 
            
            # update history 
            self.history[1] = self.history[0] 
            self.history[0] = piece 

            return piece


# Tetris the Grand Master
class tgm1:

    def __init__(self):
    	self.h_size = 4
        self.history = np.zeros([self.h_size], dtype=np.int64) # initial history ZZZZ
        self.first_piece = 1

    def rand(self):

		# select next piece
		for rolls in range(4):

			# roll
			if self.first_piece == 1:
				while self.first_piece == 1:
					piece = rng.randint(0,radix)
					if piece not in (1, 2, 5): # Z, S, O forbidden as first piece
						self.first_piece = 0
			else:
				piece = rng.randint(0,radix)

			# check history
			if piece not in self.history:
				break
		
		# update history
		for h in range(self.h_size-1, 0, -1):
			self.history[h] = self.history[h-1]
		self.history[0] = piece

		return piece


# Tetris the Grand Master 2: The Absolute Plus
class tgm2:

    def __init__(self):
    	self.h_size = 4
        self.history = [1, 2, 1, 2] # initial history ZSZS
        self.first_piece = 1

    def rand(self):

		# select next piece
		for rolls in range(6):

			# roll
			if self.first_piece == 1:
				while self.first_piece == 1:
					piece = rng.randint(0,radix)
					if piece not in (1, 2, 5): # Z, S, O forbidden as first piece
						self.first_piece = 0
			else:
				piece = rng.randint(0,radix)

			# check history
			if piece not in self.history:
				break
		
		# update history
		for h in range(self.h_size-1, 0, -1):
			self.history[h] = self.history[h-1]
		self.history[0] = piece

		return piece


# Tetris the Grand Master 3: Terror Instinct
class tgm3:

    def __init__(self):
    	self.h_size = 4
        self.history = [1, 2, 1, 2] # initial history ZSZS
        self.first_piece = 1
        self.drought = np.zeros([radix], dtype=np.int64)
        self.droughtest = 0
        self.pool = np.zeros([radix*5], dtype=np.int64)
        for i in range(radix):
        	self.pool[(i*5)+0] = i
        	self.pool[(i*5)+1] = i
        	self.pool[(i*5)+2] = i
        	self.pool[(i*5)+3] = i
        	self.pool[(i*5)+4] = i
        	self.drought[i] = -999

    def rand(self):
		# select next piece
		for rolls in range(6):

			# roll first piece
			if self.first_piece == 1:
				while self.first_piece == 1:
					piece = rng.randint(0,radix)
					if piece not in (1, 2, 5): # Z, S, O forbidden as first piece
						break
			# roll general piece
			else:
				index = rng.randint(0,35)
				piece = self.pool[index]
				self.pool[index] = self.droughtest
				if piece not in self.history:
					break
		
		# update history
		for h in range(self.h_size-1, 0, -1):
			self.history[h] = self.history[h-1]
		self.history[0] = piece

		# unless it's the first piece...
		if self.first_piece == 1:
			self.first_piece = 0
		# ... update droughts
		else:
			for p in range(radix):
				self.drought[p] += 1
			self.drought[piece] = 0
			# new droughtest
			if piece == self.droughtest:
				self.droughtest = np.argmax(self.drought)
				# real game bug -- under specific conditions the piece pool is not updated with the new droughtest
				if not (piece == self.droughtest and rolls > 0 and np.argmin(self.drought) >= 0):
					self.pool[index] = self.droughtest

		return piece


# Tetris with Cardcaptor Sakura: Eternal Heart
class ccs:

    def __init__(self):
    	self.h_size = 6
        self.history = [-1, -1, -1, -1, -1, -1]

    def rand(self):

		# select next piece
		for rolls in range(4):

			#roll
			piece = rng.randint(0,radix)

			# check history
			if piece not in self.history:
				break

		# weird bonus 5th roll
		else:
			if rng.randint(0,2) == 1:
				if self.history[1] != -1:
					piece = self.history[1]
				else:
					piece = rng.randint(0,radix)
			else:
				if self.history[5] != -1:
					piece = self.history[1]
				else:
					piece = rng.randint(0,radix)

		# update history
		for h in range(self.h_size-1, 0, -1):
			self.history[h] = self.history[h-1]
		self.history[0] = piece

		return piece


# Super Rotation System / Tetris Guideline / "Random Generator" aka 7-bag
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


# Tetris Online Japan (beta)
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


# Double Bag aka 14-bag
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


# The New Tetris
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


# a recursion to sum calculate the entropy partials
def ent_calc(d, pat_cnt, history, entropy):
	for history[d] in range(radix):
		if pat_cnt[d][tuple(history[:d+1])] > 0:
			entropy[d] -= ( (pat_cnt[d][tuple(history[:d+1])] / iterations) * np.log(pat_cnt[d][tuple(history[:d+1])] / pat_cnt[d-1][tuple(history[:d])]) )
			if d < depth-1:
				ent_calc(d+1, pat_cnt, history, entropy)

def stats_calc(randomizer):
	
	# interval vars
	intervals = np.zeros([1000], dtype=np.int64)
	last_seen = np.zeros([radix], dtype=np.int64)

	# entropy vars	
	history = np.zeros((depth), dtype=np.int64)
	entropy = np.zeros((depth), dtype=np.float64)
	pat_cnt = [np.zeros((radix), dtype=np.int64)]
	for i in range(1, depth):
		pat_cnt.append(np.repeat(np.expand_dims(pat_cnt[i-1], axis = i), radix, i))

	for _ in itertools.repeat(None, iterations):

		# get next piece
		piece = randomizer()
	
		# update history
		for h in reversed(range(depth)):
			history[h] = history[h-1]
		history[0] = piece

		# update interval counters
		last_seen += 1
		intervals[last_seen[piece]] += 1
		last_seen[piece] = 0

		# update pattern counters
		for h in range(depth):
			pat_cnt[h][tuple(history[depth-1-h:])] += 1
	
	# calculate entropy
	for history[0] in range(radix):
		if pat_cnt[0][history[0]] > 0:
			entropy[0] -= ( (pat_cnt[0][history[0]] / iterations) * np.log(pat_cnt[0][history[0]] / iterations) )
			if 0 < depth-1:
				ent_calc(1, pat_cnt, history, entropy)
	
	intervals = intervals / iterations # converts to % all intervals
	entropy = entropy / np.log(radix) # converts to % of pure random

	print randomizer.im_class
	print "interval: ", intervals[:20]
	print "entropy: ", entropy
	return (intervals, entropy)

# calculate the intervals for the various randomizers
rnd_int, rnd_ent = stats_calc(pure().rand)
nes_int, nes_ent = stats_calc(nes().rand)
gby_int, gby_ent = stats_calc(gboy().rand)
gm1_int, gm1_ent = stats_calc(tgm1().rand)
gm2_int, gm2_ent = stats_calc(tgm2().rand)
gm3_int, gm3_ent = stats_calc(tgm3().rand)
ccs_int, ccs_ent = stats_calc(ccs().rand)
srs_int, srs_ent = stats_calc(srs().rand)
toj_int, toj_ent = stats_calc(toj().rand)
b14_int, b14_ent = stats_calc(bag2x().rand)
b63_int, b63_ent = stats_calc(tnt64().rand)

# create plots
plt.figure(num=1, figsize=(10, 5), dpi=160, facecolor='w', edgecolor='k')

plt.subplot(121)
plt.plot(rnd_int, '.-', label='pure_random', color='#000000')
plt.plot(nes_int, '.-', label='nes', color='#2277EE')
plt.plot(gby_int, '.-', label='gboy', color='#113311')
plt.plot(gm1_int, '.-', label='tgm1', color='#CC6666')
plt.plot(gm2_int, '.-', label='tgm2', color='#EE6666')
plt.plot(gm3_int, '.-', label='tgm3', color='#FF0000')
plt.plot(ccs_int, '.-', label='ccs', color='#FF00FF')
plt.plot(srs_int, '.-', label='srs', color='#00FFFF')
plt.plot(toj_int, '.-', label='toj', color='#00FF88')
plt.plot(b14_int, '.-', label='bag2x', color='#0000FF')
plt.plot(b63_int, '.-', label='tnt64', color='#FFFF00')
plt.title('probability of drought intervals')
plt.xlabel('interval')
plt.xlim(xmin=1, xmax=15)
plt.ylabel('probability')

plt.subplot(122)
plt.plot(rnd_ent, '.-', label='pure_random', color='#000000')
plt.plot(nes_ent, '.-', label='nes', color='#2277EE')
plt.plot(gby_ent, '.-', label='gboy', color='#113311')
plt.plot(gm1_ent, '.-', label='tgm1', color='#CC6666')
plt.plot(gm2_ent, '.-', label='tgm2', color='#EE6666')
plt.plot(gm3_ent, '.-', label='tgm3', color='#FF0000')
plt.plot(ccs_ent, '.-', label='ccs', color='#FF00FF')
plt.plot(srs_ent, '.-', label='srs', color='#00FFFF')
plt.plot(toj_ent, '.-', label='toj', color='#00FF88')
plt.plot(b14_ent, '.-', label='bag2x', color='#0000FF')
plt.plot(b63_ent, '.-', label='tnt64', color='#FFFF00')
plt.title('conditional entropy given history')
plt.xlabel('history size')
plt.xlim(xmin=0, xmax=14)
plt.ylabel('entropy')
plt.ylim(ymin=0, ymax=1)

plt.tight_layout()
plt.show()