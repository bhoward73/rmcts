import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from time import perf_counter
import numpy.ctypeslib as npct
import ctypes
from pathlib import Path

from . import game, metaparm

c_int = ctypes.c_int
c_float = ctypes.c_float
array_int32 = npct.ndpointer(dtype=np.int32,ndim=1,flags='CONTIGUOUS')
array_float = npct.ndpointer(dtype=np.float32,ndim=1,flags='CONTIGUOUS')
array_double = npct.ndpointer(dtype=np.float64,ndim=1,flags='CONTIGUOUS')
ptr_int = ctypes.POINTER(c_int)
ptr_float = ctypes.POINTER(c_float)
ptr_void = ctypes.c_void_p # same as ctypes.c_voidp

libRMCTS_path = Path(__file__).parent / 'libRMCTS.so'
libRMCTS = npct.load_library(libRMCTS_path,".")

libRMCTS.MCTS_init.restype = ptr_void
libRMCTS.MCTS_init.argtypes = [c_int]*2 + [c_float] + [array_float]*7 + [array_int32]*10

libRMCTS.MCTS_free.restype = None
libRMCTS.MCTS_free.argtypes = [ptr_void]

libRMCTS.MCTS_flush_new_stack.restype = None
libRMCTS.MCTS_flush_new_stack.argtypes = [ptr_void]

def mcts_data(G, numSims, nnet, c_puct=None):
    if c_puct is None:
        c_puct = metaparm.c_puct
    mcts = MCTS(G,numSims,c_puct,nnet)
    pi1, v1 = mcts()
    m = G.shape[0]
    pi0 = mcts.policy[:m*mcts.n].reshape((m,mcts.n))
    v0 = mcts.value[:m]
    N = mcts.N[:m*mcts.n].reshape((m,mcts.n))
    Q = mcts.Q[:m*mcts.n].reshape((m,mcts.n))
    pi1 = pi1.reshape((m, mcts.n))
    return pi1, v1, pi0, v0, N, Q

def learn_pi_and_v(G, numSims, nnet, c_puct=None, return_mcts=False):
    if c_puct is None:
        c_puct = metaparm.c_puct
    assert numSims >= 1
    if numSims == 1:
        pi, v = nnet(G)
        v = v.flatten()
        # legalize policy
        for i in range(len(pi)):
            actions = game.getValidActions(G[i])
            pi_legal = np.zeros(pi.shape[1],dtype=np.float32)
            pi_legal[actions] = pi[i][actions]
            pi_legal /= np.sum(pi_legal)
            pi[i] = pi_legal
        if return_mcts:
            return pi, v, None
        else:
            return pi, v
    mcts = MCTS(G,numSims,c_puct,nnet)
    pi, v = mcts()
    pi = pi.reshape((mcts.num_lanes, mcts.n))
    if return_mcts:
        return pi, v, mcts
    else:
        return pi, v


def assign_simulations(S,pi):
    '''
    S is a positive integer
    pi is a probability distribution
    let n = len(pi)
    this code returns a random integer 
    array s, where the expectation 
    of s[i] = pi[i]*S, and where 
    the variance is minimal (I think)
    It uses Emma's x, x+1, x+2 method
    '''
    pi = np.array(pi,dtype=np.float64)
    pi /= np.sum(pi)
    n = len(pi)
    x = np.random.random()
    X = np.arange(S,dtype=np.float64) + x
    chunks = S*pi
    splits = np.zeros(n+1)
    splits[1:] = np.cumsum(chunks)
    s = np.histogram(X,bins=splits)[0]
    return s


def new_policy_common_ucb_Newton(Q,c,pi0,T,epsilon=1.0e-12):
    pi0 = pi0.astype(np.float64)
    Q = Q.astype(np.float64)
    assert np.min(pi0) > 0.0
    pi0 /= np.sum(pi0)
    c0 = np.float64(c) / np.sqrt(np.float64(T))
    epsilon = np.float64(epsilon)
    a_max = np.argmax(Q)
    Q_max = Q[a_max]
    delta = c0 * pi0[a_max]
    if delta < epsilon:
        delta = epsilon
    def f(delta):
        return c0*np.sum(pi0 / ((Q_max - Q) + delta)) - 1.0
    def fprime(delta):
        return -c0*np.sum(pi0 / ((Q_max - Q) + delta)**2)
    while f(delta) > epsilon:
        new_delta = delta - f(delta)/fprime(delta)
        if new_delta <= delta:
            break
        delta = new_delta
    pi1 = c0 *  pi0 / ((Q_max - Q) + delta)
    pi1 = np.maximum(pi1, 0.0) # just in case
    pi1 /= np.sum(pi1) # should already be close to 1
    pi1 = np.float32(pi1) # cast back to float32 again
    assert pi1.flags['C_CONTIGUOUS']
    return pi1

def new_policy_common_ucb_Simpleton(Q,c,pi0,T):
    c0 = c / np.sqrt(T)
    maxQ = np.max(Q)
    u = maxQ + c0
    pi1 = pi0 / (u - Q)
    pi1 /= np.sum(pi1)
    return pi1

def initial_Q(g, v):
    n = game.numActions()
    Q = np.zeros(n,dtype=np.float32)
    Q[:] = -np.inf
    actions = game.getValidActions(g)
    Q[actions] = v
    return Q

def legalize_policy(pi, g):
    '''
    pi is a policy
    g is a gamestate
    returns a policy that is supported on legal actions
    '''
    actions = game.getValidActions(g)
    pi_legal = np.zeros(len(pi),dtype=np.float32)
    pi_legal[actions] = pi[actions]
    pi_legal /= np.sum(pi_legal)
    return pi_legal, actions

class MCTS:
    def __init__(self, root_states, numSims, c_puct, nnet):
        assert numSims >= 2 #  TODO
        self.root_states = np.array(root_states, dtype=np.float32)

        # assert that root states are not terminal
        for i in range(self.root_states.shape[0]):
            ended, score = game.gameEnded(self.root_states[i])
            assert not ended

        self.numSims = numSims
        self.c_puct = c_puct
        self.nnet = nnet
        self.gamesize = game.gameLength()
        self.n = game.numActions()
        self.num_lanes = self.root_states.shape[0]

        # main part of the data
        # maximum possible number of rows to be used is 
        # the product of the number of root states with 
        # the number of simulations
        self.NUM_ALL = self.num_lanes * self.numSims
        
        # float32 parts
        self.G = np.zeros(self.NUM_ALL * self.gamesize, dtype=np.float32)
        self.G[:self.num_lanes * self.gamesize] = self.root_states.flatten()
        self.policy = np.zeros(self.NUM_ALL * self.n, dtype=np.float32)
        self.value = np.zeros(self.NUM_ALL, dtype=np.float32)
        self.Q = np.zeros(self.NUM_ALL * self.n, dtype=np.float32)
        self.N = np.zeros(self.NUM_ALL * self.n, dtype=np.float32)

        # int32 parts
        self.parent = np.zeros(self.NUM_ALL, dtype=np.int32)
        self.parent[:self.num_lanes] = -1
        self.a0 = np.zeros(self.NUM_ALL, dtype=np.int32)
        self.a0[:self.num_lanes] = -1
        self.sims = np.zeros(self.NUM_ALL, dtype=np.int32)
        self.sims[:self.num_lanes] = numSims
        self.sims_remaining = np.copy(self.sims)

        # number of occupied rows
        self.row_count = np.zeros(1, dtype=np.int32)
        self.row_count[0] = self.num_lanes

        self.inference_stack = np.zeros(self.NUM_ALL, dtype=np.int32)
        self.inference_stack[:self.num_lanes] = np.arange(self.num_lanes, dtype=np.int32)
        self.inference_stack_size = np.zeros(1, dtype=np.int32)
        self.inference_stack_size[0] = self.num_lanes
        
        self.new_stack = np.zeros(self.NUM_ALL, dtype=np.int32)
        self.new_stack_size = np.zeros(1, dtype=np.int32)

        self.new_policy = np.zeros(self.num_lanes * self.n, dtype=np.float32)
        self.new_value = np.zeros(self.num_lanes, dtype=np.float32)
        self.num_completed = np.zeros(1, dtype=np.int32)

        self._t = libRMCTS.MCTS_init(self.num_lanes, 
                                     self.numSims, 
                                     self.c_puct,
                                     self.new_policy,
                                     self.new_value, 
                                     self.G, 
                                     self.policy, 
                                     self.value, 
                                     self.Q, 
                                     self.N, 
                                     self.parent, 
                                     self.a0, 
                                     self.sims, 
                                     self.sims_remaining,
                                     self.inference_stack,
                                     self.inference_stack_size,
                                     self.new_stack,
                                     self.new_stack_size,
                                     self.num_completed,
                                     self.row_count)
        
        self.initialize()
        
        self.cpu_times = []
        self.gpu_times = []

    def __dealloc__(self):
        libRMCTS.MCTS_free(self._t)

    def local_engine(self, G):
        G = G.reshape(-1, self.gamesize)
        if self.nnet is None:
            m = G.shape[0]
            pi = np.zeros((m,self.n),dtype=np.float32)
            pi[:,:] = 1.0/self.n
            v = np.zeros(m,dtype=np.float32)
            return pi,v
        else:
            pi,v = self.nnet(G)
            v = v.flatten()
            return pi,v

    def check_data(self):
        attrs = ['G','policy','value','Q','N','parent','a0','sims','sims_remaining','inference_stack','new_stack']
        for attr in attrs:
            assert not np.any(np.isnan(getattr(self,attr)))

    def initialize(self):
        root_policy, root_values = self.local_engine(self.root_states)
        self.policy[:self.num_lanes*self.n] = root_policy.flatten()
        self.value[:self.num_lanes] = root_values.flatten()
        self.inference_stack_size[0] = 0
        self.new_stack[:self.num_lanes] = np.arange(self.num_lanes, dtype=np.int32)
        self.new_stack_size[0] = self.num_lanes

    def flush_new_stack(self):
        libRMCTS.MCTS_flush_new_stack(self._t)

    def flush_inference_stack(self,batchsize_only=False):
        # flushes the inference stack
        if batchsize_only:
            if self.nnet is None:
                batchsize = self.inference_stack_size[0]
            else:
                batchsize = self.nnet.batchsize
        else:
            batchsize = self.inference_stack_size[0]
        G = self.G.reshape(self.NUM_ALL, self.gamesize)
        policy = self.policy.reshape(self.NUM_ALL, self.n) # should be in place
        value = self.value
        start = max(0, self.inference_stack_size[0] - batchsize)
        inference_indices = self.inference_stack[start : self.inference_stack_size[0]]
        G = G[inference_indices]
        pi, v = self.local_engine(G)
        policy[inference_indices] = pi
        value[inference_indices] = v.flatten()
        self.new_stack[self.new_stack_size[0] : self.new_stack_size[0] + len(inference_indices)] = inference_indices
        self.new_stack_size[0] += len(inference_indices)
        self.inference_stack_size[0] -= len(inference_indices)

    def main_mcts(self, batchsize_only=False):
        t0 = perf_counter()
        self.initialize()
        t1 = perf_counter()
        self.gpu_times.append(t1 - t0)
        t0 = t1

        while self.num_completed[0] < self.num_lanes:
            self.flush_new_stack()
            t1 = perf_counter()
            self.cpu_times.append(t1 - t0)
            if self.inference_stack_size[0] == 0:
                break
            self.flush_inference_stack(batchsize_only=batchsize_only)
            t2 = perf_counter()
            self.gpu_times.append(t2 - t1)
            t0 = t2
        return self.new_policy, self.new_value

    def __call__(self, batchsize_only=False):
        return self.main_mcts(batchsize_only=batchsize_only)

    def plot_comparison(self, i):
        '''
        plot the comparison of the new policy and the old policy
        for the i-th lane
        '''
        n = self.n
        assert self.num_completed == self.num_lanes
        pi0 = self.policy[i*n:(i+1)*n]
        pi1 = self.new_policy[i*n:(i+1)*n]
        fig, ax = plt.subplots(2)
        ax[0].plot(pi0, label='old')
        ax[0].plot(pi1, label='new')
        ax[1].plot(self.Q[i*n:(i+1)*n], label='Q')
        ax[0].legend()
        ax[1].legend()
        plt.show()

# The function recursive_mcts below is not meant to be used,
# but serves to illustrate a simpler, purely pythonic,
# recursive implementation of MCTS using the eucb optimized-policy method.
# learn_pi_and_v is functionally equivalent to recursive_mcts, but much faster.
# The big speedup is not only because of C code,
# but also because it avoids recursion and dictionaries,
# and reduces latency cost for GPU inferences with larger size batches of inference requests,
# because it explores the tree in a breadth-first manner.
# In contrast, the code below is depth-first and recursive,
# it gets fewer inferences at a time, and is written altogether in python.
# Example timing for Othello resnet (depth 8, 48 channels):
# numSims = 1024, c_puct = 6:
# learn_pi_and_v: 40 milliseconds
# recursive_mcts: 2.4 seconds (this is 60X slower!)
def recursive_mcts(game_state, numSims, engine, c_puct, root=True):
    ended, score = game.gameEnded(game_state)
    player_id = game.playerId(game_state)
    if ended:
        assert not root
        return None, score*player_id
    assert numSims > 0
    n = game.numActions()
    pi0, v0 = engine(game_state.reshape(1,-1)) 
    pi0 = pi0.flatten()
    pi0, actions = legalize_policy(pi0, game_state)
    v0 = v0.flatten()[0]
    if numSims == 1:
        if root:
            return pi0, v0
        else:
            return None, v0
    Q = np.zeros(n,dtype=np.float32)
    Q[:] = v0
    N = np.ones(n,dtype=np.float32)
    action_count = assign_simulations(numSims-1, pi0)
    action_state_count = defaultdict(int)
    for a in np.argwhere(action_count > 0).ravel():
        for _ in range(action_count[a]):
            child_state = game.nextState(game_state,a)
            action_state_count[(a, child_state.tobytes())] += 1
    for (a,child_bytes), numSims_child in action_state_count.items():
        child_state = np.frombuffer(child_bytes, dtype=np.float32)
        _, v_child = recursive_mcts(child_state, numSims_child, engine, c_puct, root=False)
        if game.playerId(child_state) == player_id:
            v_a_child = v_child
        else:
            v_a_child = -v_child
        Q[a] = (Q[a]*N[a] + v_a_child*numSims_child) / (N[a] + numSims_child)
        N[a] += numSims_child
    # learn the new policy
    if not root:
        supp = np.argwhere(action_count > 0).flatten()
    else:
        supp = actions
    #pi1_supp = new_policy_common_ucb_Simpleton(Q[supp], c_puct, pi0[supp], numSims-1)
    pi1_supp = new_policy_common_ucb_Newton(Q[supp], c_puct, pi0[supp], numSims-1)
    pi1 = np.zeros(n, dtype=np.float32)
    pi1[supp] = pi1_supp
    v1 = (v0 + (numSims - 1) * np.dot(Q[supp], pi1_supp)) / numSims
    if root:
        return pi1, v1
    else:
        return None, v1


def learn_pi_Q_from_fixed_policy(g, numSims, engine):
    '''
    g is a game state
    numSims is a positive integer
    engine is a taz.mcts.inference object
    returns N, Q
    following the engine policy
    '''
    root_state = np.copy(g)
    root_player = game.playerId(root_state)
    ended, score = game.gameEnded(root_state)
    if ended:
        return score*root_player, 0.0
    
    Q = np.zeros(game.numActions(),dtype=np.float32)
    N = np.zeros(game.numActions(),dtype=np.float32)

    policies = {}
    def get_policy(g):
        nonlocal policies
        g_bytes = g.tobytes()
        if g_bytes in policies:
            return policies[g_bytes]
        pi,v = engine(g.reshape(1,-1))
        pi = pi.flatten()
        pi,actions = legalize_policy(pi, g)
        policies[g.tobytes()] = pi
        return pi

    # write a minimal variance estimator from here,
    # using assign_simulations...
    game_freq = defaultdict(int)
    af = assign_simulations(numSims, get_policy(root_state))
    for a in np.argwhere(af > 0).ravel():
        for i in range(af[a]):
            h = game.nextState(root_state,a)
            game_freq[(a,h.tobytes())] += 1

    while game_freq:
        new_game_freq = defaultdict(int)
        for (a0,gbytes), T in game_freq.items():
            g = np.frombuffer(gbytes, dtype=np.float32)
            ended, score = game.gameEnded(g)
            if ended:
                root_score = score*root_player
                Q[a0] = (Q[a0]*N[a0] + root_score*T)/(N[a0] + T)
                N[a0] += T
                continue
            pi = get_policy(g)
            af = assign_simulations(T, pi)
            for a in np.argwhere(af > 0).ravel():
                for i in range(af[a]):
                    h = game.nextState(g,a)
                    new_game_freq[(a0,h.tobytes())] += 1
        game_freq = new_game_freq

    pi = N / np.sum(N)

    u, pi1 = new_policy_common_ucb_Newton(Q, metaparm.c_puct, pi, numSims)

    return pi1, Q
