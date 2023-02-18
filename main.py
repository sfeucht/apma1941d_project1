import sys
import time
import random
import numpy as np
import regex as re
from string import ascii_lowercase

RAW = "joyce_raw.txt"
CHARS = list(ascii_lowercase) + [' ']
IDX_TO_CHAR = dict(enumerate(CHARS))
CHAR_TO_IDX = {v:k for k,v in IDX_TO_CHAR.items()}
STOP = 10000
if __name__ == "__main__":
    SEED = int(sys.argv[2])
    BETA = float(sys.argv[3])
    np.random.seed(SEED)
    random.seed(SEED)

"""Helper function to load in and preprocess the raw War and Peace data. 
lowercase and remove special characters.
"""
def preprocess(path):
    with open(path, 'r') as f:
        text = f.read().lower()
    text = re.sub(r"[^a-z ]", "", text)
    text = re.sub(r"  ", " ", text)
    return text

"""Helper function to obtain the transition matrix M, where each row has probabilities 
of what characters should come next. e.g. M[0][1] will have probability of 'b' coming after 'a'. 
"""
def get_M(text):
    matrix = []
    for a in CHARS:
        row = []
        for b in CHARS:
            row.append(text.count(a+b))
        matrix.append(row)
    matrix = np.array(matrix) + 1.0 # janky laplace smoothing
    matrix /= matrix.sum(axis=1, keepdims=True)
    return matrix


"""Helper function for below mcmc to calculate -log(Plausibility), or negative
log likelihood, of a function f given some scrambled data and prior probabilities M. 
f is a "function" that is really just a 27-dim array mapping scrambled to unscrambled letters.
"""
def neglog_plausibility(M, scrambled, f):
    # for the first word, take that as _s1 where we assume space is at the end.
    pl = -np.log(M[len(CHARS)-1][f[CHAR_TO_IDX[scrambled[0]]]])

    # now calculate s1s2, s2s3, etc.
    for i in range(1, len(scrambled)-1):
        pl += -np.log(M[f[CHAR_TO_IDX[scrambled[i]]]][f[CHAR_TO_IDX[scrambled[i+1]]]])

    return pl


"""Helper function that takes in an unscrambler function f and unscrambles a given text
"""
def unscramble(scrambled, f):
    # is there no better way than a for loop buh...
    unscrambled = ""
    for char in scrambled:
        unscrambled += IDX_TO_CHAR[f[CHAR_TO_IDX[char]]]
    return unscrambled


"""function that takes in a transition matrix M and a scrambled text and uses the
Metropolis MCMC method to unscramble the text. Returns the optimal f: scrambled->unscrambled

From Diaconis (2009):
• Start with a preliminary guess, say f.
• Compute Pl(f) by multiplying M(f(si), f(si+1)) for all i <-- formula 1.5.3 in lecture notes.
• Change to f* by making a random transposition of the values f assigns to two symbols.
• Compute Pl(f*); if this is larger than Pl(f), accept f*.
• If not, flip a Pl(f*)/Pl(f) coin; if it comes up heads, accept f*
• If the coin toss comes up tails, stay at f.

We actually want to do this in log space, and minimize the negative log probabilities.
So from Section 2.2 of the lecture notes, we have: 
1. Start with a preliminary guess, say f.
2. Compute Pl(f) by multiplying M(f(si), f(si+1)) for all i <-- formula 1.5.3 in lecture notes.
   In reality, we are actually doing -sum(logM(f(si), f(si+1))) for all i to avoid underflow. 
3. Change to f* by making a random transposition of the values f assigns to two symbols.
4. Compute -logPl(f*); if this is SMALLER than -logPl(f), accept f*.
5. If -logPl(f*) is BIGGER than -logPl(f), then accept f* with probability exp(-beta * (new-old))
"""
def mcmc(M, scrambled, beta=BETA):
    f = np.random.permutation(len(CHARS)) # start with preliminary guess for f (randomized array mapping)
    plausibilities = []

    i = 0
    while True:
        # compute Pl(f) using formula above
        pl_f = neglog_plausibility(M, scrambled, f)

        # change to f* by doing a random change  ## f[[0, 26]] = f[[26, 0]]
        idx1, idx2 = np.random.choice(np.arange(len(CHARS)), size=(2,), replace=False)
        fstar = np.copy(f)
        fstar[[idx1, idx2]] = fstar[[idx2, idx1]]

        # compute -logPl(f*) and then compare it to -logPl(f). We want to minimize
        pl_fstar = neglog_plausibility(M, scrambled, fstar)
        if pl_fstar < pl_f:
            f = fstar
        elif pl_fstar >= pl_f:
            flip = random.uniform(0, 1)
            if flip <= np.exp(-beta * (pl_fstar - pl_f)):
                f = fstar # accept the new one with this probability
            else:
                pass
        
        # document what's happening and break if needed
        plausibilities.append(pl_f)
        if i % 100 == 0:
            print(f"Step {i}: Plausibility {pl_f}")
        if i % 2000 == 0 and i != 0:
            print(unscramble(scrambled, f))

        i += 1
        if i > STOP:
            break 
    
    return f, plausibilities


"""the entire pipeline.
"""
if __name__ == "__main__":
    # load in data
    train = preprocess(RAW)
    transitions = get_M(train)
    scrambled_path = sys.argv[1]
    with open(f"{scrambled_path}", 'r') as f:
        scrambled = f.read()

    # time how long mcmc takes to converge 
    tick = time.time()
    f, plausibilities = mcmc(transitions, scrambled)
    print(f"Unscrambled text: \n{unscramble(scrambled, f)}")
    tock = time.time()
    print(f"time taken: {tock - tick}")

    # save the plausibilities for that seed in a pickle
    # import pickle
    # plaus_path = f"{scrambled_path.split('.')[0].split('_')[-1]}_plausibilities.pkl"
    # with open(plaus_path, 'rb') as f:
    #     curr = pickle.load(f)
    #     assert(type(curr) == dict)
    #     curr[SEED] = plausibilities
    # with open(plaus_path, 'wb') as f:
    #     pickle.dump(curr, f)

    # for the beta experiments: save beta and plausibilities here per seed (switch back in plot.py) 
    import pickle
    plaus_path = f"{scrambled_path.split('.')[0].split('_')[-1]}_beta.pkl"
    with open(plaus_path, 'rb') as f:
        curr = pickle.load(f)
        assert(type(curr) == dict)
        if SEED in curr.keys():
            curr[SEED][BETA] = plausibilities
        else:
            curr[SEED] = {BETA : plausibilities}
    with open(plaus_path, 'wb') as f:
        pickle.dump(curr, f)
