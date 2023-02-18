import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from main import STOP

font = {'weight' : 'normal',
        'size'   : 16}

import matplotlib
matplotlib.rc('font', **font)
plt.rc('legend', fontsize=10)  

def plot_seeds_for_file(title, directory):
    textno = title.split('.')[0].split('_')[-1]
    with open(f"{directory}/{textno}_plausibilities.pkl", 'rb') as f:
        plausibilities = pickle.load(f)
        assert(type(plausibilities) == dict)

    # plot the plausibility scores over time for each step
    plt.figure(figsize=(10,6), tight_layout=True)
    _, ax = plt.subplots()
    for seed, plaus_list in plausibilities.items():
        # plaus_list = [np.exp(-x) for x in plaus_list]
        ax.plot(plaus_list, '-', linewidth=2, label=f"Seed {seed}")
    plt.xticks(range(0, STOP, 1000))
    plt.xlabel('Steps')
    plt.ylabel('Negative Log Plausibility')
    plt.title(f'Plausibility Over Time: Decoding {title}')
    plt.show()

def plot_datasets_across_files(texts, directories):
    all_data = []
    for text in texts:
        textno = text.split('.')[0].split('_')[-1]
        data = []
        for d in directories:
            with open(f"{d}/{textno}_plausibilities.pkl", 'rb') as f:
                plausibilities = pickle.load(f)
                assert(type(plausibilities) == dict)
            slce = [v[2000] for v in plausibilities.values()]
            data.append(slce)
        all_data.append(data)

    def set_box_color(bp, color):
        plt.setp(bp['boxes'], color=color)
        plt.setp(bp['whiskers'], color=color)
        plt.setp(bp['caps'], color=color)
        plt.setp(bp['medians'], color=color)

    plt.figure()
    ticks = ["War and Peace \nby Leo Tolstoy \n(3M Chars)", "Small Health Care Bill \n(0.13M Chars)", "Large Collection of \nHealth Care Bills \n(3.1M Chars)", "James Joyce Works \n(2M Chars)"]

    bpl = plt.boxplot(all_data[0], positions=np.array(range(len(all_data[0])))*3.0-0.8, sym='', widths=0.6)
    bpr = plt.boxplot(all_data[1], positions=np.array(range(len(all_data[1])))*3.0, sym='', widths=0.6)
    bpg = plt.boxplot(all_data[2], positions=np.array(range(len(all_data[2])))*3.0+0.8, sym='', widths=0.6)
    set_box_color(bpl, '#D7191C') # colors are from http://colorbrewer2.org/
    set_box_color(bpr, '#2C7BB6')
    set_box_color(bpg, '#34eb3a')

    # draw temporary red and blue lines and use them to create a legend
    plt.plot([], c='#D7191C', label='Text 1: 19th Century Russian Lit')
    plt.plot([], c='#2C7BB6', label='Text 2: Affordable Care Act')
    plt.plot([], c='#34eb3a', label='Text 3: 20th Century Irish Lit')
    plt.legend()

    plt.xticks([0, 3, 6, 9], ticks)

    plt.xlabel("Dataset")
    plt.ylabel("Negative Log Plausibility at Step 2000 Across 10 Seeds \u2193")
    plt.title("Spread of Plausibilities for Different Training Corpuses Across Given Texts")
    plt.tight_layout()
    plt.show()

def plot_beta_choices(texts):
    per_text = []
    for text in texts:
        textno = text.split('.')[0].split('_')[-1]
        with open(f"tolstoy_data/{textno}_beta.pkl", 'rb') as f:
            by_seed = pickle.load(f)
            # by_beta = pickle.load(f)
        
        # flip the dict to be by beta, doy... 
        by_beta = defaultdict(dict)
        for key, val in by_seed.items():
            for subkey, subval in val.items():
                by_beta[subkey][key] = subval
        
        betas = by_beta.keys()
        array = np.array([list(d.values()) for d in by_beta.values()]) #(6 betas, 10 seeds, 10000 plaus)
        global_min = np.min(array) #the lowest possible plausibility for this data
        run_mins = np.min(array, axis=2) #the lowest value that each run went down to, shape (6 betas, 10 seeds)
        n_converged = np.sum(1 - (run_mins > global_min + 10), axis=1) # shape (6,1)
        
        per_text.append(n_converged)
    
    plt.figure(figsize=(10,6), tight_layout=True)
    _, ax = plt.subplots()
    for i, ns in enumerate(per_text):
        ax.plot(ns, '-', linewidth=2, label=f"Text {i+1}")
    ax.legend()
    plt.xticks(range(len(betas)), betas)
    plt.xlabel("Value of \u03B2 Tested")
    plt.ylabel("Number of Seeds That Converged")
    plt.title("")
    plt.show()


d = "joyce_data"
# plot_seeds_for_file("student_10_text1.txt", d)
# plot_seeds_for_file("student_204_text2.txt", d)
# plot_seeds_for_file("student_87_text3.txt", d)

texts = ["student_10_text1.txt", "student_204_text2.txt", "student_87_text3.txt"]
directories = ["tolstoy_data", "law_data", "largelaw_data", "joyce_data"]
# plot_datasets_across_files(texts, directories)
plot_beta_choices(texts)