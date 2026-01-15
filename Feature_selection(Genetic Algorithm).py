import itertools
from matplotlib import rcParams
import numpy as np
import xlrd
from sklearn.feature_selection import mutual_info_classif
from sklearn.datasets import load_breast_cancer
from deap import base, creator, tools, algorithms
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

import random
random.seed(200)
np.random.seed(200)

def get_data(x_index,y_index = -1): # get data 
    x,y = [],[]
    sheet = file.sheet_by_index(0)
    for i in range(1,sheet.nrows):  
        info = []
        info.append(sheet.row_values(i))
        user = []
        for id in x_index:
            user.append(info[0][id])
        x.append(user)
        y.append(info[0][y_index])
    return x,y


name = 'xxx'   # outcome variable name
draw = True

file = xlrd.open_workbook("xxxx.xlsx") # input data path

X,y = get_data(x_index = list(range(26)))

# compute info_gain
info_gain = mutual_info_classif(X, y)
feature_selection_frequency = np.zeros(len(info_gain))

# set parameters
POP_SIZE = 30    # population size
NGEN = 30        # generations
CXPB = 0.6      # cross probability
MUTPB = 0.1   # mutation probability


# define individual and population
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_bool", lambda: np.random.randint(0, 2))
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=len(info_gain))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# define fitness function
def eval_individual(individual,coef = 0.03):
    selected_features = [i for i in range(len(individual)) 
                         if individual[i] == 1]
    if len(selected_features) == 0:
        return 0.0,
    info_gain_sum = np.sum([info_gain[i] for i in selected_features])
    penalty = len(selected_features) * coef 
    score = info_gain_sum - penalty
    return score,

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", eval_individual)

# main process
def main():
    global feature_selection_frequency
    
    pop = toolbox.population(n=POP_SIZE)

    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + stats.fields

    for gen in range(NGEN):
        random.seed(200)
        np.random.seed(200)
        offspring = algorithms.varAnd(pop, toolbox, cxpb=CXPB, mutpb=MUTPB)
        fits = toolbox.map(toolbox.evaluate, offspring)

        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit        

        for ind in offspring:
            for i in range(len(ind)):
                if ind[i] == 1:
                    feature_selection_frequency[i] += 1
                    
        pop = toolbox.select(offspring, k=len(pop))
        hof.update(pop)
        record = stats.compile(pop)
        logbook.record(gen=gen, nevals=len(offspring), **record)
        print(logbook.stream)

    feature_selection_frequency = feature_selection_frequency / (NGEN * POP_SIZE)
    return pop, logbook, hof

def get_var_name():
    sheet = file.sheet_by_index(0)
    return list(sheet.row_values(0))

# set Grid search range 
param_grid = {
    'POP_SIZE': [30, 50, 70],
    'NGEN': [30, 40, 50],
    'CXPB': [0.6, 0.7, 0.8],
    'MUTPB': [0.1, 0.2, 0.3],
}

# plot fitness curve
def plot_fitness(logbook,name):
    generations = logbook.select("gen")
    avg_fitness = logbook.select("avg")
    max_fitness = logbook.select("max")
    min_fitness = logbook.select("min")
    
    plt.figure(figsize=(15, 10))
    plt.plot(generations, avg_fitness, label="Average Fitness",color='#f16c23')
    plt.plot(generations, max_fitness, label="Max Fitness",color='#2b6a99')
    plt.plot(generations, min_fitness, label="Min Fitness",color='#1b7c3d')
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.title("Fitness Evolution")
    plt.legend()
    plt.grid(True)
    rcParams['pdf.fonttype'] = 42
    plt.savefig(name+' Fitness Evolution.pdf')
    plt.show()

# grid search for genetic algorithm
def grid_search(param_grid):
    global X,y
    
    param_combinations = [dict(zip(param_grid.keys(), values)) for values in itertools.product(*param_grid.values())]
    best_score = float("-inf")
    best_params = None
    for params in param_combinations:
        print(f"Evaluating params: {params}")
        pop = toolbox.population(n=params['POP_SIZE'])
        
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        algorithms.eaSimple(pop, toolbox, cxpb=params['CXPB'], mutpb=params['MUTPB'], ngen=params['NGEN'], 
                            stats=stats, halloffame=hof, verbose=False)

        X = np.array(X)
        y = np.array(y)
        
        best_individual = hof[0][:X.shape[-1]]
        selected_features = [i for i in range(len(best_individual)) 
                             if best_individual[i] == 1]
        if len(selected_features) == 0:
            continue
        
        for feature in selected_features:
            feature_selection_frequency[feature] += 1

        print(selected_features)

        score = eval_individual(best_individual)[0]

        if score > best_score:
            best_score = score
            best_params = params
            best_selected_features = selected_features

    return best_params, best_selected_features

def search():
    best_params, best_selected_features = grid_search(param_grid)
    print(name)
    print("Best parameters found: ", best_params)
    var_name = get_var_name()
    print("Best selected features: ", [var_name[i] for i in best_selected_features])
    return best_selected_features


def draw_two():  
    pop, logbook, hof = main()

    plot_fitness(logbook,name)

if __name__ == "__main__":

    # search first, then set parameters and draw
    if not draw:
        X = np.array(X)
        y = np.array(y)
        selected_features = search()

    else:
        draw_two()


