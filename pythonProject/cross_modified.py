import random
import ga_compound_generation
# in1 = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27]
# in2 = [1,11,12,13,14,15,16,17,18,19,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127]
indv2_list = ga_compound_generation.fitness(ga_compound_generation.population(size=50))
# cross_over_frequency = 0.2
# mutation_rate = 0.2
# print(indv2_list.info())
# print(indv2_list.loc[0].values.tolist())
def to_crossover(indv1, indv2, cross_over_frequency):
    a = random.random()
    if random.random() < cross_over_frequency:
        indv1[1] = indv2[1]
    for each in range(8,len(indv1)):
        if (each ==8) or (each == 18) or (each == 19) or (each == 20):
            if random.random()< cross_over_frequency:
                indv1[each] = indv2[each]
            continue
        if a < cross_over_frequency:
            indv1[each] = indv2[each]
            indv1[3] = indv2[3]
    return indv1
    # print('i',indv1)
# print((to_crossover(in1, in2, cross_over_frequency=0.5)),'\n')

def to_mutation(individual1, mutation_rate):
    individual2 = indv2_list.iloc[random.randrange(20)].values.tolist()
    # print(individual2)
    mut = to_crossover(individual1, individual2, mutation_rate)
    return mut
# print(len(in1),to_mutation(in1, mutation_rate=0.1))

# print(to_mutation(in1, mutation_rate))