from scipy import spatial
import random

RESTAURANTS = [f"rest_{i}" for i in [1,2,3,4,5,6,7,8,9,10]]



def check_candidate(x):
    count5 = 0
    count1 = 0
    for num in x:
        if num==5:
            count5 = count5 + 1
        elif num==1:
            count1 = count1 + 1
    return count5 > 1 and count5 < 4 and count1 > 1 and count1 < 4

def euclidean_sim(x, y, min_v, max_v):
    val = [min_v, max_v]
    xa = [val[i % 2] for i in range(len(x))]
    xb = [val[(i+1) % 2] for i in range(len(x))]
    max_dist = spatial.distance.euclidean(xa,xb)
    
    # print(xa, xb)
    
    x1 = [1] * len(x)
    min_dist = spatial.distance.euclidean(x1,x1)
    
    # print(x1)
    
    norm_dist = (spatial.distance.euclidean(x,y) - min_dist) / (max_dist - min_dist)
    
    return 1 - min(max(norm_dist, 0), 1)


# Thresholds based on previous work -> full calculations+distributions in config-gen-sim.ipynb
t1 = 0.6119055734094894
t2 = 0.4956975236484348

def generate_minorty_conf(n=4, m=10, r=10, options=RESTAURANTS):
    # MINORITY 
    group_conf = list()
    i = 0
    while (i < n):
        new_X = [random.randint(1, r) for i in range(m)]
        if check_candidate(new_X):
        #     print("Candidate: ", new_X)
            selected = True 
            for old_X in group_conf:
        #         print(new_X)
        #         print(old_X)
                if (i < n-1):
                    if euclidean_sim(new_X,old_X,1,r) < t1:
                        selected = False
                else:
                    if euclidean_sim(new_X,old_X,1,r) > t2:
                        selected = False
            if selected:
                group_conf.append(new_X)
                i = i + 1
                #print(new_X)
    restaurant_names = options
    formatted_output = [
        dict(zip(restaurant_names, person_ratings)) for person_ratings in group_conf
    ]
    return formatted_output

def generate_uniform_conf(n=4, m=10, r=10, options=RESTAURANTS):
    group_conf = list()
    i = 0
    while (i < n):
        new_X = [random.randint(1, r) for i in range(m)]
        if check_candidate(new_X):
    #     print("Candidate: ", new_X)
            selected = True 
            for old_X in group_conf:
    #         print(new_X)
    #         print(old_X)
                if euclidean_sim(new_X,old_X,1,r) < t1:
                    selected = False
            if selected:
                group_conf.append(new_X)
                i = i + 1
                #print(new_X)
    restaurant_names = options

    formatted_output = [
        dict(zip(restaurant_names, person_ratings)) for person_ratings in group_conf
    ]
    return formatted_output

def generate_coalitional_conf(n=4, m=10, r=10, options=RESTAURANTS):
    group_conf_1 = list()
    group_conf_2 = list()
    i = 0
    sg1 = int(n / 2)
    sg2 = n - sg1

    #print(sg1, sg2)

    while (i < sg1):
        new_X = [random.randint(1, r) for i in range(m)]
        if check_candidate(new_X):
    #     print("Candidate: ", new_X)
            selected = True 
            for old_X in group_conf_1:
    #         print(new_X)
    #         print(old_X)
                if euclidean_sim(new_X,old_X,1,r) < t1:
                    selected = False
                
            if selected:
                group_conf_1.append(new_X)
                i = i + 1
                #print(new_X)
            
    i = 0
    while (i < sg2):
        new_X = [random.randint(1, r) for i in range(m)]
        if check_candidate(new_X):
    #     print("Candidate: ", new_X)
            selected = True 
            for old_X in group_conf_2:
    #         print(new_X)
    #         print(old_X)
                if euclidean_sim(new_X,old_X,1,r) < t1:
                    selected = False
        
            for old_X in group_conf_1:
    #         print(new_X)
    #         print(old_X)
                if euclidean_sim(new_X,old_X,1,r) > t2:
                    selected = False
                
            if selected:
                group_conf_2.append(new_X)
                i = i + 1
                #print(new_X)

    group_conf_1.extend(group_conf_2)
    restaurant_names = options

    formatted_output = [
        dict(zip(restaurant_names, person_ratings)) for person_ratings in group_conf_1
    ]
    return formatted_output

def generate_divergent_conf(n=4, m=10, r=10, options=RESTAURANTS):
    group_conf = list()
    i = 0
    while (i < n):
        new_X = [random.randint(1, r) for i in range(m)]
        if check_candidate(new_X):
    #     print("Candidate: ", new_X)
            selected = True 
            for old_X in group_conf:
    #         print(new_X)
    #         print(old_X)
                if euclidean_sim(new_X,old_X,1,r) > t2:
                    selected = False
            if selected:
                group_conf.append(new_X)
                i = i + 1
                #print(new_X)
        restaurant_names = options

    formatted_output = [
        dict(zip(restaurant_names, person_ratings)) for person_ratings in group_conf
    ]
    return formatted_output