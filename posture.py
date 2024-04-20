import math as m

def neck_inclination(angle):
    if (angle >=0 and angle <=20):
        score =1
    else:
        score =2
    return score

def neck_tilt_score(ratio):
    if (ratio >=0.3 and ratio <= 1.3):
        score =0
    else:
        score =1
    return score
def trunk_score(angle):
    if angle ==0:
        score =1
    elif (angle >0 and angle <=20):
        score =2
    elif (angle >20 and angle <=60):
        score =3
    elif angle <0:
        score =3
    else:
        score =4
    return score
def trunk_twist_score(ratio):
    if (ratio >=0.3 and ratio <= 1.3):
        score =0
    else:
        score =1
    return score
def uarm_score(angle):
    if angle ==0:
        score =1
    elif (angle >-20 and angle <=20):
        score =1
    elif (angle >20 and angle <=45):
        score =2
    elif (angle >45 and angle <=90):
        score =3
    elif (angle >90):
        score =4
    else:
        score =4
    return score
def larm_score(angle):
    if angle ==0:
        score =1
    elif (angle >0 and angle <=20):
        score =1
    else:
        score =2
    return score

def wrist_score(angle):
    if angle ==0:
        score =1
    elif (angle >-15 and angle <=15):
        score =1
    else:
        score =2
    return score


