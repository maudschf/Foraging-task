
"""
Functions to import for processing code 
"""

import numpy as np
from math import factorial
#import pandas as pd

        

def trapezes_from_patch(patch, width):
    """
    generate the trapezes coordinates surrounding a patch
    inputs:
    patch - coordinates of a patch [[Xa, Ya], [Xb, Yb], [Xc, Yc], [Xd, Yd]]
    width - width of the trapeze in pixels
    outputs:
    coordinates [[Xa, Ya], [Xb, Yb], [Xc, Yc], [Xd, Yd]] for the 4 trapezes.
    
    trapezes_from_patch(SWpatch_coords, 200)
    """

    N = [patch[0], patch[1], [patch[1][0]+width, patch[1][1]+width], [patch[0][0]-width, patch[0][1]+width]]
    E = [patch[1], patch[2], [patch[2][0]+width, patch[2][1]-width], [patch[1][0]+width, patch[1][1]+width]]
    S = [patch[2], patch[3], [patch[3][0]-width, patch[3][1]-width], [patch[2][0]+width, patch[2][1]-width]]
    W = [patch[3], patch[0], [patch[0][0]-width, patch[0][1]+width], [patch[3][0]-width, patch[3][1]-width]]
    return N, E, S, W
    

def points_in_polygon(polygon, pts):
    """
    inputs:
    polygon - coordinates of the trapeze [[Xa, Ya], [Xb, Yb], [Xc, Yc], [Xd, Yd]] or N/E/S/W
    pts - mouse coordinates [[x, y]]
    output:
    returns True/False if pts is inside/outside polygon
    e.g. points_in_polygon([[1300,1650],[1600,1650],[1750,1800],[1150,1800]], [[x, y]])

    The idea is draw an infinite line to the right of [x, y] and count the number of time it 
    crosses the shape, if odd it's inside, if even it's outside.


        P-------------      No hit = 0: outside

         xxxxxxx
        x       x
    P---x-------x-----      2 hits = 0 mod(2): outside
        x       x
        x  P----x-----      1 hit = 1 mod(2): inside
        x       x
         xxxxxxx

    ### Method 2: compare areas
    """

    pts = np.asarray(pts,dtype='float32')
    polygon = np.asarray(polygon,dtype='float32')
    contour2 = np.vstack((polygon[1:], polygon[:1]))
    test_diff = contour2-polygon
    mask1 = (pts[:,None] == polygon).all(-1).any(-1)
    m1 = (polygon[:,1] > pts[:,None,1]) != (contour2[:,1] > pts[:,None,1])
    slope = ((pts[:,None,0]-polygon[:,0])*test_diff[:,1])-(test_diff[:,0]*(pts[:,None,1]-polygon[:,1]))
    m2 = slope == 0
    mask2 = (m1 & m2).any(-1)
    m3 = (slope < 0) != (contour2[:,1] < polygon[:,1])
    m4 = m1 & m3
    count = np.count_nonzero(m4,axis=-1)
    mask3 = ~(count%2==0)
    mask = mask1 | mask2 | mask3
    return mask[0]


def solenoID(currentPatch, currentTrapeze):
    """
    retrieve solenoid# from its location in the maze
    inputs:
    currentPatch - patch the animal is in, str
    currentTrapeze - trapeze the animal entered, str
    output:
    valve#, int
    """
    valve_patch = {'NW':0, 'NE':4, 'SW':8, 'SE':12} # with NE = object 3, NW = 4, SE = 1 and SW= 2
    valve_trapeze = {'N':0, 'E':1, 'S':2, 'W':3}
    # return number/ but can return GPIO for instance
    if str(currentTrapeze) != 'Nope':
        return valve_patch[currentPatch] + valve_trapeze[currentTrapeze]


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less than `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
"""

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')


def angle_between(p1, p2):
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return np.rad2deg((ang1 - ang2) % (2 * np.pi))



####################################################################################################
####################################################################################################

def patchNumber(x, y, resolution) :
    """send back the string indicating the current patch based on the number recieved
        # 0 = 'NE', 1 = 'NW', 2 = 'SE', 3 = 'SW'
    """
    return (x < resolution[0] / 2) * 1 + (y < resolution[1] / 2) * 2

################################
def whichPatch(number) :
    """send back the string indicating the current patch based on the number recieved
        # 0 = 'NE', 1 = 'NW', 2 = 'SE', 3 = 'SW'
    """
    if number == 0:
        return "NE"
    elif number == 1:
        return "NW"
    elif number == 2:
        return "SE"
    else : return "SW"

################################
def is_in_a_goal(xposition, yposition, current_patch, dictionnaire_of_goals) :
    """
    for every goal in the list, test if the position is inside using points_in_polygon 
    return a bool
    """

    in_a_trapeze = False
    for i in dictionnaire_of_goals[current_patch] :
        if points_in_polygon(polygon= dictionnaire_of_goals[current_patch][i], pts = [[xposition, yposition]]) :
            in_a_trapeze = True
    return in_a_trapeze

######################
def coordinate_patch(patch): #give a y coordinate

    """ give the number corresponding to the patch. Patch must be either 'SW', 'NW', 'SE' or 'NE' """

    if patch == "NE" :
        return 1
    elif patch == "NW" :
        return 2
    elif patch == "SE" :
        return 3
    else :
        return 4

################################3
def stay_in_patch(patch, xpositions, ypositions, Resolution) :
    """ check if the point change patch at a given moment. 
    Patch must be either 'SW', 'NW', 'SE' or 'NE' """
    stay_in_place = True
    indice = 0
    max_indice = len(xpositions)

    patch = coordinate_patch(patch) - 1

    while indice < max_indice and stay_in_place :#check for every point of the trajectory if they are in a different patch than the first one
                    if patch != (xpositions[indice] < Resolution[0] / 2) * 1 + (ypositions[indice] < Resolution[1] / 2) * 2 :
                        stay_in_place = False
                    indice += 1

    return stay_in_place


############################
def cut_in_epoch_speed(cut_off_speed, speed_values, time, MINIMAL_DURATION_STOP, MINIMAL_DURATION_EPOCH): 
    """
    cut the trajectory into different part where the animal is moving
    """
    list_of_epochs = []
    speed_size = len(speed_values)
    beginning_epoch = 0

    if speed_size != len(time) : raise ValueError("speed and time have different length")

    for i in range(speed_size) : #
        if speed_values[i] >= cut_off_speed : #if the speed is above the cut-off value
            if beginning_epoch ==0 : beginning_epoch = i #if there were no epoch being studied, this is the beginning of the epoch
        elif beginning_epoch == 0 : pass #if we were not in an epoch, failure to be above the threshold does nothing
        elif list_of_epochs!=[] and (time[beginning_epoch] - time[list_of_epochs[-1][1]] < MINIMAL_DURATION_STOP) : #if the interval with the previous epoch was too short, change its end to the end of this epoch
            list_of_epochs[-1][1] = i-1 ; beginning_epoch = 0
        
        else :
            
            list_of_epochs.append([beginning_epoch, i-1, "N", 0])#by default, every epoch is noted "N" for "not a quarter turn"
            beginning_epoch = 0


    if beginning_epoch == 0 : pass #once the loop is ended check if there is a suitable epoch in memory
    elif list_of_epochs!=[] and (time[beginning_epoch] - time[list_of_epochs[-1][1]] < MINIMAL_DURATION_STOP) : #if the interval with the previous epoch was too short, change its end to the end of this epoch
            list_of_epochs[-1][1] = i-1 ; beginning_epoch = 0
    elif (time[speed_size - 1] - time[beginning_epoch]) < MINIMAL_DURATION_EPOCH : pass
    else :
        list_of_epochs.append([beginning_epoch, speed_size - 1, "N", 0]) #the N at the end is for "not a quarter turn". every epoch is not a quarter turn until proven otherwise
        

    size_len_epoch = len(list_of_epochs) ; a =0#check if the epoch is long enough
    while a < size_len_epoch:           
        if (time[list_of_epochs[a][1]] - time[list_of_epochs[a][0]]) < MINIMAL_DURATION_EPOCH : 
            _ = list_of_epochs.pop(a) #if the epoch is too short to be considerded, discard it
            size_len_epoch -= 1 
        else : a+= 1
    
    for i in range(len(list_of_epochs)) :
        current_point = list_of_epochs[i][0] #get the current beginning of the epoch
        acceleration = (speed_values[current_point + 1] - speed_values[current_point]) / (time[current_point + 1] - time[current_point])
        try :
            previous_acceleration = (speed_values[current_point ] - speed_values[current_point - 1]) / (time[current_point] - time[current_point - 1])
        except :
            previous_acceleration = -1#if this is the first point, there is no previous acceleration, so don't go in the loop

        while previous_acceleration > (0.4 * acceleration) and previous_acceleration > 0:#continue to go further until we reach the end of the acceleration
            current_point = current_point - 1
            try : previous_acceleration = (speed_values[current_point ] - speed_values[current_point - 1]) / (time[current_point] - time[current_point - 1])
            except : previous_acceleration = -1#if this is the first point, there is no previous acceleration, so break out of the loop
        list_of_epochs[i][0] = current_point#change the beginning of the epoch for the beginning of the acceleration


        current_point = list_of_epochs[i][1] #get the current end of the epoch
        try : acceleration = (speed_values[current_point + 1] - speed_values[current_point]) / (time[current_point + 1] - time[current_point])#calculate the acceleration of the segment just AFTER the end of the epoch
        except : acceleration = 1
        previous_acceleration = (speed_values[current_point ] - speed_values[current_point - 1]) / (time[current_point] - time[current_point - 1])#calculate the acceleration of the segment just BEFORE the end of the epoch

        while acceleration < (0.4 * previous_acceleration)  and acceleration < 0:
            current_point = current_point + 1
            try : acceleration = (speed_values[current_point + 1] - speed_values[current_point]) / (time[current_point + 1] - time[current_point])
            except : acceleration = 1
        #change the end of the epoch for the end of the decceleration
        list_of_epochs[i][1] = current_point


    
    
    return list_of_epochs

##############################
def calcul_angle(ycoordinate, ecart, xcoordinate ) :
    angles  = np.array([np.angle(xcoordinate[i]- xcoordinate[i-ecart] + (ycoordinate[i] - ycoordinate[i-ecart]) * 1j , deg= True) for i in range(ecart, len(xcoordinate))])
    angles = np.insert(angles, obj= 0, values= np.zeros(ecart ))#the calcul of angles change the size of the data. To avoid it, add as much time the first value
    
    return angles #return the orientation of the mouse across time and the modified epochs

##############################
def analysis_trajectory(time, xgauss, ygauss,
                        collection_trapeze, dinfo,
                        cut_speed, ecart_angle, Resolution, MIN_DURATION_STOP, MIN_DURATION_EPOCH) :
    """ Arguments =
    time, xgauss, ygauss : the time of each frame in the TXY csv file and the smoothen positions;
    collection_trapeze : dictionnary with an entry for each patch containing each a dictionnary on their side containing the angles coordinate of the detection trapeze;
    dinfo the dataframe containing the informations of the turnsinfo csv corresponding to the sequence;
    cut_speed : speed under which the mouse is considerd to not be moving;
    ecart_angle : ecart between two frame used to calculate the angle, speed and acceleration;
    Resolution : resolution of the setup in pixels (the size in m is 0.84. if this change, the code must be updqted manualy);
    MIN_DURATION_STOP : minimal duration accepted for a stop;
    MIN_DURATION_EPOCH : minimal duration considered for a stop

    Output = (all are list)
    distance done between this frame and the last one (in cm);
    speed at the moment corresponding to time_average in cm.s-1;
    time_average time at the moment corresponding to the speed;
    acceleration of the mouse at the time t+1 in cm.s-2;
    direction of the mouse at the moment of time average in degree;
    angular speed at the given time t+1 in degre.s-2;

    list_epoch : list of the epochs under the form  [indice of the first frame of the epoch, indice of the last frame of the epoch, indicator]
    See documentation on indicator for more informations
    """

    # compute the distance but on the data with the gaussian filter
    distances_gauss = np.array([((((xgauss[i]-xgauss[i-1])**2)+((ygauss[i]-ygauss[i-1])**2))**0.5) for i in range(1,len(ygauss))]) 
    distances_gauss = np.insert(distances_gauss *(0.84/Resolution[0]), 0, 0)


    #because the distance is computed using two points, it does no longer correspond to time. To fix it, the average of the time used the calculate the distance is used
    timebeweenframe=np.insert(np.diff(time), 0, 1)#get the gap between the frames. Add 1 at the beginning to have a consistant size (any value is possible, it will divide 0)
    #compute the speed in m/s
    speeds_gauss =np.divide(distances_gauss,timebeweenframe)
    #get the speed in cm/s and add a speed of 0 at the beginning to keep the same data size
    speeds_gauss = speeds_gauss * 100
    list_epochs = cut_in_epoch_speed(cut_off_speed= cut_speed, speed_values = speeds_gauss, time = time, MINIMAL_DURATION_STOP= MIN_DURATION_STOP, 
                                     MINIMAL_DURATION_EPOCH= MIN_DURATION_EPOCH) #calculate the epochs with the true cut_off speed and store it

    #calculate the orientation with the chosen value and get the changed epochs
    angles = calcul_angle(ycoordinate= ygauss, ecart= ecart_angle, xcoordinate= xgauss)
    time_average = np.array([time[0]]*ecart_angle + [(time[i] + time[i-ecart_angle]) /2 for i in range(ecart_angle, len(time))])

    angles_relatifs = np.insert(np.diff([angles, time_average])[0], obj= 0, values= np.zeros(1 )) #derive les angles par rapport au temps
    angular_speed = [360  + x if x < -180 else -360 + x if x>180 else x for x in angles_relatifs] #correct for the brutal acceleration when angle pass from -180 to 180

    #calcul of acceleration
    acceleration = np.insert(np.diff([speeds_gauss, time_average])[0], obj = 0, values= np.zeros(1))#derive speed relative to time


    #Advance analysis = identify the quarter turns, the trajectory towards and between objetcs
    #format of a quarter turn indicator : [0] = 'Q' for quarter turn     [1] = 'k'/'w' for counterclockwise / clockwise   
    # [2] = 'O'/'E'/'B'/'G'/'H' for wrong object /extra turn / bad direction / Good / double wrong             
    # [3-4] = patch

    #format for between objects indicator : [0] = 'B' for between object    [1 - 2] = previous patch    [3-4] = current patch
    # [5] = 'n'/'r' for non-rewarded/ rewarded (if multiple turns are done in the movement, only the last one is considered)
    in_an_epoch_but_no_quarter = [] #will contain a list under the form [time, corresponding epoch, bool rewarded]

    for a in range(dinfo.index[0], dinfo.index[-1]):#the epochs are written as "not a quarter" by default. We just need to change it for those which are
        aprime = a - dinfo.index[0]
        not_past_nor_found = True; i = 0; turn_time = dinfo.loc[dinfo.index[aprime], "time"]    #dinfo.iat[aprime , 0]
        if time[list_epochs[-1][1]] < turn_time : #if the last epoch end before the recorded turn, discard the turn
            not_past_nor_found = False
        
        while not_past_nor_found :
            if time[list_epochs[i][1]] < turn_time : i+=1 #if the end of the epoch is before the time of the turn, the epoch does not contain the turn so try the next epoch. We work with the original list of epochs
            elif time[list_epochs[i][0]] > turn_time :#if we reach a point where the beginning of the epoch is after the turn, then the turn was not in an epoch
                not_past_nor_found = False
            else : #if the time is in this epoch, test if this is a true quarter turn
                    #check if the beginning of the epoch (movement) is in the polygon it's supposed to                                                                             #check if the beginning of the epoch (movement) is in the polygon it's supposed to 

                #set the value of  epoch[3] for this epoch to the number of reward the aniimal had at the beginning off the movement
                list_epochs[i][3] =  dinfo.loc[dinfo.index[aprime - 1], "totalnberOfRewards"] #dinfo.iat[aprime -1, 14]

                if points_in_polygon(polygon= collection_trapeze[dinfo.loc[dinfo.index[aprime], "currentPatch"]][dinfo.loc[dinfo.index[aprime], "previousTrapeze"]], pts = [[xgauss[list_epochs[i][0]], ygauss[list_epochs[i][0]]]]) and points_in_polygon(polygon = collection_trapeze[dinfo.loc[dinfo.index[aprime], "currentPatch"]][dinfo.loc[dinfo.index[aprime], "currentTrapeze"]], pts= [[xgauss[list_epochs[i][1]], ygauss[list_epochs[i][1]]]]) :
                    #current patch is obtained from a number between 0 and 3 indicating in which patch it is (True = 1, False = 0)
                    current_patch = whichPatch((xgauss[list_epochs[i][0]] < Resolution[0] / 2) * 1 + (ygauss[list_epochs[i][0]] < Resolution[1] / 2) * 2)

                    #check if the mouse does not go to another patch. If it does, it is not a quarter turn
                    if stay_in_patch(current_patch, xgauss[list_epochs[i][0] : list_epochs[i][1] + 1], ygauss[list_epochs[i][0] : list_epochs[i][1] + 1], Resolution) :
                        if int(dinfo.iat[aprime, 7]) == 90 : w_turn = "k"  #add a marker depending of the type of turn
                        else : w_turn = "w"

                        #select the type of turn 
                        if len(dinfo.loc[dinfo.index[aprime], "typeOfTurn"]) == 6 : t_turn = 'E' #E stand for Extra turn
                        elif dinfo.loc[dinfo.index[aprime], "typeOfTurn"][0] == 'b' : 
                            if dinfo.loc[dinfo.index[aprime], "typeOfTurn"][2] == 'b' :t_turn = 'H' #H for horrible (neither good direction nor good good object)
                            else : t_turn = 'O' # O stand for wrong Object
                        elif dinfo.loc[dinfo.index[aprime], "typeOfTurn"][2] == 'b' : t_turn = 'B'# B stand for Bad turn
                        elif dinfo.loc[dinfo.index[aprime], "typeOfTurn"][0] == 'e' : t_turn = 'X' # X for exploration
                        else : t_turn = 'G' # G stands for Good turn

                        if dinfo.loc[dinfo.index[aprime], "Rewarded"] : reward = "R"
                        else : reward = "N"

                        list_epochs[i][2] = "Q" + w_turn + t_turn + current_patch + reward# Q for quarter turn. 
                    else : in_an_epoch_but_no_quarter += [(turn_time, i, dinfo.loc[dinfo.index[aprime], "Rewarded"])]
                else : 
                    in_an_epoch_but_no_quarter += [(turn_time, i, dinfo.loc[dinfo.index[aprime], "Rewarded"])]

                not_past_nor_found = False # the correct epoch was found, no need to continue

    for a in range(len(list_epochs)) :
        if list_epochs[a][2][0] == "N" : #if the epoch is not a quarter_turn, look at if it can be either a movement between objects or a movement towards an object
            #current patch is a number between 0 and 3 indicating in which patch it is (True = 1, False = 0)
            # 0 = NE, 1 = NW, 2 = SE, 3 = SW
            current_patch = whichPatch((xgauss[list_epochs[a][1]] < Resolution[0] / 2) * 1 + (ygauss[list_epochs[a][1]] < Resolution[1] / 2) * 2)        
            
            #if the epoch end in a trapeze it's either a movement towards an object or a movement between objects, or a very small exploration epoch
            if is_in_a_goal(xgauss[list_epochs[a][1]], ygauss[list_epochs[a][1]], current_patch, collection_trapeze) :
                previous_patch = whichPatch((xgauss[list_epochs[a][0]] < Resolution[0] / 2) * 1 + (ygauss[list_epochs[a][0]] < Resolution[0] / 2) * 2)
                
                #check if the beginning of the epoch was also in a trapeze
                if is_in_a_goal(xgauss[list_epochs[a][0]], ygauss[list_epochs[a][0]], previous_patch, collection_trapeze) :
                    if current_patch == previous_patch :
                        #we consider two possibilities in this case : either this is a small exploration trajectory or 
                        # the animal move to multiple objects while trying to find the reward and end in the same patch in a between object trajectory
                        if not stay_in_patch(current_patch, xgauss[list_epochs[a][0] : list_epochs[a][1] + 1], ygauss[list_epochs[a][0] : list_epochs[a][1] + 1], Resolution) :
                            #then it's a between object trajectory
                            list_epochs[a][2] = "B" + previous_patch + current_patch + 'n' # n for n rewards
                        #else nothing, the exploratory trajectory are marked by an 'N' Which is the default
                    
                    #if previous_patch and current patch are different, it's a trajectory between object
                    else : list_epochs[a][2] = "B" + previous_patch + current_patch + 'n' # n for no rewards
                #if the beginning of the epoch is not in a goal, it is considered a trajectory toward an object
                else : list_epochs[a][2] = "T" + current_patch
    
    return distances_gauss, speeds_gauss, time_average, acceleration, angles, angular_speed, list_epochs
