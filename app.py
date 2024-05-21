from flask import Flask, render_template, request, jsonify
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') 
import pandas as pd
import json
from copy import deepcopy
from itertools import permutations
import os
from collections import defaultdict
import psutil
app = Flask(__name__)

storage_boxes=pd.DataFrame()
storage_truck_spec={}

max_memory_usage = 0  # Initialize max memory usage

def memory_usage():
    global max_memory_usage  # Access the global variable
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / (1024 * 1024)  # in MB
    max_memory_usage = max(max_memory_usage, memory_mb)  # Update max memory usage if needed
    return memory_mb


truck_specs = {
    "General Purpose container 20'": {
        "length_container":5900,
        "width_container":2352,
        "height_container":2393,
        "max_weight": 32500,
        # Add more specifications as needed
    },
    "General Purpose container 40'": {
        "length_container":12032,
        "width_container":2352,
        "height_container":2395,
        "max_weight": 32500,
        # Add more specifications as needed
    },
    "High - Cube General Purpose container 40'": {
        "length_container":12032,
        "width_container":2432,
        "height_container":2700,
        "max_weight": 32500,
    },
    # Add more specifications as needed
}



def dataProcess_2(df,truck_spec):
    length_container = truck_spec['length_container']
    width_container = truck_spec['width_container']
    height_container = truck_spec['height_container']
    max_weight= truck_spec['max_weight']

    # Accessing columns using bracket notation
    length = df['Length']
    width = df['Width']
    height = df['Height']
    # num_boxes_per_strip_column = df['NumOfBoxesPerStrip']
    # total_num_strips_column = df['TotalNumStrips']
    # rem_boxes_column = df['Rem_Boxes']
    numOfcases = df['TotalCases']
    # marked_column = df['Marked']
    rotation_allowed = df['Alpha(rotation about Z-axis)']
    gross_weight = df['GrossWeight']

    

    class Product:
        def __init__(self, length, width, height,grossWeight,numberOfCases):
            self.length = length
            self.width = width
            self.height = height
            self.grossWeight = grossWeight
            # self.netWeight=netWeight
            # self.temperature=temperature
            # self.volume=volume
            self.numberOfCases=numberOfCases



    class Container:
        def __init__(self, length, width, height, max_weight, front_axle_weight, rear_axle_weight, front_axle_distance, rear_axle_distance):
            self.length = length
            self.width = width
            self.height = height
            self.max_weight = max_weight
            self.front_axle_weight = front_axle_weight
            self.rear_axle_weight = rear_axle_weight
            self.front_axle_distance = front_axle_distance
            self.rear_axle_distance = rear_axle_distance

    def create_strip_list(box, container):

        box_len = float(box.length)
        box_width = float(box.width)
        box_height = float(box.height)

        container_len = float(container.length)
        container_width = float(container.width)
        container_height = float(container.height)

        if box_len < container_len and box_width < container_width:
            num_of_boxes_fit = container_height//box_height
            return [box_len,box_width,box_height,num_of_boxes_fit]

        else:
            return []



    front_axle_weight = 16000
    rear_axle_weight = 12400
    front_axle_dist = 2890
    rear_axle_dist = 3000



    container_toFit = Container(length_container,width_container,height_container,max_weight,front_axle_weight,rear_axle_weight,front_axle_dist,rear_axle_dist)


    num_typesOfBoxes = len(gross_weight)
    # print(num_typesOfBoxes)
    box_set = []

    for i in range(num_typesOfBoxes):
        box = Product(length[i],width[i],height[i],gross_weight[i],numOfcases[i])
        box_set.append(box)
                                    
    strip_list =[]
    for box in box_set:
        strips= create_strip_list(box,container_toFit)
        strip_list.append(strips)



    # print(container_toFit.length)

    ## Creating simple strips (all of same type of boxes.)

    def remBoxes(box_set,strip_list):
        rem_boxes = []
        num_of_strips_per_boxType=[]
        i=0
        for box in box_set:
            num = int(box.numberOfCases)
            # print(i)
            num_per_strip = strip_list[i][3]
            num_of_strips= num//num_per_strip
            rem = num%num_per_strip
            num_of_strips_per_boxType.append(num_of_strips)
            rem_boxes.append(rem)
            i+=1
        return rem_boxes, num_of_strips_per_boxType

    rem_boxes, num_strips_box = remBoxes(box_set,strip_list)

    for i in range(len(strip_list)):
        strip_list[i].append(num_strips_box[i])
        strip_list[i].append(rem_boxes[i])
        strip_list[i].append(int(box_set[i].numberOfCases))  # Indicating that it has been used
        strip_list[i].append(True)
        strip_list[i].append(rotation_allowed[i])
        strip_list[i].append(float(box_set[i].grossWeight))

    # print(strip_list)

    for i in range(len(strip_list)):
        for j in range(len(strip_list[i])):
            strip_list[i][j] = float(strip_list[i][j])
    # strip_list
    df_new = pd.DataFrame(strip_list)
    df_new.columns= ['Length','Width','Height','NumOfBoxesPerStrip','TotalNumStrips','Rem_Boxes','TotalCases','Marked', 'Alpha(rotation about Z-axis)','GrossWeight']
    # Drop the 'Marked' column
    # df_new.drop(columns=['Marked'], inplace=True)

    # Add 'BoxNumber' column
    df_new['BoxNumber'] = df_new.index

    # Define color dictionary
    colors = {0: 'red', 1: 'blue', 2: 'yellow', 3: 'orange', 4: 'green', 5: 'violet', 6: 'white', 7: 'indigo', 8: 'purple'}

    # Add 'Color' column
    df_new['Color'] = df_new['BoxNumber'].map(colors)
    df_new = df_new[['BoxNumber', 'Color'] + [col for col in df_new.columns if col not in ['BoxNumber', 'Color']]]
    df_new['Rem_Strips'] = df['Rem_Strips']

    return df_new,container_toFit,strip_list


def dataProcess_1(data,truck_spec):
    length_container = truck_spec['length_container']
    width_container = truck_spec['width_container']
    height_container = truck_spec['height_container']
    max_weight= truck_spec['max_weight']

    col_list= data.columns.tolist()

    gross_weight = data[col_list[0]].tolist()
    net_weight = data[col_list[1]].tolist()
    vol = data[col_list[2]].tolist()
    height = data[col_list[6]].tolist()
    numOfcases = data[col_list[7]].tolist()
    rotation_allowed = data[col_list[8]].tolist()
    temperature = data[col_list[3]].tolist()
    length = data[col_list[4]].tolist()
    width = data[col_list[5]].tolist()
    

    class Product:
        def __init__(self, length, width, height,grossWeight,netWeight,temperature,volume,numberOfCases):
            self.length = length
            self.width = width
            self.height = height
            self.grossWeight = grossWeight
            self.netWeight=netWeight
            self.temperature=temperature
            self.volume=volume
            self.numberOfCases=numberOfCases



    class Container:
        def __init__(self, length, width, height, max_weight, front_axle_weight, rear_axle_weight, front_axle_distance, rear_axle_distance):
            self.length = length
            self.width = width
            self.height = height
            self.max_weight = max_weight
            self.front_axle_weight = front_axle_weight
            self.rear_axle_weight = rear_axle_weight
            self.front_axle_distance = front_axle_distance
            self.rear_axle_distance = rear_axle_distance

    def create_strip_list(box, container):

        box_len = float(box.length)
        box_width = float(box.width)
        box_height = float(box.height)

        container_len = float(container.length)
        container_width = float(container.width)
        container_height = float(container.height)

        if box_len < container_len and box_width < container_width:
            num_of_boxes_fit = container_height//box_height
            return [box_len,box_width,box_height,num_of_boxes_fit]

        else:
            return []



    front_axle_weight = 16000
    rear_axle_weight = 12400
    front_axle_dist = 2890
    rear_axle_dist = 3000



    container_toFit = Container(length_container,width_container,height_container,max_weight,front_axle_weight,rear_axle_weight,front_axle_dist,rear_axle_dist)


    num_typesOfBoxes = len(gross_weight)
    # print(num_typesOfBoxes)
    box_set = []

    for i in range(num_typesOfBoxes):
        box = Product(length[i],width[i],height[i],gross_weight[i],net_weight[i],temperature[i],vol[i],numOfcases[i])
        box_set.append(box)
                                    
    strip_list =[]
    for box in box_set:
        strips= create_strip_list(box,container_toFit)
        strip_list.append(strips)



    # print(container_toFit.length)

    ## Creating simple strips (all of same type of boxes.)

    def remBoxes(box_set,strip_list):
        rem_boxes = []
        num_of_strips_per_boxType=[]
        i=0
        for box in box_set:
            num = int(box.numberOfCases)
            # print(i)
            num_per_strip = strip_list[i][3]
            num_of_strips= num//num_per_strip
            rem = num%num_per_strip
            num_of_strips_per_boxType.append(num_of_strips)
            rem_boxes.append(rem)
            i+=1
        return rem_boxes, num_of_strips_per_boxType

    rem_boxes, num_strips_box = remBoxes(box_set,strip_list)

    for i in range(len(strip_list)):
        strip_list[i].append(num_strips_box[i])
        strip_list[i].append(rem_boxes[i])
        strip_list[i].append(int(box_set[i].numberOfCases))  # Indicating that it has been used
        strip_list[i].append(True)
        strip_list[i].append(rotation_allowed[i])
        strip_list[i].append(float(box_set[i].grossWeight))

    # print(strip_list)

    for i in range(len(strip_list)):
        for j in range(len(strip_list[i])):
            strip_list[i][j] = float(strip_list[i][j])
    # strip_list
    df = pd.DataFrame(strip_list)
    df.columns= ['Length','Width','Height','NumOfBoxesPerStrip','TotalNumStrips','Rem_Boxes','TotalCases','Marked', 'Alpha(rotation about Z-axis)','GrossWeight']
    # Drop the 'Marked' column
    # df.drop(columns=['Marked'], inplace=True)

    # Add 'BoxNumber' column
    df['BoxNumber'] = df.index

    # Define color dictionary
    colors = {0: 'red', 1: 'blue', 2: 'yellow', 3: 'orange', 4: 'green', 5: 'violet', 6: 'white', 7: 'indigo', 8: 'purple'}

    # Add 'Color' column
    df['Color'] = df['BoxNumber'].map(colors)
    df = df[['BoxNumber', 'Color'] + [col for col in df.columns if col not in ['BoxNumber', 'Color']]]
    df['Rem_Strips'] = df['TotalNumStrips']


    return df,container_toFit,strip_list



@app.route('/')
def index():
    return render_template('index.html')


def perform_computation(df,container_toFit,strip_list,key,roll):


    def widthRem(width_rem,strip_list):
        for box in strip_list:

            if box[1] <= width_rem:
                return True
        
        return False


    def choose_best_dimension(x,end_x,z,strip_list,container,stored_plac):

        width_container = float(container.width)
        height_container = float(container.height)
        depth_container = float(container.length)

        if end_x ==0:
            end_x = width_container

        width_rem = end_x-x
        box_num = -1
        checker = widthRem(width_rem,strip_list)
        if checker == False:
            width_rem = width_container

        mini = -1e9
        index = 0
        # print("strip_list_in cbd",strip_list)
        best_width = []
        for box_dim in strip_list:
            # if 
            #     continue
            # print(df.at[index,'Rem_Strips'])
            if(box_dim[7]==0 and df.at[index,'Rem_Strips']==0):
                index+=1
                continue
            length_diff = 1e4
            num_box = width_rem//box_dim[1]
            total_num_strips = box_dim[4]
            height = box_dim[2]
            fill = True
            if num_box > total_num_strips:
                fill= False
            perc = (num_box*box_dim[1]/width_rem)
            if len(stored_plac)>0:
                prev_length = stored_plac[len(stored_plac)-1][7]
                length_diff = abs(box_dim[0]-prev_length)
                # print("TES")
            best_width.append([index,perc,box_dim[1],fill,length_diff,((height_container//height)*height)])
            index+=1
        # print(strip_list)
        # print(best_width)
        sorted_data = sorted(best_width, key=lambda x: (x[3], x[1], x[5],x[2]), reverse=True)
        # print(sorted_data)
        n =1    #Currently taking the best only based on the efficiency.
        ind =0
        maxi =1e5
        best_box =-1
        while ind < len(sorted_data) and ind <= n :
            if maxi > sorted_data[ind][4] and sorted_data[ind][3]:
                maxi = min(maxi,sorted_data[ind][4])
                # print("maxi",maxi)
                best_box = sorted_data[ind][0]
            ind+=1
        if best_box==-1 and len(sorted_data)>1:
            best_box = sorted_data[0][0]

        return best_box

        # return ans

    def invertOrNot(x,end_x,box_num,box_length,box_width,box_height,width_container,total_strips):
        rem_width= end_x-x
        if rem_width <=0:
            return False
        num_strips_required = rem_width//box_width
        if num_strips_required > total_strips:
            return False
        perc_nonInverted = ((rem_width//box_width)*box_width)/rem_width
        perc_Inverted = ((rem_width//box_length)*box_length)/rem_width


        if perc_Inverted > perc_nonInverted:
            return True

        else:
            return False



    def findoptlen(prev_row,x,y,end_x,box_width,row,prev_y,prev_row_num):
        prev_row = [prev_row for prev_row in prev_row if ((prev_row[0] != prev_row[2])) ]
        # print("Row_inside_optlen",row)
        # print(prev_row)
        # print("X_inside_optlen",x)
        i = 0
        while i<len(prev_row) and(prev_row[i][6]>row or prev_row[i][6]<row-1):
            i+=1
        
        # print("inside_opt_len_iterator",i)
        
        if end_x+box_width> float(container_toFit.width) and len(prev_row)<=1:
            end_x= container_toFit.width
        else:
            end_x = deepcopy(prev_row[i][2])
        
        # print("endX_inside_optlen",end_x)

            
        # print("PREV_ROW",prev_row)
        # if i < len(prev_row):
        y = prev_row[i][3]
        prev_row_num=deepcopy(prev_row[i][6])
        prev_row[i][5] = 0
        prev_y = deepcopy(prev_row[i][1])
        del prev_row[i]
        return y, end_x,row,prev_row,prev_y,prev_row_num
            
            


    def after_plac(x,y,z,end_x,box_num,strip_list,container,ax,color,curr_weight,stored_plac,row,strip_storage,prev_y,prev_row,prev_row_num,vol_occ,y_min,df,vol_wasted):
        width_container = float(container.width)
        height_container = float(container.height)
        depth_container = float(container.length)
        max_weight = container.max_weight

        # print("UES",strip_list)

        init_x=x
        init_y=y
        init_z=z

        total_strips = deepcopy(df.at[box_num,'Rem_Strips'])
        # print(total_strips)
        box_length = deepcopy(float(strip_list[box_num][0]))
        box_width = deepcopy(float(strip_list[box_num][1]))
        box_height = deepcopy(float(strip_list[box_num][2]))
        box_weight = deepcopy(float(strip_list[box_num][9]))

        change_init = invertOrNot(x,end_x,box_num,box_length,box_width,box_height,width_container,total_strips)
        if change_init == True:
            
            y = y+box_length
            temp = deepcopy(box_length)
            box_length = deepcopy(box_width)
            box_width = deepcopy(temp)
            prev_row[len(prev_row)-1][1] = prev_row[len(prev_row)-1][1] + box_width- box_length
            # storage_strip[len(storage_strip)-1][1] = storage_strip[len(storage_strip)-1][1] + box_width- box_length
            y = y-box_length



        while total_strips > 0 and y > 0 and curr_weight+box_weight<max_weight:
            if x + box_width <= end_x and x + box_width <= width_container:  ## added the max weight check constraints
                # print("prev_row",prev_row)
                if len(prev_row) >1 and prev_row[len(prev_row)-2][1]>=0 and y-prev_row[len(prev_row)-2][1] > box_length and row== prev_row[len(prev_row)-2][6]:
                    # x-= box_width
                    
                    # print("x",x)
                    # print("y",y)

                    # print("yes went inside")
                    # y = y-box_length
                    num_strips = strip_list[box_num][3]
                    storage_strip.append([y,strip_list[box_num][3] * strip_list[box_num][9]])
                    while num_strips > 0 and curr_weight+box_weight < max_weight:
                        ax.bar3d(x, y, z, box_width, box_length, box_height, color=color, edgecolor='black')
                        z += box_height
                        num_strips -= 1
                        curr_weight+=strip_list[box_num][9]
                        vol_occ+=box_length*box_width*box_height
                        # strip_storage.append[y,]
                    
                    

                    # storage_strip.append([x,y,z,box_length,box_width,box_height,box_num])
                    # x += box_width.
                    z = 0
                    total_strips -= 1
                    # prev_row[len(prev_row)-1][2]=deepcopy(x)
                    if total_strips==0:
                        x+=box_width
                        continue
                    if total_strips>0:
                        if y-box_length>=0:
                            y-=box_length
                        else:
                            continue
                        prev_row[len(prev_row)-1][1]=deepcopy(y)
                        # storage_strip[len(storage_strip)-1][1]=deepcopy(y)
                        

                        num_strips = strip_list[box_num][3]
                        # print("UES")
                        storage_strip.append([y,strip_list[box_num][3] * strip_list[box_num][9]])

                        while num_strips > 0 and curr_weight+box_weight < max_weight:
                            ax.bar3d(x, y, z, box_width, box_length, box_height, color=color, edgecolor='black')
                            z += box_height
                            num_strips -= 1
                            curr_weight+=strip_list[box_num][9]
                            vol_occ+=box_length*box_width*box_height
                       

                        
                        x+=box_width
                        # print("Yes goes in",end_x)
                        # print("Y_before",y)

                        if x+box_width<=end_x:
                            y+=box_length
                        z=0
                        # print("Y_after",y)

                        total_strips-=1

                    
                    # if total_strips==0:


                else:
                    num_strips = strip_list[box_num][3]
                    storage_strip.append([y,strip_list[box_num][3] * strip_list[box_num][9]])

                    while num_strips > 0 and curr_weight+box_weight < max_weight:
                        ax.bar3d(x, y, z, box_width, box_length, box_height, color=color, edgecolor='black')
                        z += box_height
                        num_strips -= 1
                        curr_weight+=strip_list[box_num][9]
                        vol_occ+=box_length*box_width*box_height
                   
                    

                    # storage_strip.append([x,y,z,box_length,box_width,box_height,box_num])
                    x += box_width
                    z = 0
                    total_strips -= 1

                

            else: 
                # x = end_x
                y_min = min(y_min,y)
                if x+box_width> width_container:
                    # print("Inside1",end_x-x)
                    # print("vol_wasted",abs(width_container-x)*box_length*height_container)

                    vol_wasted += abs(width_container-x)*box_length*height_container

                    x = width_container
                

                prev_row[len(prev_row)-1].append(x)
                prev_row[len(prev_row)-1].append(y)
                prev_row[len(prev_row)-1].append(box_length)
                prev_row[len(prev_row)-1].append(1)
                prev_row[len(prev_row)-1].append(row)
                
                

                # print("Y",y)
                # print("prev_row_before",prev_row)
                # print("prev_y",prev_y)
                # print("x",x)
                # print("Row",row)
                index=0
                if x + box_width > width_container:
                    # efficiency_new.append([y_min,row])
                    row+=1
                    x = 0
                    z = 0
                while index<len(prev_row) and (prev_row[index][6]!=row-1):
                    index+=1

                # print("INdex",index)
                # print("prev_row_num",prev_row_num)
                rem=0
                rem_y=0
                went_in_1 = False
                went_in_2 =False
                if x!=0 and end_x-x< (x+box_width)-end_x: 
                    ##Checks the better one between putting one extra or shifting according to the previous row
                    # print("Inside2",end_x-x)
                    # vol_wasted += (end_x-x)*abs(prev_row[len(prev_row)-1][4]-box_length)*height_container
                    rem=deepcopy(abs(x-end_x))
                    if len(prev_row) >1 and index<len(prev_row) and prev_row[index][6] == prev_row_num and prev_row[index][3] > prev_y:
                        went_in_1=True
                    x +=(end_x-x)
                else:
                    if len(prev_row) >1 and index<len(prev_row) and prev_row[index][6] == prev_row_num and prev_row[index][3] > prev_y and x + box_width <= width_container:
                        went_in_2=True
                        # print("Inside3",end_x-x)
                        # vol_wasted += (x+box_width-end_x)*abs(prev_row[len(prev_row)-1][4]-box_length)*height_container
                        num_strips = strip_list[box_num][3]
                        # print("UES")
                        storage_strip.append([y,strip_list[box_num][3] * strip_list[box_num][9]])

                        while num_strips > 0 and curr_weight+box_weight < max_weight:
                            ax.bar3d(x, y, z, box_width, box_length, box_height, color=color, edgecolor='black')
                            z += box_height
                            num_strips -= 1
                            curr_weight+=strip_list[box_num][9]
                            vol_occ+=box_length*box_width*box_height
                        

                        

                        # storage_strip.append([x,y,z,box_length,box_width,box_height,box_num])
                        x += box_width
                        z = 0
                        total_strips -= 1
                        prev_row[len(prev_row)-1][2]=deepcopy(x)
                        rem=deepcopy(abs(x-end_x))
                        # storage_strip[len(storage_strip)-1][2]=deepcopy(x)


                p_y=0
                if went_in_1 is True:
                    p_y = deepcopy(y+box_length)
                elif went_in_1 is False and went_in_2 is False:
                    p_y = deepcopy(box_length)
                else:
                    p_y = deepcopy(y+box_length)




                y,end_x,row,prev_row,prev_y,prev_row_num= (findoptlen(prev_row,x,y,end_x,box_width,row,prev_y,prev_row_num))
                # print("y_diff",abs(p_y-y))
                # print("rem",rem)

                if x!=0 and went_in_1 is True:
                    # print("went_1_true:",y-box_length-p_y)
                    vol_wasted += abs(y-box_length-p_y)*rem*height_container
                elif x!=0 and went_in_1 is False and went_in_2 is False:
                    # print("went_1_false and went_2_false",p_y)
                    vol_wasted += abs(p_y)*rem*height_container
                else:
                    if x!=0:
                        # print("all others",y-p_y)
                        vol_wasted += abs(y-p_y)*rem*height_container
                y_min = min(y_min,y)

         
                if end_x+box_width>=width_container:  
                    end_x = deepcopy(width_container)
                
                change = invertOrNot(x,end_x,box_num,box_length,box_width,box_height,width_container,total_strips)
        

                if change == True:
                    temp = deepcopy(box_length)
                    box_length = deepcopy(box_width)
                    box_width = deepcopy(temp)


                y = y -1-box_length
                y_min = min(y_min,y)

                prev_row.append([x,y])
                # storage_strip.append([x,y])


        if y<0 and total_strips!=0 and curr_weight+box_weight<max_weight:
            y+=box_length
            y-=box_width
            if y>0:
                temp = deepcopy(box_length)
                box_length = deepcopy(box_width)
                box_width = deepcopy(temp)
                while total_strips > 0 and y > 0:
                    if x + box_width <= end_x and x + box_width <= width_container:  ## added the max weight check constraints
                        # print("prev_row",prev_row)
                        if len(prev_row) >1 and prev_row[len(prev_row)-2][1]>=0 and y-prev_row[len(prev_row)-2][1] > box_length and row== prev_row[len(prev_row)-2][6]:
                            # x-= box_width
                            
                            # print("x",x)
                            # print("y",y)

                            # print("yes went inside")
                            # y = y-box_length
                            num_strips = strip_list[box_num][3]
                            # print("UES")
                            
                            storage_strip.append([y,strip_list[box_num][3] * strip_list[box_num][9]])
                            
                            while num_strips > 0 and curr_weight+box_weight < max_weight:
                                ax.bar3d(x, y, z, box_width, box_length, box_height, color=color, edgecolor='black')
                                z += box_height
                                num_strips -= 1
                                curr_weight+=strip_list[box_num][9]
                                vol_occ+=box_length*box_width*box_height
        

                            # storage_strip.append([x,y,z,box_length,box_width,box_height,box_num])
                            # x += box_width.
                            z = 0
                            total_strips -= 1
                            # prev_row[len(prev_row)-1][2]=deepcopy(x)
                            if total_strips==0:
                                x+=box_width
                                continue
                            if total_strips>0:
                                if y-box_length>=0:
                                    y-=box_length
                                else:
                                    continue
                                prev_row[len(prev_row)-1][1]=deepcopy(y)
                                # storage_strip[len(storage_strip)-1][1]=deepcopy(y)
                                

                                num_strips = strip_list[box_num][3]
                                # print("UES")
                                storage_strip.append([y,strip_list[box_num][3] * strip_list[box_num][9]])

                                while num_strips > 0 and curr_weight+box_weight < max_weight:
                                    ax.bar3d(x, y, z, box_width, box_length, box_height, color=color, edgecolor='black')
                                    z += box_height
                                    num_strips -= 1
                                    curr_weight+=strip_list[box_num][9]
                                    vol_occ+=box_length*box_width*box_height
                                x+=box_width

                                # print("Yes goes in",end_x)
                                # print("Y_before",y)

                                if x+box_width<=end_x:
                                    y+=box_length
                                z=0
                                # print("Y_after",y)

                                total_strips-=1

                            
                            # if total_strips==0:


                        else:
                            num_strips = strip_list[box_num][3]
                            storage_strip.append([y,strip_list[box_num][3] * strip_list[box_num][9]])

                            while num_strips > 0 and curr_weight+box_weight < max_weight:
                                ax.bar3d(x, y, z, box_width, box_length, box_height, color=color, edgecolor='black')
                                z += box_height
                                num_strips -= 1
                                curr_weight+=strip_list[box_num][9]
                                vol_occ+=box_length*box_width*box_height
                

                            # storage_strip.append([x,y,z,box_length,box_width,box_height,box_num])
                            x += box_width
                            z = 0
                            total_strips -= 1

                        

                    else: 
                        # x = end_x
                        y_min = min(y_min,y)
                        if x+box_width> width_container:
                            # print("Inside1",end_x-x)
                            # print("vol_wasted",abs(width_container-x)*box_length*height_container)

                            vol_wasted += abs(width_container-x)*box_length*height_container

                            x = width_container
                        

                        prev_row[len(prev_row)-1].append(x)
                        prev_row[len(prev_row)-1].append(y)
                        prev_row[len(prev_row)-1].append(box_length)
                        prev_row[len(prev_row)-1].append(1)
                        prev_row[len(prev_row)-1].append(row)
                        
                        

                        # print("Y",y)
                        # print("prev_row_before",prev_row)
                        # print("prev_y",prev_y)
                        # print("x",x)
                        # print("Row",row)
                        index=0
                        if x + box_width > width_container:
                            # efficiency_new.append([y_min,row])
                            row+=1
                            x = 0
                            z = 0
                        while index<len(prev_row) and (prev_row[index][6]!=row-1):
                            index+=1

                        # print("INdex",index)
                        # print("prev_row_num",prev_row_num)
                        rem=0
                        rem_y=0
                        went_in_1 = False
                        went_in_2 =False
                        if x!=0 and end_x-x< (x+box_width)-end_x: 
                            ##Checks the better one between putting one extra or shifting according to the previous row
                            # print("Inside2",end_x-x)
                            # vol_wasted += (end_x-x)*abs(prev_row[len(prev_row)-1][4]-box_length)*height_container
                            rem=deepcopy(abs(x-end_x))
                            if len(prev_row) >1 and index<len(prev_row) and prev_row[index][6] == prev_row_num and prev_row[index][3] > prev_y:
                                went_in_1=True
                            x +=(end_x-x)
                        else:
                            if len(prev_row) >1 and index<len(prev_row) and prev_row[index][6] == prev_row_num and prev_row[index][3] > prev_y and x + box_width <= width_container:
                                went_in_2=True
                                # print("Inside3",end_x-x)
                                # vol_wasted += (x+box_width-end_x)*abs(prev_row[len(prev_row)-1][4]-box_length)*height_container
                                num_strips = strip_list[box_num][3]
                                # print("UES")
                                storage_strip.append([y,strip_list[box_num][3] * strip_list[box_num][9]])

                                while num_strips > 0 and curr_weight+box_weight < max_weight:
                                    ax.bar3d(x, y, z, box_width, box_length, box_height, color=color, edgecolor='black')
                                    z += box_height
                                    num_strips -= 1
                                    curr_weight+=strip_list[box_num][9]
                                    vol_occ+=box_length*box_width*box_height
                
                                

                                # storage_strip.append([x,y,z,box_length,box_width,box_height,box_num])
                                x += box_width
                                z = 0
                                total_strips -= 1
                                prev_row[len(prev_row)-1][2]=deepcopy(x)
                                rem=deepcopy(abs(x-end_x))
                                # storage_strip[len(storage_strip)-1][2]=deepcopy(x)


                        p_y=0
                        if went_in_1 is True:
                            p_y = deepcopy(y+box_length)
                        elif went_in_1 is False and went_in_2 is False:
                            p_y = deepcopy(box_length)
                        else:
                            p_y = deepcopy(y+box_length)




                        y,end_x,row,prev_row,prev_y,prev_row_num= (findoptlen(prev_row,x,y,end_x,box_width,row,prev_y,prev_row_num))
                        # print("y_diff",abs(p_y-y))
                        # print("rem",rem)

                        if x!=0 and went_in_1 is True:
                            # print("went_1_true:",y-box_length-p_y)
                            vol_wasted += abs(y-box_length-p_y)*rem*height_container
                        elif x!=0 and went_in_1 is False and went_in_2 is False:
                            # print("went_1_false and went_2_false",p_y)
                            vol_wasted += abs(p_y)*rem*height_container
                        else:
                            if x!=0:
                                # print("all others",y-p_y)
                                vol_wasted += abs(y-p_y)*rem*height_container

                        y_min = min(y_min,y)

                
                        if end_x+box_width>=width_container:  
                            end_x = deepcopy(width_container)
                        
                        # change = invertOrNot(x,end_x,box_num,box_length,box_width,box_height,width_container,total_strips)
                

                        # if change == True:
                        #     temp = deepcopy(box_length)
                        #     box_length = deepcopy(box_width)
                        #     box_width = deepcopy(temp)


                        y = y -1-box_length
                        y_min = min(y_min,y)

                        prev_row.append([x,y])
                        # storage_strip.append([x,y])

            

        prev_row[len(prev_row)-1].append(x)
        prev_row[len(prev_row)-1].append(y)
        prev_row[len(prev_row)-1].append(box_length)
        prev_row[len(prev_row)-1].append(1)
        prev_row[len(prev_row)-1].append(row)
        y_min = min(y_min,y)
        df.at[box_num, 'Rem_Strips'] =total_strips
        df.at[box_num,'Marked'] = 0
        # stored_plac.append([init_x,init_y,init_z,x,y,z,box_num,box_length,box_width,box_height,row])
        return x, y, z,row,prev_y,prev_row,end_x,prev_row_num,vol_occ,y_min,df,vol_wasted,storage_strip
    
    def place_nonH(x,y,z,colors,nH_list,container,ax,curr_weight,stored_plac,vol_occ,y_min,df,vol_wasted): # Used for placing the non homogenous boxes
        width_container = float(container.width)
        height_container = float(container.height)
        depth_container = float(container.length)
        max_weight = float(container.max_weight)


        init_x=x
        init_y=y
        init_z=z

        total_boxes= 0
        max_len = 0
        max_width= 0

        for num in nH_list:
            total_boxes+= num[3]
            max_len = max(max_len,num[0])
            max_width = max(max_width,num[1])

        index = 0
        while total_boxes>0 and y > 0:

            while z<height_container and index < len(nH_list):
                rem_boxes = nH_list[index][3]
                box_num = nH_list[index][4]
                box_length = nH_list[index][0]
                box_width = nH_list[index][1]
                box_height = nH_list[index][2]

                temp = rem_boxes
                dw = False
                while temp>0 and z<height_container:
                    if(x+box_width < width_container and curr_weight < max_weight and z+box_height < height_container):
                        ax.bar3d(x, y, z, box_width, box_length, box_height, color=colors[box_num], edgecolor='black')
                        z += box_height
                        temp -= 1
                        curr_weight+=nH_list[index][5]
                    else:
                        dw = True
                        break
                nH_list[index][3] -= rem_boxes-temp
                total_boxes-=(rem_boxes-temp)
                if(dw==False and nH_list[index][3]==0):
                    index+=1
                if dw == True and nH_list[index][3]!=0:
                    x = x+max_width
                    z=0


                if(x+max_width > width_container):
                    x=0
                    y = y-max_len
                    z=0
                # stored_plac.append([init_x,init_y,init_z,x,y,z,box_num])

            if z >height_container:
                if(x+max_width> width_container):
                    x=0
                    y = y-max_len
                    z=0
                else:
                    x= x+ max_width
                    z = 0

        # plt.show()

        return  x,y,z,vol_occ,y_min,df,vol_wasted,storage_strip


    def stability(weight_leftHalf, weight_rightHalf,best_widht_order,curr_width_order,vol_wasted,vol_occupied):
        # Finding the percentage difference betweeenn the load on the front and rear axel from the centre line 
        front_axel_perc = weight_leftHalf/(weight_rightHalf+weight_leftHalf+0.01)
        penalty_weight = 0
        penalty_width_order = 0
        if front_axel_perc>0.6 :
            penalty_weight = abs(front_axel_perc-0.6)*10
        elif front_axel_perc<0.5:
            penalty_weight = abs(front_axel_perc-0.5)*10

        for i in range(len(curr_width_order)):
            if best_widht_order[i]==curr_width_order[i]:
                penalty_width_order+=1
            else:
                penalty_width_order-=1

        # print(best_widht_order)
        # print(curr_width_order)
        # print("Vol_was", 0.8*vol_wasted)
        # print("Vol_occ",0.1*vol_occupied)
        # print("width_order", 0.05*penalty_width_order)
        # print("weight_pen", 0.05*penalty_weight)

        
        stab = round(0.8*vol_wasted-( 0.1*vol_occupied/100)-0.05*penalty_width_order-0.05*penalty_weight,2)

        return stab
 

    def create_plot(container):
        width_container = float(container.width)
        height_container = float(container.height)
        length_container = float(container.length)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(0, width_container)
        ax.set_ylim(0, length_container)
        ax.set_zlim(0, height_container)

        # Set labels for axes
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_box_aspect([1, length_container/width_container, height_container/width_container])


        # Dont show the plot
        plt.ioff()
        return ax

    def weight_distribution(container_toFit,storage_strip):
        weight_sum_lower_half = 0
        weight_sum_upper_half = 0

        # Calculate the threshold for comparison
        threshold = container_toFit.length / 2

        # Iterate through the storage_strip list and sum weights based on the condition
        for item in storage_strip:
            y, weight = item
            if y < threshold:
                weight_sum_lower_half += weight
            else:
                weight_sum_upper_half += weight

        # Print the results
        # print("Sum of weights for y < container_length / 2:", weight_sum_lower_half)
        # print("Sum of weights for y > container_length / 2:", weight_sum_upper_half)

        return weight_sum_lower_half,weight_sum_upper_half



    def create_bottom_view(ax, vol_occ, vol_wasted, key, roll, stability_fin):
    # Adjust the viewing angle for bottom view
        ax.view_init(elev=90, azim=180)
        key = key.replace("'", "")

        # Create text annotation including vol_occ and vol_wasted
        text_top = f'vol_occ: {vol_occ:.2f}%\nvol_wasted: {vol_wasted:.2f}m^3\n{key}, roll: {roll}'
        ax.text2D(0.05, 0.95, text_top, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        
        # Create text annotation for stability_fin at the bottom
        text_bottom = f'stability_fin: {stability_fin}'
        ax.text2D(0.05, 0.05, text_bottom, transform=ax.transAxes, fontsize=12,
                verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        
        # Save the plot as an image with a fixed filename
        filename = os.path.join('static', f"{key}_roll{roll}_bottom_view.png")
        plt.savefig(filename)

        # Close the plot to free up resources
        plt.close()

        return filename  # Return the filename for reference

    def widthOrder(strip_list):
        temp = []
        for details in strip_list:
            temp.append(details[1])
        temp.sort(reverse=True)

        return temp





    def generate_colors(n):
        distinct_colors = ['red', 'blue', 'yellow', 'orange', 'green', 'violet', 'white', 'indigo', 'cyan', 'magenta', 'lime', 'pink', 'teal', 'lavender', 'brown', 'gray', 'black']

        if n <= len(distinct_colors):
            return {i: distinct_colors[i] for i in range(n)}
        else:
            new_colors = {}
            for i in range(n):
                new_colors[i] = distinct_colors[i % len(distinct_colors)]
            return new_colors

    # Example usage:
    n = len(strip_list)
    # print(generate_colors(n))
    colors = generate_colors(n)
    # print(colors)

    # print(strip_list)
    stored_plac = []
    storage_strip=[]
    prev_row= []
    curr_width_order = []
    end_x = float(container_toFit.width)
    curr_weight = 0
    prev_y =-1
    y_min = 1e5
    row =0
    vol_occ = 0
    vol_wasted=0
    prev_row_num=-1
    #Creating Plot
    ax = create_plot(container_toFit)
    x,y,z= 0,0,0
    
    for i in range(len(strip_list)):
        # if i==3:
        # break
        if len(prev_row)==0 or len(prev_row)==1:
            end_x = float(container_toFit.width)
    
        ans =choose_best_dimension(x,end_x,z,strip_list,container_toFit,stored_plac)
        if ans==-1:
            break
        curr_width_order.append(strip_list[ans][1])
        strip_list[ans][7] = False
        if i == 0:
            y=container_toFit.length-strip_list[ans][0]-1
            y_min = min(y_min,y)
        if y<=0 or curr_weight+strip_list[ans][9]>container_toFit.max_weight:
            break
        if(i!=0 and row!=0):
            y = y-1+prev_row[len(prev_row)-1][4]-strip_list[ans][0]
            y_min = min(y_min,y)
            prev_row.append([x,y])
            x,y,z,row,prev_y,prev_row,end_x,prev_row_num,vol_occ,y_min,df,vol_wasted,storage_strip= after_plac(x,y,z,end_x,ans,strip_list,container_toFit,ax,colors[ans],curr_weight,stored_plac,row,storage_strip,prev_y,prev_row,prev_row_num,vol_occ,y_min,df,vol_wasted)
        else:
            if(i!=0 and row==0):
                y = y-1+prev_row[len(prev_row)-1][4]-strip_list[ans][0]
            prev_row.append([x,y])
            x,y,z,row,prev_y,prev_row,end_x,prev_row_num,vol_occ,y_min,df,vol_wasted,storage_strip= after_plac(x,y,z,end_x,ans,strip_list,container_toFit,ax,colors[ans],curr_weight,stored_plac,row,storage_strip,prev_y,prev_row,prev_row_num,vol_occ,y_min,df,vol_wasted)
    

    #Non homo placement
    nH_list =  []

    for i in range(len(df)):
        nH_list.append([df.at[i,'Length'],df.at[i,'Width'],df.at[i,'Height'],df.at[i,'Rem_Boxes'],i,df.at[i,'GrossWeight']])
    x,y,z,vol_occ,y_min,df,vol_wasted,storage_strip = place_nonH(x,y,z,colors,nH_list,container_toFit,ax,curr_weight,stored_plac,vol_occ,y_min,df,vol_wasted)


    if y_min <0:
        y_min = 0
    vol_occ_curr =round(vol_occ/(container_toFit.length*container_toFit.width*container_toFit.height),2)*100
    vol_wasted=round(vol_wasted*pow(10,-9),2)
    


    # print(storage_strip)
    weight_leftHalf, weight_rightHalf = weight_distribution(container_toFit,storage_strip)
    best_width_order = widthOrder(strip_list)
    stability_fin = stability(weight_leftHalf,weight_rightHalf,best_width_order,curr_width_order,vol_wasted,vol_occ_curr)
    filename_final = create_bottom_view(ax,vol_occ_curr,vol_wasted,key,roll,stability_fin)
    # plt.close('all')
    # print(df)
    
    return filename_final,df


def worker(df, container_toFit,strip_list,keys,roll):
    length_container = container_toFit.length
    width_container = container_toFit.width
    height_container = container_toFit.height
    
    def widthRem(width_rem,strip_list):
        for box in strip_list:

            if box[1] <= width_rem:
                return True
        
        return False


    def choose_best_dimension(x,end_x,z,strip_list,container,stored_plac):

        width_container = float(container.width)
        height_container = float(container.height)
        depth_container = float(container.length)

        if end_x ==0:
            end_x = width_container

        width_rem = end_x-x
        box_num = -1
        checker = widthRem(width_rem,strip_list)
        if checker == False:
            width_rem = width_container

        mini = -1e9
        index = 0
        best_width = []
        for box_dim in strip_list:
            if(box_dim[7]==0):
                index+=1
                continue
            length_diff = 1e4
            num_box = width_rem//box_dim[1]
            total_num_strips = box_dim[4]
            height = box_dim[2]
            fill = True
            if num_box > total_num_strips:
                fill= False
            perc = (num_box*box_dim[1]/width_rem)
            if len(stored_plac)>0:
                prev_length = stored_plac[len(stored_plac)-1][7]
                length_diff = abs(box_dim[0]-prev_length)
                # print("TES")
            best_width.append([index,perc,box_dim[1],fill,length_diff,((height_container//height)*height)])
            index+=1
        # print(strip_list)
        sorted_data = sorted(best_width, key=lambda x: (x[3], x[1], x[5],x[2]), reverse=True)
    
        n =1    #Currently taking the best only based on the efficiency.
        ind =0
        maxi =1e5
        best_box =-1
        while ind < len(sorted_data) and ind <= n :
            if maxi > sorted_data[ind][4] and sorted_data[ind][3]:
                maxi = min(maxi,sorted_data[ind][4])
                # print("maxi",maxi)
                best_box = sorted_data[ind][0]
            ind+=1
        if best_box==-1:
            best_box = sorted_data[0][0]

        return best_box

        # return ans

    def invertOrNot(x,end_x,box_num,box_length,box_width,box_height,width_container,total_strips):
        rem_width= end_x-x
        if rem_width <=0:
            return False
        num_strips_required = rem_width//box_width
        if num_strips_required > total_strips:
            return False
        perc_nonInverted = ((rem_width//box_width)*box_width)/rem_width
        perc_Inverted = ((rem_width//box_length)*box_length)/rem_width


        if perc_Inverted > perc_nonInverted:
            return True

        else:
            return False



    def findoptlen(prev_row,x,y,end_x,box_width,row,prev_y,prev_row_num):
        prev_row = [prev_row for prev_row in prev_row if ((prev_row[0] != prev_row[2])) ]
        # print("Row_inside_optlen",row)
        # print(prev_row)
        # print("X_inside_optlen",x)
        i = 0
        while i<len(prev_row) and(prev_row[i][6]>row or prev_row[i][6]<row-1):
            i+=1
        
        # print("inside_opt_len_iterator",i)
        
        if end_x+box_width> float(container_toFit.width) and len(prev_row)<=1:
            end_x= container_toFit.width
        else:
            end_x = deepcopy(prev_row[i][2])
        
        # print("endX_inside_optlen",end_x)

            
        # print("PREV_ROW",prev_row)
        # if i < len(prev_row):
        y = prev_row[i][3]
        prev_row_num=deepcopy(prev_row[i][6])
        prev_row[i][5] = 0
        prev_y = deepcopy(prev_row[i][1])
        del prev_row[i]
        return y, end_x,row,prev_row,prev_y,prev_row_num
            
            



    def after_plac(x,y,z,end_x,box_num,strip_list,container,ax,color,curr_weight,stored_plac,row,strip_storage,prev_y,prev_row,prev_row_num,vol_occ,y_min,df,vol_wasted):
        width_container = float(container.width)
        height_container = float(container.height)
        depth_container = float(container.length)
        max_weight = container.max_weight


        init_x=x
        init_y=y
        init_z=z

        total_strips = deepcopy(df.at[box_num,'Rem_Strips'])
        box_length = deepcopy(float(strip_list[box_num][0]))
        box_width = deepcopy(float(strip_list[box_num][1]))
        box_height = deepcopy(float(strip_list[box_num][2]))
        box_weight = deepcopy(float(strip_list[box_num][9]))

        change_init = invertOrNot(x,end_x,box_num,box_length,box_width,box_height,width_container,total_strips)
        if change_init == True:
            
            y = y+box_length
            temp = deepcopy(box_length)
            box_length = deepcopy(box_width)
            box_width = deepcopy(temp)
            prev_row[len(prev_row)-1][1] = prev_row[len(prev_row)-1][1] + box_width- box_length
            # storage_strip[len(storage_strip)-1][1] = storage_strip[len(storage_strip)-1][1] + box_width- box_length
            y = y-box_length



        while total_strips > 0 and y > 0 and curr_weight+box_weight<max_weight:
            if x + box_width <= end_x and x + box_width <= width_container:  ## added the max weight check constraints
                # print("prev_row",prev_row)
                if len(prev_row) >1 and prev_row[len(prev_row)-2][1]>=0 and y-prev_row[len(prev_row)-2][1] > box_length and row== prev_row[len(prev_row)-2][6]:
                    
                    num_strips = strip_list[box_num][3]
                    # print("UES")
                    storage_strip.append([y,strip_list[box_num][3] * strip_list[box_num][9]])
                    while num_strips > 0 and curr_weight+box_weight < max_weight:
                        ax.bar3d(x, y, z, box_width, box_length, box_height, color=color, edgecolor='black')
                        z += box_height
                        num_strips -= 1
                        curr_weight+=strip_list[box_num][9]
                        vol_occ+=box_length*box_width*box_height

                    
                    z = 0
                    total_strips -= 1
                    # prev_row[len(prev_row)-1][2]=deepcopy(x)
                    if total_strips==0:
                        x+=box_width
                        continue
                    if total_strips>0:
                        if y-box_length>=0:
                            y-=box_length
                        else:
                            continue
                        prev_row[len(prev_row)-1][1]=deepcopy(y)
                        # storage_strip[len(storage_strip)-1][1]=deepcopy(y)
                        

                        num_strips = strip_list[box_num][3]
                        # print("UES")
                        storage_strip.append([y,strip_list[box_num][3] * strip_list[box_num][9]])

                        while num_strips > 0 and curr_weight+box_weight < max_weight:
                            ax.bar3d(x, y, z, box_width, box_length, box_height, color=color, edgecolor='black')
                            z += box_height
                            num_strips -= 1
                            curr_weight+=strip_list[box_num][9]
                            vol_occ+=box_length*box_width*box_height
                        x+=box_width

                        if x+box_width<=end_x:
                            y+=box_length
                        z=0
                        

                        total_strips-=1

                    
                    


                else:
                    num_strips = strip_list[box_num][3]
                    storage_strip.append([y,strip_list[box_num][3] * strip_list[box_num][9]])

                    while num_strips > 0 and curr_weight+box_weight < max_weight:
                        ax.bar3d(x, y, z, box_width, box_length, box_height, color=color, edgecolor='black')
                        z += box_height
                        num_strips -= 1
                        curr_weight+=strip_list[box_num][9]
                        vol_occ+=box_length*box_width*box_height

                    
                    x += box_width
                    z = 0
                    total_strips -= 1

                

            else: 
                
                y_min = min(y_min,y)
                if x+box_width> width_container:
                    

                    vol_wasted += abs(width_container-x)*box_length*height_container

                    x = width_container
                

                prev_row[len(prev_row)-1].append(x)
                prev_row[len(prev_row)-1].append(y)
                prev_row[len(prev_row)-1].append(box_length)
                prev_row[len(prev_row)-1].append(1)
                prev_row[len(prev_row)-1].append(row)
                
                

                
                index=0
                if x + box_width > width_container:
                    # efficiency_new.append([y_min,row])
                    row+=1
                    x = 0
                    z = 0
                while index<len(prev_row) and (prev_row[index][6]!=row-1):
                    index+=1

                
                rem=0
                rem_y=0
                went_in_1 = False
                went_in_2 =False
                if x!=0 and end_x-x< (x+box_width)-end_x: 
                    rem=deepcopy(abs(x-end_x))
                    if len(prev_row) >1 and index<len(prev_row) and prev_row[index][6] == prev_row_num and prev_row[index][3] > prev_y:
                        went_in_1=True
                    x +=(end_x-x)
                else:
                    if len(prev_row) >1 and index<len(prev_row) and prev_row[index][6] == prev_row_num and prev_row[index][3] > prev_y and x + box_width <= width_container:
                        went_in_2=True
                        
                        num_strips = strip_list[box_num][3]
                        storage_strip.append([y,strip_list[box_num][3] * strip_list[box_num][9]])

                        while num_strips > 0 and curr_weight+box_weight < max_weight:
                            ax.bar3d(x, y, z, box_width, box_length, box_height, color=color, edgecolor='black')
                            z += box_height
                            num_strips -= 1
                            curr_weight+=strip_list[box_num][9]
                            vol_occ+=box_length*box_width*box_height

                        
                        x += box_width
                        z = 0
                        total_strips -= 1
                        prev_row[len(prev_row)-1][2]=deepcopy(x)
                        rem=deepcopy(abs(x-end_x))
                        


                p_y=0
                if went_in_1 is True:
                    p_y = deepcopy(y+box_length)
                elif went_in_1 is False and went_in_2 is False:
                    p_y = deepcopy(box_length)
                else:
                    p_y = deepcopy(y+box_length)




                y,end_x,row,prev_row,prev_y,prev_row_num= (findoptlen(prev_row,x,y,end_x,box_width,row,prev_y,prev_row_num))
                

                if x!=0 and went_in_1 is True:
                    # print("went_1_true:",y-box_length-p_y)
                    vol_wasted += abs(y-box_length-p_y)*rem*height_container
                elif x!=0 and went_in_1 is False and went_in_2 is False:
                    vol_wasted += abs(p_y)*rem*height_container
                else:
                    if x!=0:
                        vol_wasted += abs(y-p_y)*rem*height_container

                y_min = min(y_min,y)

         
                if end_x+box_width>=width_container:  
                    end_x = deepcopy(width_container)
                
                change = invertOrNot(x,end_x,box_num,box_length,box_width,box_height,width_container,total_strips)
        

                if change == True:
                    temp = deepcopy(box_length)
                    box_length = deepcopy(box_width)
                    box_width = deepcopy(temp)


                y = y -1-box_length
                y_min = min(y_min,y)

                prev_row.append([x,y])


        if y<0 and total_strips!=0 and curr_weight+box_weight<max_weight:
            y+=box_length
            y-=box_width
            if y>0:
                temp = deepcopy(box_length)
                box_length = deepcopy(box_width)
                box_width = deepcopy(temp)
                while total_strips > 0 and y > 0 and curr_weight+box_weight<max_weight:
                    if x + box_width <= end_x and x + box_width <= width_container:  ## added the max weight check constraints
                        # print("prev_row",prev_row)
                        if len(prev_row) >1 and prev_row[len(prev_row)-2][1]>=0 and y-prev_row[len(prev_row)-2][1] > box_length and row== prev_row[len(prev_row)-2][6]:
                            num_strips = strip_list[box_num][3]
                            # print("UES")
                            storage_strip.append([y,strip_list[box_num][3] * strip_list[box_num][9]])

                            while num_strips > 0 and curr_weight+box_weight < max_weight:
                                ax.bar3d(x, y, z, box_width, box_length, box_height, color=color, edgecolor='black')
                                z += box_height
                                num_strips -= 1
                                curr_weight+=strip_list[box_num][9]
                                vol_occ+=box_length*box_width*box_height

                            # storage_strip.append([x,y,z,box_length,box_width,box_height,box_num])
                            # x += box_width.
                            z = 0
                            total_strips -= 1
                            # prev_row[len(prev_row)-1][2]=deepcopy(x)
                            if total_strips==0:
                                x+=box_width
                                continue
                            if total_strips>0:
                                if y-box_length>=0:
                                    y-=box_length
                                else:
                                    continue
                                prev_row[len(prev_row)-1][1]=deepcopy(y)
                                # storage_strip[len(storage_strip)-1][1]=deepcopy(y)
                                

                                num_strips = strip_list[box_num][3]
                                # print("UES")
                                storage_strip.append([y,strip_list[box_num][3] * strip_list[box_num][9]])

                                while num_strips > 0 and curr_weight+box_weight < max_weight:
                                    ax.bar3d(x, y, z, box_width, box_length, box_height, color=color, edgecolor='black')
                                    z += box_height
                                    num_strips -= 1
                                    curr_weight+=strip_list[box_num][9]
                                    vol_occ+=box_length*box_width*box_height
                                x+=box_width
                                # print("Yes goes in",end_x)
                                # print("Y_before",y)

                                if x+box_width<=end_x:
                                    y+=box_length
                                z=0
                                # print("Y_after",y)

                                total_strips-=1

                            
                            # if total_strips==0:


                        else:
                            num_strips = strip_list[box_num][3]
                            storage_strip.append([y,strip_list[box_num][3] * strip_list[box_num][9]])

                            while num_strips > 0 and curr_weight+box_weight < max_weight:
                                ax.bar3d(x, y, z, box_width, box_length, box_height, color=color, edgecolor='black')
                                z += box_height
                                num_strips -= 1
                                curr_weight+=strip_list[box_num][9]
                                vol_occ+=box_length*box_width*box_height

                            # storage_strip.append([x,y,z,box_length,box_width,box_height,box_num])
                            x += box_width
                            z = 0
                            total_strips -= 1

                        

                    else: 
                        # x = end_x
                        y_min = min(y_min,y)
                        if x+box_width> width_container:
                            # print("Inside1",end_x-x)
                            # print("vol_wasted",abs(width_container-x)*box_length*height_container)

                            vol_wasted += abs(width_container-x)*box_length*height_container

                            x = width_container
                        

                        prev_row[len(prev_row)-1].append(x)
                        prev_row[len(prev_row)-1].append(y)
                        prev_row[len(prev_row)-1].append(box_length)
                        prev_row[len(prev_row)-1].append(1)
                        prev_row[len(prev_row)-1].append(row)
                        
                        

                        # print("Y",y)
                        # print("prev_row_before",prev_row)
                        # print("prev_y",prev_y)
                        # print("x",x)
                        # print("Row",row)
                        index=0
                        if x + box_width > width_container:
                            # efficiency_new.append([y_min,row])
                            row+=1
                            x = 0
                            z = 0
                        while index<len(prev_row) and (prev_row[index][6]!=row-1):
                            index+=1

                        # print("INdex",index)
                        # print("prev_row_num",prev_row_num)
                        rem=0
                        rem_y=0
                        went_in_1 = False
                        went_in_2 =False
                        if x!=0 and end_x-x< (x+box_width)-end_x: 
                            ##Checks the better one between putting one extra or shifting according to the previous row
                            # print("Inside2",end_x-x)
                            # vol_wasted += (end_x-x)*abs(prev_row[len(prev_row)-1][4]-box_length)*height_container
                            rem=deepcopy(abs(x-end_x))
                            if len(prev_row) >1 and index<len(prev_row) and prev_row[index][6] == prev_row_num and prev_row[index][3] > prev_y:
                                went_in_1=True
                            x +=(end_x-x)
                        else:
                            if len(prev_row) >1 and index<len(prev_row) and prev_row[index][6] == prev_row_num and prev_row[index][3] > prev_y and x + box_width <= width_container:
                                went_in_2=True
                                # print("Inside3",end_x-x)
                                # vol_wasted += (x+box_width-end_x)*abs(prev_row[len(prev_row)-1][4]-box_length)*height_container
                                num_strips = strip_list[box_num][3]
                                # print("UES")
                                storage_strip.append([y,strip_list[box_num][3] * strip_list[box_num][9]])

                                while num_strips > 0 and curr_weight+box_weight < max_weight:
                                    ax.bar3d(x, y, z, box_width, box_length, box_height, color=color, edgecolor='black')
                                    z += box_height
                                    num_strips -= 1
                                    curr_weight+=strip_list[box_num][9]
                                    vol_occ+=box_length*box_width*box_height

                                # storage_strip.append([x,y,z,box_length,box_width,box_height,box_num])
                                x += box_width
                                z = 0
                                total_strips -= 1
                                prev_row[len(prev_row)-1][2]=deepcopy(x)
                                rem=deepcopy(abs(x-end_x))
                                # storage_strip[len(storage_strip)-1][2]=deepcopy(x)
                            else:
                                rem=deepcopy(abs(x-end_x))



                        p_y=0
                        if went_in_1 is True:
                            p_y = deepcopy(y+box_length)
                        elif went_in_1 is False and went_in_2 is False:
                            p_y = deepcopy(box_length)
                        else:
                            p_y = deepcopy(y+box_length)




                        y,end_x,row,prev_row,prev_y,prev_row_num= (findoptlen(prev_row,x,y,end_x,box_width,row,prev_y,prev_row_num))
                        # print("y_diff",abs(p_y-y))
                        # print("rem",rem)

                        if x!=0 and went_in_1 is True:
                            # print("went_1_true:",y-box_length-p_y)
                            vol_wasted += abs(y-box_length-p_y)*rem*height_container
                        elif x!=0 and went_in_1 is False and went_in_2 is False:
                            # print("went_1_false and went_2_false",p_y)
                            vol_wasted += abs(p_y)*rem*height_container
                        else:
                            if x!=0:
                                # print("all others",y-p_y)
                                vol_wasted += abs(y-p_y)*rem*height_container

                        y_min = min(y_min,y)

                
                        if end_x+box_width>=width_container:  
                            end_x = deepcopy(width_container)
                        
                        # change = invertOrNot(x,end_x,box_num,box_length,box_width,box_height,width_container,total_strips)
                

                        # if change == True:
                        #     temp = deepcopy(box_length)
                        #     box_length = deepcopy(box_width)
                        #     box_width = deepcopy(temp)


                        y = y -1-box_length
                        y_min = min(y_min,y)

                        prev_row.append([x,y])
                        # storage_strip.append([x,y])

            

        prev_row[len(prev_row)-1].append(x)
        prev_row[len(prev_row)-1].append(y)
        prev_row[len(prev_row)-1].append(box_length)
        prev_row[len(prev_row)-1].append(1)
        prev_row[len(prev_row)-1].append(row)
        y_min = min(y_min,y)
        df.at[box_num, 'Rem_Strips'] =total_strips
        df.at[box_num,'Marked'] = 0
        
        # stored_plac.append([init_x,init_y,init_z,x,y,z,box_num,box_length,box_width,box_height,row])
        return x, y, z,row,prev_y,prev_row,end_x,prev_row_num,vol_occ,y_min,df,total_strips,vol_wasted,storage_strip
    
    def place_nonH(x,y,z,colors,nH_list,container,ax,curr_weight,stored_plac,vol_occ,y_min,df,vol_wasted): # Used for placing the non homogenous boxes
        width_container = float(container.width)
        height_container = float(container.height)
        depth_container = float(container.length)
        max_weight = float(container.max_weight)


        init_x=x
        init_y=y
        init_z=z

        total_boxes= 0
        max_len = 0
        max_width= 0

        for num in nH_list:
            total_boxes+= num[3]
            max_len = max(max_len,num[0])
            max_width = max(max_width,num[1])

        index = 0
        while total_boxes>0 and y > 0:

            while z<height_container and index < len(nH_list):
                rem_boxes = nH_list[index][3]
                box_num = nH_list[index][4]
                box_length = nH_list[index][0]
                box_width = nH_list[index][1]
                box_height = nH_list[index][2]

                temp = rem_boxes
                dw = False
                while temp>0 and z<height_container:
                    if(x+box_width < width_container and curr_weight < max_weight and z+box_height < height_container):
                        ax.bar3d(x, y, z, box_width, box_length, box_height, color=colors[box_num], edgecolor='black')
                        z += box_height
                        temp -= 1
                        curr_weight+=nH_list[index][5]
                    else:
                        dw = True
                        break
                nH_list[index][3] -= rem_boxes-temp
                total_boxes-=(rem_boxes-temp)
                if(dw==False and nH_list[index][3]==0):
                    index+=1
                if dw == True and nH_list[index][3]!=0:
                    x = x+max_width
                    z=0


                if(x+max_width > width_container):
                    x=0
                    y = y-max_len
                    z=0
                # stored_plac.append([init_x,init_y,init_z,x,y,z,box_num])

            if z >height_container:
                if(x+max_width> width_container):
                    x=0
                    y = y-max_len
                    z=0
                else:
                    x= x+ max_width
                    z = 0

        # plt.show()

        return  x,y,z,vol_occ,y_min,df,vol_wasted,storage_strip

    def stability(weight_leftHalf, weight_rightHalf,best_widht_order,curr_width_order,vol_wasted,vol_occupied):
    # Finding the percentage difference betweeenn the load on the front and rear axel from the centre line 
        front_axel_perc = weight_leftHalf/(weight_rightHalf+weight_leftHalf)
        penalty_weight = 0
        penalty_width_order = 0
        if front_axel_perc>0.6 :
            penalty_weight = abs(front_axel_perc-0.6)*10
        elif front_axel_perc<0.5:
            penalty_weight = abs(front_axel_perc-0.5)*10

        for i in range(len(curr_width_order)):
            if best_widht_order[i]==curr_width_order[i]:
                penalty_width_order+=1
            else:
                penalty_width_order-=1

        # print(best_widht_order)
        # print(curr_width_order)
        # print("Vol_was", 0.8*vol_wasted)
        # print("Vol_occ",0.1*vol_occupied)
        # print("width_order", 0.05*penalty_width_order)
        # print("weight_pen", 0.05*penalty_weight)

        
        stab = round(0.8*vol_wasted-( 0.1*vol_occupied/100)-0.05*penalty_width_order-0.05*penalty_weight,2)

        return stab


    def create_bottom_view(ax, iteration, vol_occ_curr, vol_wasted, keys, roll,stability_fin):
    # Adjust the viewing angle for bottom view
        ax.view_init(elev=90, azim=180)
        keys=keys.replace("'","")
        
        # Create text annotation including iteration number, vol_occ_curr, and vol_wasted
        text = f'Iteration: {iteration}\nvol_occ_curr: {vol_occ_curr:.2f}%\nvol_wasted: {vol_wasted:.2f}m^3\nKeys: {keys}\nRoll: {roll}'
        
        ax.text2D(0.05, 0.95, text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        
        # Construct the filename including the iteration number, keys, and roll
        # filename = f'bottom_view_iteration_{iteration}_keys_{keys}_roll_{roll}.png'
        text_bottom = f'stability_fin: {stability_fin}'
        ax.text2D(0.05, 0.05, text_bottom, transform=ax.transAxes, fontsize=12,
                verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        
        filename = os.path.join('static', f"bottom_view_iteration_{iteration}_keys_{keys}_roll_{roll}.png")
        plt.savefig(filename)
        
        # Close the plot to free up resources
        plt.close()

        return filename
    # Return the filename for reference
    def weight_distribution(container_toFit,storage_strip):
        weight_sum_lower_half = 0
        weight_sum_upper_half = 0

        # Calculate the threshold for comparison
        threshold = container_toFit.length / 2

        # Iterate through the storage_strip list and sum weights based on the condition
        for item in storage_strip:
            y, weight = item
            if y < threshold:
                weight_sum_lower_half += weight
            else:
                weight_sum_upper_half += weight

        # Print the results
        # print("Sum of weights for y < container_length / 2:", weight_sum_lower_half)
        # print("Sum of weights for y > container_length / 2:", weight_sum_upper_half)

        return weight_sum_lower_half,weight_sum_upper_half


    def generate_colors(n):
        distinct_colors = ['red', 'blue', 'yellow', 'orange', 'green', 'violet', 'white', 'indigo', 'cyan', 'magenta', 'lime', 'pink', 'teal', 'lavender', 'brown', 'gray', 'black']

        if n <= len(distinct_colors):
            return {i: distinct_colors[i] for i in range(n)}
        else:
            new_colors = {}
            for i in range(n):
                new_colors[i] = distinct_colors[i % len(distinct_colors)]
            return new_colors

    # Example usage:
    n = len(strip_list)
    # print(generate_colors(n))
    colors = generate_colors(n)

    
    def generate_permutations(n):
        numbers = list(range(n))
        perms = list(permutations(numbers))
        return perms
    
    def widthOrder(strip_list):
        temp = []
        for details in strip_list:
            temp.append(details[1])
        temp.sort(reverse=True)

        return temp

    n = len(strip_list)
    perms = generate_permutations(n)
    perms = list(perms)

    weight_storer = []
    df_stored=[]
    all_y_min=[]
    file_storer =[]
    vol_eff =[]
    wasted_vol = []
    best_width_order = widthOrder(strip_list)
    min_stab= 1e5
    max_index3=-1

    for i in range(len(perms)):
        # if i !=18:
        print(perms[i])
        curr_order =[]

        #     continue
        # print("iteration: ",i)
        # print("Perm",perms[i])
        # if len(perms)!=0:
        #     if y <0:
        #         y_min = min(y,y_min)
        #         break
        tmp= deepcopy(df)
        vol_wasted=0
        stored_plac = []
        storage_strip=[]
        vol_occ = 0
        y_min = 1e5
        prev_row= []
        end_x = float(container_toFit.width)
        curr_weight = 0
        prev_y =-1
        row =0
        prev_row_num=-1
        #Creating Plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(0, float(container_toFit.width))
        ax.set_ylim(0, float(container_toFit.length))
        ax.set_zlim(0, float(container_toFit.height))

        # Set labels for axes
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_box_aspect([1, length_container/width_container, height_container/width_container])

        x,z= 0,0
        # print("Perm",perms[i])
        for j in range(len(perms[i])):
            curr_order.append(strip_list[perms[i][j]][1])


            # print("Memory usage at {i}:", memory_usage(), "MB")

            # print(perms[i][j])
            # print(prev_row)
            # if j==4:
            #     break

            if len(prev_row)==0 or len(prev_row)==1:
                end_x = float(container_toFit.width)
        
            # ans =choose_best_dimension(x,end_x,z,strip_list,container_toFit,stored_plac)
            # strip_list[ans][7] = False
            if j == 0:
                y=deepcopy(length_container-strip_list[perms[i][j]][0]-1)
                y_min = min(y_min,y)
            
            if y<0:
                break

            
            if(j!=0 and row!=0):
                y = y-1+prev_row[len(prev_row)-1][4]-strip_list[perms[i][j]][0]
                y_min = min(y_min,y)
                prev_row.append([x,y])
                # storage_strip.append([x,y])
                x,y,z,row,prev_y,prev_row,end_x,prev_row_num,vol_occ,y_min,tmp,total_strips,vol_wasted,storage_strip= after_plac(x,y,z,end_x,perms[i][j],strip_list,container_toFit,ax,colors[perms[i][j]],curr_weight,stored_plac,row,storage_strip,prev_y,prev_row,prev_row_num,vol_occ,y_min,tmp,vol_wasted)
            else:
                if(j!=0 and row==0):
                    y = y-1+prev_row[len(prev_row)-1][4]-strip_list[perms[i][j]][0]
                y_min = min(y_min,y)
                prev_row.append([x,y])
                # storage_strip.append([x,y])
                x,y,z,row,prev_y,prev_row,end_x,prev_row_num,vol_occ,y_min,tmp,total_strips,vol_wasted,storage_strip= after_plac(x,y,z,end_x,perms[i][j],strip_list,container_toFit,ax,colors[perms[i][j]],curr_weight,stored_plac,row,storage_strip,prev_y,prev_row,prev_row_num,vol_occ,y_min,tmp,vol_wasted)
            
            tmp.at[perms[i][j], 'Rem_Strips'] =total_strips
            tmp.at[perms[i][j],'Marked'] = 0
        
        # for m in range(len(df)):
        #     if tmp.at[m,'Marked']==1:
        #         tmp.at[m,'Rem_Strips'] = tmp.at[m,'TotalNumStrips']
        
        # tmp.drop(columns=['Marked'], inplace=True)
        # tmp =tmp.to_html()
        df_stored.append(tmp)
        nH_list =  []

        for m in range(len(tmp)):
            nH_list.append([tmp.at[m,'Length'],tmp.at[m,'Width'],tmp.at[m,'Height'],tmp.at[m,'Rem_Boxes'],m,tmp.at[m,'GrossWeight']])
        x,y,z,vol_occ,y_min,tmp,vol_wasted,storage_strip = place_nonH(x,y,z,colors,nH_list,container_toFit,ax,curr_weight,stored_plac,vol_occ,y_min,tmp,vol_wasted)
 
        if y_min <0:
            y_min = 0        
        vol_occ_curr =round(vol_occ/(container_toFit.length*container_toFit.width*container_toFit.height),2)*100
        wasted_vol.append(round(vol_wasted*pow(10,-9),2))
        vol_eff.append(vol_occ_curr)
        weight_storer.append(storage_strip)
        weight_leftHalf, weight_rightHalf = weight_distribution(container_toFit,storage_strip)
        stability_fin = stability(weight_leftHalf,weight_rightHalf,best_width_order,curr_order,round(vol_wasted*pow(10,-9),2),round(vol_occ/(container_toFit.length*container_toFit.width*container_toFit.height),2)*100)
        # print(stability_fin)
        # print("min_stab",min_stab)
        if min_stab>stability_fin:
            min_stab = stability_fin
            max_index3 = deepcopy(i)
        # print(i)
        filename = create_bottom_view(ax, i,vol_occ_curr,round(vol_wasted*pow(10,-9),2),keys,roll,stability_fin)
        file_storer.append(filename)
        all_y_min.append(y_min)
        # print("Memory usage:", memory_usage(), "MB")
        ax.clear()
       
    


    # print(max_index3)
    # df_html = df.to_html()  
    final = df_stored[max_index3]
    filename = file_storer[max_index3]

    # print("file_name",filename)
    # print("Final",final)


    return filename,final

    

@app.route('/load_backend_function', methods=['POST'])
def load_backend_function():
    # Perform your backend function here
    # For demonstration purposes, just printing a message
    # print(1)
    df_storer=[]
    img_paths=[]
    outer_index= 0
    
    # Return a response if needed
    df, container_data = load_data_from_files()
    for keys, values in container_data.items():
        selected_truck_spec = truck_specs.get(keys, {})
        # print("selected_truck_specs",selected_truck_spec)
        if outer_index==0:
            df,container_toFit,strip_list= dataProcess_1(df,selected_truck_spec)
        if outer_index!=0:
            # print("df_prev",df)
            df,container_toFit,strip_list = dataProcess_2(df,selected_truck_spec)
            # print("df_after",df)

        roll = values
        index_= 0
        while roll>0:
            filename,df= worker(df,container_toFit,strip_list,keys,index_)
            temp= deepcopy(df)
            df_storer.append(temp.to_html(classes='data')) 
            index_+=1
            roll-=1
            filename = filename.replace('\\', '/')
            img_paths.append(filename)

        outer_index+=1
    # selected_truck_spec = truck_specs.get(truck_spec, {})

    # print(img_paths)

    response = {
    'show_optimal_solution': True,  # Set this to True if you want to display the optimal solution
    'df_html_array': df_storer,  # DataFrame converted to HTML format
    'image_Array': img_paths
    }
    # print("Memory usage at end:", memory_usage(), "MB")

    # # Print maximum memory usage
    # print("Max memory usage:", max_memory_usage, "MB")
    # print(response)

    
    return jsonify(response)

    



def save_data_to_files(df, truck_specifications):
    # Save DataFrame to CSV file
    df.to_csv('static/data.csv', index=False)

    # Save truck specifications to a JSON file
    with open('static/truck_specs.json', 'w') as file:
        json.dump(truck_specifications, file)

def load_data_from_files():
    # Read DataFrame from CSV file
    df = pd.read_csv('static/data.csv')

    # Read truck specifications from JSON file
    with open('static/truck_specs.json', 'r') as file:
        truck_specs = json.load(file)

    return df, truck_specs


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'No file part'
    
    # truck_specification = request.form['truckSpec']
    file = request.files['file']

    
    if file.filename == '':
        return 'No selected file'
    total_containers = int(request.form['totalContainers'])

    # Retrieve the type and count of each container
    container_data = {}
    for i in range(1, total_containers + 1):
        container_type = request.form['containerType{}'.format(i)]
        container_count = int(request.form['containerCount{}'.format(i)])
        container_data[container_type] = container_count

    # print(container_data)

    
    df = pd.read_excel(file)
    save_data_to_files(df, container_data)
    # df,container_toFit,strip_list= dataProcess_1(df,selected_truck_spec)

    # print(df)
    df_storer=[]
    img_paths=[]
    outer_index= 0
 
    # for keys, values in container_data.items():
    #     roll = values
    #     index_= 0
    #     while roll>0:
    #         filename,df= perform_computation(df,container_toFit,strip_list,keys,index_)
    #         df_storer.append(df.to_html(classes='data')) 
    #         index_+=1
    #         roll-=1
    #         img_paths.append(filename)

    for keys, values in container_data.items():
        selected_truck_spec = truck_specs.get(keys, {})
        # print("selected_truck_specs",selected_truck_spec)
        if outer_index==0:
            df,container_toFit,strip_list= dataProcess_1(df,selected_truck_spec)
        if outer_index!=0:
            # print("df_prev",df)
            df,container_toFit,strip_list = dataProcess_2(df,selected_truck_spec)
            # print("df_after",df)

        roll = values
        index_= 0
        while roll>0:
            filename,df= perform_computation(df,container_toFit,strip_list,keys,index_)
            df_storer.append(df.to_html(classes='data')) 
            index_+=1
            roll-=1
            img_paths.append(filename)

        outer_index+=1

        # print(img_paths)
    

    return render_template('output.html', tables=df_storer,img_paths = img_paths)

if __name__ == '__main__':
    app.run(debug=True)
