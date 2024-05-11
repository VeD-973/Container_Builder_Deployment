from flask import Flask, render_template, request
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy
app = Flask(__name__)

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

@app.route('/')
def index():
    return render_template('index.html')


def perform_computation(data,truck_spec):
    # Perform computations here using the received data
    # Example:
    # print(data)
    # print(truck_spec)
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

    # """#### Taking inputs"""

    # def read_csv_file(file_path):
    #     """Read data from a CSV file and return as a list of lists."""
    #     data_lists = []
    #     with open(file_path, 'r') as file:
    #         csv_reader = csv.reader(file)
    #         for row in csv_reader:
    #             data_lists.append(row)
    #     return data_lists

    # # Example usage:
    # # Tolerance = input("Please enter the tolerance you want to set (in cms)")
    # case_num = input("Put the case you want to use:")

    # file_path = f'Case{case_num}.csv'  # Provide the path to your CSV file # Provide the path to your CSV file
    # data_lists = read_csv_file(file_path)

    # # Convert data_lists to a Pandas DataFrame
    # del data_lists[0]
    # df = pd.DataFrame(data_lists)
    # # Print the DataFrame
    # # print(df)

    # """#### Segregating Data into the values which are required for computation"""

    # gross_weight=[]
    # net_weight=[]
    # vol=[]
    # temperature=[]
    # length =[]
    # width =[]
    # height=[]
    # numOfcases=[]


    # for index in range(len(data_lists)):
    #     # if index==0 :
    #     #     continue
    #     for index2 in range(len(data_lists[index])):
    #         if index2==0:
    #             gross_weight.append(data_lists[index][index2])
    #         if index2==1:
    #             net_weight.append(data_lists[index][index2])
    #         if index2==2:
    #             vol.append(data_lists[index][index2])
    #         if index2==3:
    #             temperature.append(data_lists[index][index2])
    #         if index2==4:
    #             length.append(data_lists[index][index2])
    #         if index2==5:
    #             width.append(data_lists[index][index2])
    #         if index2==6:
    #             height.append(data_lists[index][index2])
    #         if index2==7:
    #             numOfcases.append(data_lists[index][index2])


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
    df.drop(columns=['Marked'], inplace=True)

    # Add 'BoxNumber' column
    df['BoxNumber'] = df.index

    # Define color dictionary
    colors = {0: 'red', 1: 'blue', 2: 'yellow', 3: 'orange', 4: 'green', 5: 'violet', 6: 'white', 7: 'indigo', 8: 'purple'}

    # Add 'Color' column
    df['Color'] = df['BoxNumber'].map(colors)
    df = df[['BoxNumber', 'Color'] + [col for col in df.columns if col not in ['BoxNumber', 'Color']]]
    # print(df)


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
            
            



    def after_plac(x,y,z,end_x,box_num,strip_list,container,ax,color,curr_weight,stored_plac,row,strip_storage,prev_y,prev_row,prev_row_num,area_covered,y_min):
        width_container = float(container.width)
        height_container = float(container.height)
        depth_container = float(container.length)
        max_weight = container.max_weight


        init_x=x
        init_y=y
        init_z=z

        total_strips = strip_list[box_num][4]
        box_length = deepcopy(float(strip_list[box_num][0]))
        box_width = deepcopy(float(strip_list[box_num][1]))
        box_height = deepcopy(float(strip_list[box_num][2]))

        change_init = invertOrNot(x,end_x,box_num,box_length,box_width,box_height,width_container,total_strips)
        if change_init == True:
            
            y = y+box_length
            temp = deepcopy(box_length)
            box_length = deepcopy(box_width)
            box_width = deepcopy(temp)
            prev_row[len(prev_row)-1][1] = prev_row[len(prev_row)-1][1] + box_width- box_length
            # storage_strip[len(storage_strip)-1][1] = storage_strip[len(storage_strip)-1][1] + box_width- box_length
            y = y-box_length



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
                    while num_strips > 0 and curr_weight < max_weight:
                        ax.bar3d(x, y, z, box_width, box_length, box_height, color=color, edgecolor='black')
                        z += box_height
                        num_strips -= 1
                        curr_weight+=strip_list[box_num][9]
                    # storage_strip.append([x,y,z,box_length,box_width,box_height,box_num])
                    # x += box_width.
                    z = 0
                    total_strips -= 1
                    # prev_row[len(prev_row)-1][2]=deepcopy(x)
                    area_covered+=box_length*box_width
                    if total_strips==0:
                        x+=box_width
                        continue
                    if total_strips>0:
                        if y-box_length>=0:
                            y-=box_length
                        else:
                            continue
                        prev_row[len(prev_row)-1][1]=deepcopy(y)
                        storage_strip[len(storage_strip)-1][1]=deepcopy(y)
                        

                        num_strips = strip_list[box_num][3]
                        # print("UES")
                        while num_strips > 0 and curr_weight < max_weight:
                            ax.bar3d(x, y, z, box_width, box_length, box_height, color=color, edgecolor='black')
                            z += box_height
                            num_strips -= 1
                            curr_weight+=strip_list[box_num][9]
                        x+=box_width
                        # print("Yes goes in",end_x)
                        # print("Y_before",y)

                        if x+box_width<=end_x:
                            y+=box_length
                        z=0
                        # print("Y_after",y)

                        total_strips-=1
                        area_covered+=box_length*box_width
                    
                    # if total_strips==0:


                else:
                    num_strips = strip_list[box_num][3]
                    while num_strips > 0 and curr_weight < max_weight:
                        ax.bar3d(x, y, z, box_width, box_length, box_height, color=color, edgecolor='black')
                        z += box_height
                        num_strips -= 1
                        curr_weight+=strip_list[box_num][9]
                    # storage_strip.append([x,y,z,box_length,box_width,box_height,box_num])
                    x += box_width
                    area_covered+=box_length*box_width
                    z = 0
                    total_strips -= 1

                

            else: 
                # x = end_x
                y_min = min(y_min,y)
                if x+box_width> width_container:
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
                if end_x-x< (x+box_width)-end_x:  ##Checks the better one between putting one extra or shifting according to the previous row
                    x +=(end_x-x)
                else:
                    if len(prev_row) >1 and index<len(prev_row) and prev_row[index][6] == prev_row_num and prev_row[index][3] > prev_y and x + box_width <= width_container:
                        num_strips = strip_list[box_num][3]
                        # print("UES")
                        while num_strips > 0 and curr_weight < max_weight:
                            ax.bar3d(x, y, z, box_width, box_length, box_height, color=color, edgecolor='black')
                            z += box_height
                            num_strips -= 1
                            curr_weight+=strip_list[box_num][9]
                        # storage_strip.append([x,y,z,box_length,box_width,box_height,box_num])
                        x += box_width
                        z = 0
                        total_strips -= 1
                        prev_row[len(prev_row)-1][2]=deepcopy(x)
                        # storage_strip[len(storage_strip)-1][2]=deepcopy(x)

                        area_covered+=box_length*box_width

          
                y,end_x,row,prev_row,prev_y,prev_row_num= (findoptlen(prev_row,x,y,end_x,box_width,row,prev_y,prev_row_num))
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
                storage_strip.append([x,y])

            

        prev_row[len(prev_row)-1].append(x)
        prev_row[len(prev_row)-1].append(y)
        prev_row[len(prev_row)-1].append(box_length)
        prev_row[len(prev_row)-1].append(1)
        prev_row[len(prev_row)-1].append(row)
        y_min = min(y_min,y)


        stored_plac.append([init_x,init_y,init_z,x,y,z,box_num,box_length,box_width,box_height,row])
        return x, y, z,row,prev_y,prev_row,end_x,prev_row_num,area_covered,y_min



    def create_plot(container):
        width_container = float(container.width)
        height_container = float(container.height)
        depth_container = float(container.length)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(0, width_container)
        ax.set_ylim(0, depth_container)
        ax.set_zlim(0, height_container)

        # Set labels for axes
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_box_aspect([1, length_container/width_container, height_container/width_container])


        # Dont show the plot
        plt.ioff()
        return ax


    def create_bottom_view(ax):
        # Adjust the viewing angle for bottom view
        ax.view_init(elev=90, azim=180)

        # Save the plot as an image with a filename including the iteration number
        # ax.text2D(0.05, 0.95, f'Efficiency: {efficiency:.2f}%', transform=ax.transAxes, fontsize=12,
        #         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        
        filename = 'static/bottom_view.png'
        plt.savefig(filename)

        # Close the plot to free up resources
        plt.close()

        return filename  # Return the filename for reference

    # print(strip_list)
    stored_plac = []
    storage_strip=[]
    prev_row= []
    end_x = float(container_toFit.width)
    curr_weight = 0
    prev_y =-1
    y_min = 1e5
    row =0
    area_covered = 0
    prev_row_num=-1
    #Creating Plot
    ax = create_plot(container_toFit)
    x,z= 0,0
    colors= {0: 'red', 1: 'blue', 2: 'yellow', 3: 'orange', 4: 'green', 5: 'violet', 6: 'white', 7: 'indigo',8:'purple'}
    for i in range(len(strip_list)):
        # if i==3:
        #     break
        placed= []
        if len(prev_row)==0 or len(prev_row)==1:
            end_x = float(container_toFit.width)
    
        ans =choose_best_dimension(x,end_x,z,strip_list,container_toFit,stored_plac)
        strip_list[ans][7] = False
        if i == 0:
            y=length_container-strip_list[ans][0]
            y_min = min(y_min,y)
        if y<=0:
            break
        if(i!=0 and row!=0):
            y = y-1+prev_row[len(prev_row)-1][4]-strip_list[ans][0]
            y_min = min(y_min,y)

            prev_row.append([x,y])
            x,y,z,row,prev_y,prev_row,end_x,prev_row_num,area_covered,y_min= after_plac(x,y,z,end_x,ans,strip_list,container_toFit,ax,colors[ans],curr_weight,stored_plac,row,storage_strip,prev_y,prev_row,prev_row_num,area_covered,y_min)
        else:
            if(i!=0 and row==0):
                y = y-1+prev_row[len(prev_row)-1][4]-strip_list[ans][0]
            prev_row.append([x,y])
            x,y,z,row,prev_y,prev_row,end_x,prev_row_num,area_covered,y_min= after_plac(x,y,z,end_x,ans,strip_list,container_toFit,ax,colors[ans],curr_weight,stored_plac,row,storage_strip,prev_y,prev_row,prev_row_num,area_covered,y_min)

    filename_final = create_bottom_view(ax)

    if y_min <0:
        y_min = 0
    # print("y_min",y_min)
    # print("Efficiency",round(area_covered/((container_toFit.length-y_min)*container_toFit.width),2))
    # print("for homogenous end coordinates, ",x,y,z)
    # print(strip_list)
    plt.close('all')

    
    return filename_final,df

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'No file part'
    
    truck_specification = request.form['truckSpec']
    file = request.files['file']
    
    if file.filename == '':
        return 'No selected file'
    
    selected_truck_spec = truck_specs.get(truck_specification, {})
    df = pd.read_excel(file)
  
    result,df= perform_computation(df,selected_truck_spec)
    # print(df)

    return render_template('output.html', table=df.to_html(classes='data'))
