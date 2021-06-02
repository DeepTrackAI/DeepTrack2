import numpy as np
import os  
import cv2
from scipy.ndimage import label
from scipy.spatial.distance import cdist
import pandas as pd
import openpyxl
import glob
from inspect import signature


def Normalize_image(image, min_value=0, max_value=1, **kwargs):
    min_im = np.min(image)
    max_im = np.max(image)
    image = image/(max_im-min_im) * (max_value - min_value)
    return image - np.min(image) + min_value


def RemoveRunningMean(image, path_folder=None, tot_no_of_frames=None, center_frame=None, im_width=None, im_height=None, **kwargs):
    list_paths = os.listdir(path_folder)
    first_file_format = list_paths[0][-3:]
    for i in range(len(list_paths)):
        if list_paths[i][-3:] != first_file_format:
            print('Only the images to be analyzed can be in the folder with the images.')
            break
    frames_one_dir = int(tot_no_of_frames/2)
    first_image = cv2.imread(path_folder +'\\' + list_paths[0], 0)
    mean_image = np.zeros(first_image.shape)
    start_point, end_point = max(center_frame - frames_one_dir,0), min(center_frame + frames_one_dir + 1, len(list_paths))
    for i in range(start_point, end_point):
        mean_image += Normalize_image(cv2.imread(path_folder +'\\' + list_paths[max(i,0)], 0)) / tot_no_of_frames

    img = Normalize_image(cv2.imread(path_folder +'\\' + list_paths[center_frame], 0))
    resized_center_image = cv2.resize(img, dsize=(im_width, im_height), interpolation=cv2.INTER_AREA)
    resized_mean_image = Normalize_image(cv2.resize(mean_image, dsize=(im_width, im_height), interpolation=cv2.INTER_AREA))
    
    return Normalize_image(resized_center_image-resized_mean_image)

def get_mean_image(folder_path, im_size_width, im_size_height):
    list_paths = os.listdir(folder_path)
    mean_img = cv2.imread(folder_path +'\\' + list_paths[0], 0)/len(list_paths)
    for i in range(1,len(list_paths)):
        mean_img =+ cv2.imread(folder_path +'\\' + list_paths[i], 0)/len(list_paths)
    resized_mean_image = cv2.resize(mean_img, dsize=(im_size_width, im_size_height), interpolation=cv2.INTER_AREA)
    return resized_mean_image



def get_image_stack(*args, outputs=None, folder_path=None, 
                    frame_im0=None, im_size_width=None, im_size_height=None, 
                    im_resize_width=None, im_resize_height=None, function_img=[lambda img: 1*img], 
                    function_diff=[lambda img: 1*img], **kwargs):
    list_paths = os.listdir(folder_path)
    im_stack = np.zeros((1, im_size_height, im_size_width, len(outputs)))
    for count, num in enumerate(outputs):
        if type(num) == int:
            img = cv2.imread(folder_path +'\\' + list_paths[frame_im0 + num], 0)
            img = cv2.resize(img, dsize=(im_resize_width, im_resize_height), interpolation=cv2.INTER_AREA)
            
            for i in range(len(function_img)):
                sig = str(signature(function_img[i]))
                if 'kwargs' in sig:
                    img = function_img[i](img, center_frame=frame_im0, **kwargs)
                else:
                    img = function_img[i](img)
                
            im_stack[0, :, :, count] = img
            
        if type(num) == list:
            img0 = Normalize_image(cv2.imread(folder_path +'\\' + list_paths[frame_im0 + num[0]], 0))
            img1 = Normalize_image(cv2.imread(folder_path +'\\' + list_paths[frame_im0 + num[1]], 0))
            diff = img1-img0
            diff = cv2.resize(diff, dsize=(im_resize_width, im_resize_height), interpolation=cv2.INTER_AREA)
            
            for i in range(len(function_diff)):
                    diff = function_diff[i](diff, **kwargs)
            
            im_stack[0, :, :, count] = diff

    return im_stack




def get_blob_center(label, array):
    x, y = np.where(array==label)
    if len(x)==0:
        return np.nan, np.nan
    x_center = np.sum(x)/(len(x))
    y_center = np.sum(y)/(len(y))
    
    return x_center, y_center


def get_blob_centers(prediction, value_threshold=0.5, prediction_size=0, **kwargs):
    prediction[prediction < value_threshold]=0
    prediction[prediction >= value_threshold]=1
    labeled_array, num_features = label(prediction, structure = [[1,1,1],
                                                                 [1,1,1],
                                                                 [1,1,1]])
    centers = np.array([get_blob_center(1, labeled_array)])
    for i in range(2,num_features):
        if np.count_nonzero(labeled_array==(i)) > prediction_size:
            centers = np.vstack((centers,get_blob_center(i, labeled_array)))
    return centers

def Extract_positions_from_prediction(im_stack=None, model=None, layer=None, **kwargs):
    prediction = model.predict(im_stack)[0, :, :, layer]
    positions = get_blob_centers(prediction, **kwargs)
    return positions


def extract_positions(no_of_frames, frame_im0, **kwargs):
    
    positions = [np.nan] * no_of_frames
    for i in range(no_of_frames):
        im_stack = get_image_stack(frame_im0=frame_im0+i, **kwargs)
        positions[i] = Extract_positions_from_prediction(im_stack, **kwargs)
    return positions




class Plankton:
    def __init__(self, position, number_of_timesteps, current_timestep):
        self.positions = np.zeros(shape = (number_of_timesteps, 2), dtype='object')*np.nan
        self.positions[current_timestep,:] = position
        self.number_of_timesteps = number_of_timesteps
    
    def add_position(self, position, timestep):
        self.positions[timestep,:] = position
         
    def get_latest_position(self, timestep=0, threshold=10, **kwargs):
        latest_position = [np.nan, np.nan]
        for i in range(self.number_of_timesteps - timestep + 1, min(threshold + self.number_of_timesteps - timestep, 
                                                                    self.number_of_timesteps+1)):
            
            if np.isfinite(self.positions[-i,0]):
                latest_position = self.positions[-i,:]
                break
        return latest_position
    

        
    def get_mean_velocity(self): 
        no_of_timesteps = len(self.positions[:,0]) 
        no_of_numbers = np.count_nonzero(~np.isnan(self.positions[:,0].astype('float'))) 
        mean_velocity = 0 
        if no_of_numbers > 1: 
            for i in range(no_of_timesteps-1): 
                mean_velocity += np.nansum(np.linalg.norm(self.positions[i,:]-self.positions[i+1,:]))/(no_of_numbers) 
            else: mean_velocity = 0 
            
        return mean_velocity
    
    
def Initialize_plankton(positions=None, number_of_timesteps=None, current_timestep=0, **kwargs):
    if np.any(np.isnan(positions)):
        print('No positions recieved for this time step', 0)
        return
    no_of_plankton = np.shape(positions)[0]
    list_of_plankton = {}
    
    for i in range(no_of_plankton):
        list_of_plankton['plankton%d' % i] = Plankton(positions[i], number_of_timesteps, current_timestep)
    return list_of_plankton

def Update_list_of_plankton(list_of_plankton=None, positions=None, max_dist=10, 
                            timestep=None, threshold=10, extrapolate=False, **kwargs):
    if np.any(np.isnan(positions)):
        print('No positions recieved for time step', timestep)
        return list_of_plankton
    
    if type(positions)==tuple:
        positions = np.reshape(positions, (-1, 2))
        
    if list_of_plankton is None:
        no_of_timesteps = len(positions)
        list_of_plankton = Initialize_plankton(positions=positions, number_of_timesteps=no_of_timesteps, current_timestep=timestep)
        return list_of_plankton
    
    no_of_plankton = len(list_of_plankton)
    no_of_positions = len(positions)
    plankton_positions = np.zeros([no_of_plankton, 2])
    
    for value, key in enumerate(list_of_plankton):
        plankton_positions[value,:] = list_of_plankton[key].get_latest_position(timestep = timestep, threshold = threshold)
      
    distances0 = cdist(positions, plankton_positions)
        
    if extrapolate == True and timestep > 1:
        plankton_positions = Extrapolate_positions(list_of_plankton=list_of_plankton, timestep=timestep, threshold=threshold)
        
    distances = cdist(positions, plankton_positions)
    for i in range(no_of_positions):
        if np.nanmin(distances[i,:]) > max_dist:
            position = positions[i,:]
            list_of_plankton['plankton%d' % len(list_of_plankton)] = Plankton(position, list_of_plankton['plankton%d' % 0].number_of_timesteps, timestep)
        else:
            if np.sum(distances[i,:] < max_dist) > 1:
                temp_indices = np.where(distances[i,:] < max_dist)[0]
                temp_dists = distances0[i,:][temp_indices]
                temp_veldiffs = np.zeros((len(temp_indices),1))
                for j in range(len(temp_dists)):
                    temp_veldiffs[j] = np.abs(temp_dists[j]-list_of_plankton['plankton%d' % temp_indices[j]].get_mean_velocity())
                temp_min_vel = np.min(temp_veldiffs)
                temp_min_index = np.where(temp_veldiffs==temp_min_vel)[0][0]
                list_of_plankton['plankton%d' % temp_indices[temp_min_index]].add_position(positions[i,:], timestep)
                
            else:
                temp_min_dist = np.nanmin(distances[i,:])
                temp_min_index = np.nonzero(distances[i,:]==temp_min_dist)[0][0]
                list_of_plankton['plankton%d' % temp_min_index].add_position(positions[i,:], timestep)
    return list_of_plankton



def assign_positions_to_planktons(positions, **kwargs):
    no_of_timesteps = len(positions)
    list_of_plankton = Initialize_plankton(positions=positions[0], number_of_timesteps=no_of_timesteps)

    for i in range(1,no_of_timesteps):
        list_of_plankton = Update_list_of_plankton(list_of_plankton = list_of_plankton, 
                                                   positions = positions[i], 
                                                   timestep=i, **kwargs)
    return list_of_plankton



def Interpolate_gaps_in_plankton_positions(list_of_plankton=None, **kwargs):
    no_of_timesteps = len(list_of_plankton[list(list_of_plankton.keys())[0]].positions)
    for key in list_of_plankton:
        for j in range(1,no_of_timesteps-1):
            if np.isnan(list_of_plankton[key].positions[j,0]) and np.any(
                np.isnan([list_of_plankton[key].positions[j-1,0], 
                          list_of_plankton[key].positions[j+1,0]]))==False:
                
                list_of_plankton[key].positions[j,0] = (list_of_plankton[key].positions[j-1,0] + 
                                                                     list_of_plankton[key].positions[j+1,0])/2
                
                list_of_plankton[key].positions[j,1] = (list_of_plankton[key].positions[j-1,1] + 
                                                                     list_of_plankton[key].positions[j+1,1])/2
    return list_of_plankton




def Extrapolate_positions(list_of_plankton=None, timestep=2, **kwargs):
    no_of_plankton = len(list_of_plankton)
    plankton_positions = np.zeros([no_of_plankton, 2])
    for value, key in enumerate(list_of_plankton):
        plankton_positions[value,:] = list_of_plankton[key].get_latest_position(timestep=timestep, **kwargs)
    for value, key in enumerate(list_of_plankton):
        if np.isnan(list_of_plankton[key].positions[timestep,0]) and np.any(
            np.isnan([list_of_plankton[key].positions[timestep-1,0], 
                      list_of_plankton[key].positions[timestep-2,1]]))==False:
            
            plankton_positions[value,0] = (2 * list_of_plankton[key].positions[timestep-1,0] 
                                              - list_of_plankton[key].positions[timestep-2,0])
            
            plankton_positions[value,1] = (2 * list_of_plankton[key].positions[timestep-1,1] 
                                              - list_of_plankton[key].positions[timestep-2,1])
                
    return plankton_positions



def Trim_list_from_stationary_planktons(list_of_plankton=None, min_distance=10, **kwargs):
    
    for key in list(list_of_plankton):
        temp_dist=0
        temp_positions = list_of_plankton[key].positions
    
        for i in range(len(temp_positions[:,0])-1):
            temp_dist += np.nansum(np.linalg.norm(temp_positions[i,:]-temp_positions[i+1,:]))
        temp_dist = np.nansum(temp_dist)
        
        if temp_dist < min_distance:
            list_of_plankton.pop(key)

    return list_of_plankton





def split_plankton(list_of_plankton=None, percentage_threshold=0.5, **kwargs):
    no_of_timesteps = list_of_plankton[list(list_of_plankton.keys())[0]].number_of_timesteps
    min_positions = int(percentage_threshold*no_of_timesteps)
    plankton_track = {}
    for key in list_of_plankton:
        number_non_nan = np.count_nonzero(~np.isnan(list_of_plankton[key].positions[:,0].astype('float')))
        if number_non_nan >= min_positions:
            plankton_track[key] = list_of_plankton[key]

    plankton_dont_track = {}
    for key in list_of_plankton:
        number_non_nan = np.count_nonzero(~np.isnan(list_of_plankton[key].positions[:,0].astype('float')))
        if number_non_nan < min_positions:
            plankton_dont_track[key] = list_of_plankton[key]
    return plankton_track, plankton_dont_track


def get_mean_net_and_gross_distance(list_of_plankton=None, use_3D_dist=False, **kwargs):
    no_of_timesteps = list_of_plankton[list(list_of_plankton.keys())[0]].number_of_timesteps
    no_of_plankton = len(list_of_plankton)
    mean_net_distances = np.zeros([no_of_timesteps,1])
    mean_gross_distances = np.zeros([no_of_timesteps,1])

    positions = np.zeros((no_of_timesteps, 2*no_of_plankton))
    for index, key in enumerate(list_of_plankton):
        positions[:,2*index:2*(index+1)] = list_of_plankton[key].positions
        
    for i in range(no_of_plankton):
        counter = 0
        for j in range(no_of_timesteps):
            if np.isnan(positions[j,i*2]):
                counter+=1
            else:
                break
        positions[:,2*i:2*(i+1)] = np.roll(positions[:,2*i:2*(i+1)], -counter, axis=0)
        
        
    for i in range(no_of_timesteps-1):
        counter=0
        for j in range(no_of_plankton):
            if not np.isnan(positions[i+1,2*j]):
                counter+=1
                mean_net_distances[i] += np.linalg.norm(positions[i+1,2*j:2*(j+1)]-positions[0,2*j:2*(j+1)])
                
                for k in range(0,i+1):
                    if np.isnan(positions[k+1,2*j]):
                        break
                    mean_gross_distances[i] += np.linalg.norm(positions[k+1,2*j:2*(j+1)]-positions[k,2*j:2*(j+1)])
        if counter==0:
            
            break
        
        mean_net_distances[i] /= counter
        mean_gross_distances[i] /= counter
        
    if use_3D_dist:
        mean_net_distances = mean_net_distances * np.sqrt(3/2)
        mean_gross_distances = mean_gross_distances * np.sqrt(3/2)
    return mean_net_distances, mean_gross_distances



def save_positions(list_of_plankton, save_path=None, file_format='.xlsx', pixel_length_ratio=1):
    shape_position = np.shape(list_of_plankton[list(list_of_plankton.keys())[0]].positions.astype('float64'))

    positions_array = np.zeros((shape_position[0], len(list_of_plankton)*2))
    
    header = [None]*len(list_of_plankton)*2
    
    for i, key in enumerate(list_of_plankton):
        positions_array[:,2*i:2*(i+1)] = list_of_plankton[key].positions.astype('float64')
        header[2*i]=key + ' x-position'
        header[2*i+1]=key + ' y-position'
    
    
    
    df = pd.DataFrame(positions_array*pixel_length_ratio)
        
    filepath = save_path + file_format
    if file_format == '.xlsx':
        df.to_excel(filepath, header=header)
    
    if file_format == '.csv':
        df.to_csv(filepath, header=header)



def Make_video(frame_im0=0, folder_path=None, save_path=None, fps=7, no_of_frames=3):

    list_paths = os.listdir(folder_path)
    img_array = []
    for i in range(frame_im0, no_of_frames):
    
        img = cv2.imread(folder_path + '\\' + list_paths[i])

        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)

    out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

def crop_and_append_image(image=None, col_delete_list=[0,1], row_delete_list=[0,1], mult_of=16, print_shape=False, **kwargs):
    
    rows0, cols0 = image.shape[0:2]
    
    
    for i in range(int(len(col_delete_list)/2)):
        rows, cols = image.shape[0:2]
        start = int(col_delete_list[2*i]-cols0+cols)
        stop = int(col_delete_list[2*i+1]-cols0+cols)
        image = np.delete(image, slice(start, stop), 1)

    for i in range(int(len(row_delete_list)/2)):
        rows, cols = image.shape[0:2]
        start = int(row_delete_list[2*i]-rows0+rows)
        stop = int(row_delete_list[2*i+1]-rows0+rows)
        image = np.delete(image, slice(start, stop), 0)
    rows, cols = image.shape[0:2]
    
    image = image[0:int(rows/mult_of)*mult_of, 0:int(cols/mult_of)*mult_of] 
    
    if print_shape:
        print(image.shape[0:2])
    
    return image


def fix_positions_from_cropping(positions, col_delete_list=[None], row_delete_list=[None], **kwargs):
    new_positions = positions.copy()
    if len(col_delete_list)>1:
        for i in range(len(new_positions)):
            new_positions[i] = positions[i].copy()
            pixels_to_add = 0
            for j in range(int(len(col_delete_list)/2)):
                if j==0:
                    lower_bound = col_delete_list[j]

                    temp_correction = np.zeros(len(new_positions[i][:,1]))

                else:
                    lower_bound += col_delete_list[j*2] - col_delete_list[j*2-1]
                    
                
                bol_low = lower_bound <= positions[i][:,1]
                
                
                pixels_to_add = col_delete_list[j*2+1]-col_delete_list[j*2]
                
                temp_correction = pixels_to_add * bol_low + temp_correction
                
            new_positions[i][:, 1] += temp_correction
            

            
    if len(row_delete_list)>1:
        for i in range(len(new_positions)):
            pixels_to_add = 0
            for j in range(int(len(row_delete_list)/2)):
                if j==0:
                    lower_bound = row_delete_list[j]

                    temp_correction = np.zeros(len(new_positions[i][:,0]))
                else:
                    lower_bound += row_delete_list[j*2] - row_delete_list[j*2-1]
                    
                bol_low = lower_bound <= positions[i][:,0]

                
                pixels_to_add = row_delete_list[j*2+1]-row_delete_list[j*2]
                temp_correction = pixels_to_add * bol_low + temp_correction
                

            new_positions[i][:, 0] += temp_correction

    return new_positions


def get_track_durations(plankton_track):
    no_of_timesteps = len(plankton_track[list(plankton_track.keys())[0]].positions)
    track_durations = np.zeros(no_of_timesteps)
    for plankton in plankton_track:
        plankton = np.array(plankton_track[plankton].positions[:,0], dtype=float)
        where_list = np.where(np.isfinite(plankton))
        track_start = np.min(where_list)
        track_end = np.max(where_list)

        track_durations[track_end-track_start] += 1
    return track_durations


def get_found_plankton_at_timestep(plankton_track):
    no_timesteps = len(plankton_track[list(plankton_track.keys())[0]].positions)
    found_plankton_at_timestep = np.zeros(no_timesteps)

    for plankton in plankton_track:
        plankton = np.array(plankton_track[plankton].positions[:,0], dtype=float)
        where_list = np.where(np.isfinite(plankton))
        for i in where_list[0]:
            found_plankton_at_timestep[i] += 1
    return found_plankton_at_timestep


def extract_positions_from_list(list_of_plankton):
    shape_position = np.shape(list_of_plankton[list(list_of_plankton.keys())[0]].positions.astype('float64'))
    positions_array = np.zeros((shape_position[0], len(list_of_plankton)*2))
    for i, key in enumerate(list_of_plankton):
        positions_array[:,2*i:2*(i+1)] = list_of_plankton[key].positions.astype('float64')
    return positions_array