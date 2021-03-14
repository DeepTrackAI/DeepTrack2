import numpy as np
import os  
from deeptrack.features import LoadImage
import cv2
from scipy.ndimage import label
from scipy.spatial.distance import cdist


def Normalize_image(image, min_value=0, max_value=1):
    min_im = np.min(image)
    max_im = np.max(image)
    image = image/(max_im-min_im) * (max_value - min_value)
    return image - np.min(image) + min_value


def RemoveRunningMean(folder_path, tot_no_of_frames, center_frame, im_size_width, im_size_height):
    list_paths = os.listdir(folder_path)
    first_file_format = list_paths[0][-3:]
    for i in range(len(list_paths)):
        if list_paths[i][-3:] != first_file_format:
            print('Only the images to be analyzed can be in the folder with the images.')
            break
    frames_one_dir = int(tot_no_of_frames/2)
    first_image = np.asarray(LoadImage(folder_path +'\\' + list_paths[0]).resolve())
    mean_image = np.zeros(first_image.shape)
    start_point, end_point = max(center_frame - frames_one_dir,0), min(center_frame + frames_one_dir + 1, len(list_paths))
    for i in range(start_point, end_point):
        mean_image += np.asarray(LoadImage(folder_path +'\\' + list_paths[max(i,0)]).resolve()) / tot_no_of_frames
    
    img = cv2.imread(folder_path +'\\' + list_paths[center_frame], 0)
    resized_center_image = Normalize_image(cv2.resize(img, dsize=(im_size_width, im_size_height), interpolation=cv2.INTER_AREA))
    resized_mean_image = Normalize_image(cv2.resize(mean_image, dsize=(im_size_width, im_size_height), interpolation=cv2.INTER_AREA))
    
    return Normalize_image(resized_center_image-resized_mean_image)

def get_mean_image(folder_path, im_size_width, im_size_height):
    list_paths = os.listdir(folder_path)
    mean_img = cv2.imread(folder_path +'\\' + list_paths[0], 0)/len(list_paths)
    for i in range(1,len(list_paths)):
        mean_img =+ cv2.imread(folder_path +'\\' + list_paths[i], 0)/len(list_paths)
    resized_mean_image = Normalize_image(cv2.resize(mean_img, dsize=(im_size_width, im_size_height), interpolation=cv2.INTER_AREA))
    return resized_mean_image



def get_image_stack(
        *args, outputs = None, output_numbers=None, folder_path=None, 
        frame_im0=None, im_size_width=None, im_size_height=None, **kwargs):
    list_paths = os.listdir(folder_path)
    im_stack = np.zeros((1, im_size_height, im_size_width, len(outputs)))
    for count, (img_type, num) in enumerate(zip(outputs, output_numbers)):
        if img_type == "img":
            img = cv2.imread(folder_path +'\\' + list_paths[frame_im0 + num], 0)
            img = cv2.resize(img, dsize=(im_size_width, im_size_height), interpolation=cv2.INTER_AREA)
            
            if "remove_mean" in args:
                mean_img = kwargs["mean_image"]
                img = Normalize_image(img) - mean_img
            
            if "remove_running_mean" in args:
                running_mean_img = RemoveRunningMean(folder_path, kwargs["tot_no_of_frames"], frame_im0 + num,
                                                     im_size_width, im_size_height)
                img = Normalize_image(img) - running_mean_img
                
            if "exp" in args: 
                img = np.exp(Normalize_image(img))
                
            im_stack[0, :, :, count] = Normalize_image(img)
            
        if img_type =="diff":
            img0 = Normalize_image(cv2.imread(folder_path +'\\' + list_paths[frame_im0 + num[0]], 0))
            img1 = Normalize_image(cv2.imread(folder_path +'\\' + list_paths[frame_im0 + num[1]], 0))
            diff = img1-img0
            
            diff = cv2.resize(diff, dsize=(im_size_width, im_size_height), interpolation=cv2.INTER_AREA)
            im_stack[0, :, :, count] = diff
            
            if "normalize_diff" in args:
                im_stack[0, :, :, count] = Normalize_image(diff)
                
            if "abs_diff" in args:
                im_stack[0, :, :, count] = np.abs(diff)
            

    return im_stack




def get_blob_center(label,array):
    x, y = np.where(array==label)
    x_center = np.sum(x)/len(x)
    y_center = np.sum(y)/len(y)
    
    return x_center, y_center


def get_blob_centers(prediction, value_threshold):
    prediction[prediction < value_threshold]=0
    labeled_array, num_features = label(prediction, structure = [[1,1,1],
                                                                 [1,1,1],
                                                                 [1,1,1]])
    centers = get_blob_center(1, labeled_array)
    for i in range(2,num_features):
        if np.count_nonzero(labeled_array==(i)) > 0:
            centers = np.vstack((centers,get_blob_center(i, labeled_array)))
    return centers

def Extract_positions_from_prediction(im_stack=None, model=None, layer=None, value_threshold=0.5, **kwargs):
    prediction = model.predict(im_stack)[0, :, :, layer]
    positions = get_blob_centers(prediction, value_threshold)
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
         
    def get_latest_position(self, timestep, threshold):
        latest_position = [np.nan, np.nan]
        for i in range(self.number_of_timesteps - timestep, min(threshold + self.number_of_timesteps - timestep, self.number_of_timesteps+1)):
            if np.isfinite(self.positions[-i,0]):
                latest_position = self.positions[-i,:]
                break
        return latest_position
    
            
    def get_mean_velocity(self):
        no_of_timesteps = len(self.positions[:,0])
        no_of_numbers = sum(x is not np.nan for x in self.positions[:,0].tolist())
        mean_velocity = 0
        if no_of_numbers > 1:
            for i in range(no_of_timesteps-1):
                mean_velocity += np.nansum(np.linalg.norm(self.positions[i,:]-self.positions[i+1,:]))/(no_of_numbers)
        else:
            mean_velocity = 0
        return mean_velocity
        
def Initialize_plankton(positions=None, number_of_timesteps=None, **kwargs):
    if np.any(np.isnan(positions)):
        print('No positions recieved for this time step', 0)
        return
    no_of_plankton = np.shape(positions)[0]
    list_of_plankton = {}
    
    for i in range(no_of_plankton):
        list_of_plankton['plankton%d' % i] = Plankton(positions[i,:], number_of_timesteps, 0)
    return list_of_plankton

def Update_list_of_plankton(list_of_plankton=None, positions=None, max_dist=10, 
                            timestep=None, threshold=10, extrapolate=False, **kwargs):
    if np.any(np.isnan(positions)):
        print('No positions recieved for this time step', timestep)
        return list_of_plankton
    
    no_of_plankton = len(list_of_plankton)
    no_of_positions = len(positions)
    plankton_positions = np.zeros([no_of_plankton, 2])
    
    for value, key in enumerate(list_of_plankton):
        plankton_positions[value,:] = list_of_plankton[key].get_latest_position(timestep = timestep, threshold = threshold)

    
    if extrapolate == True and timestep > 1:
        plankton_positions = Extrapolate_positions(plankton_positions)
        
    if len(positions.shape)==1:
        positions = np.reshape(positions, (-1, 2))
        no_of_positions = len(positions)
    
    distances = cdist(positions, plankton_positions)
    
    for i in range(no_of_positions):
        if np.nanmin(distances[i,:]) > max_dist:
            position = positions[i,:]
            list_of_plankton['plankton%d' % len(list_of_plankton)] = Plankton(position, list_of_plankton['plankton%d' % 0].number_of_timesteps, timestep)
        else:
            if np.sum(distances[i,:] < max_dist) > 1:
                temp_indices = np.where(distances[i,:] < max_dist)[0]
                temp_dists = distances[i,:][temp_indices]
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




def Extrapolate_positions(list_of_plankton=None, **kwargs):
    no_of_timesteps = len(list_of_plankton[list(list_of_plankton.keys())[0]].positions)
    for key in list_of_plankton:
        for j in range(1,no_of_timesteps):
            if np.isnan(list_of_plankton[key].positions[j,0]) and np.any(
                np.isnan([list_of_plankton[key].positions[j-1,0], 
                          list_of_plankton[key].positions[j-2,1]]))==False:
                
                list_of_plankton[key].positions[j,0] = (2 * list_of_plankton[key].positions[j-1,0] 
                                                                     - list_of_plankton[key].positions[j-2,0])
                
                list_of_plankton[key].positions[j,1] = (2 * list_of_plankton[key].positions[j-1,1] 
                                                                     - list_of_plankton[key].positions[j-2,1])
                
    return list_of_plankton



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


def assign_positions_to_planktons(positions, **kwargs):
    no_of_timesteps = len(positions)
    list_of_plankton = Initialize_plankton(positions=positions[0], number_of_timesteps=no_of_timesteps)

    for i in range(1,no_of_timesteps):
        list_of_plankton = Update_list_of_plankton(list_of_plankton = list_of_plankton, 
                                                   positions = positions[i], max_dist=10, 
                                                   timestep=i, threshold = 10, extrapolate=False)
    return list_of_plankton


def split_plankton(percentage_threshold=0.5, list_of_plankton=None, **kwargs):
    no_of_timesteps = len(list_of_plankton[list(list_of_plankton.keys())[0]].positions)
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






