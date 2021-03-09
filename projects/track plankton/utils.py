import numpy as np
import os  
from deeptrack.features import LoadImage
import cv2
from scipy.ndimage import label


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


#outputs, output_numbers, folder_path, frame_im0+i, im_size_width, im_size_height


#im_stack, model, layer, value_threshold











