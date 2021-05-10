import matplotlib.pyplot as plt
import os  
import numpy as np
import cv2
from utils import Normalize_image, get_mean_net_and_gross_distance

def plot_image(image):
    plt.figure(figsize=(11, 11))
    image.update()
    image.plot(cmap="gray")

def plot_label(label_function, image):
    resolved_image = image.resolve()
    labels = label_function(resolved_image)
    no_of_labels = labels.shape[-1]
    if type(resolved_image)==list:
        shape = np.asarray(resolved_image).shape
        tot_pixels = shape[1]+shape[2]
        x_frac = shape[2]/tot_pixels
        y_frac = shape[1]/tot_pixels
    else:
        shape = resolved_image.shape
        tot_pixels = shape[0]+shape[1]
        x_frac = shape[1]/tot_pixels
        y_frac = shape[0]/tot_pixels

    plt.figure(figsize=(15*x_frac, 15*no_of_labels*y_frac))
    for i in range(no_of_labels):
        plt.subplot(no_of_labels,1,i+1)
        plt.imshow(labels[..., i], cmap="gray")
    
def plot_image_stack(im_stack):
    num_imgs = im_stack.shape[-1]
    
    shape = im_stack.shape
    tot_pixels = shape[1]+shape[2]
    x_frac = shape[2]/tot_pixels
    y_frac = shape[1]/tot_pixels
    
    plt.figure(figsize=(15*x_frac, 15*num_imgs*y_frac))
    for i in range(num_imgs):
        plt.subplot(num_imgs,1,i+1)
        plt.imshow(im_stack[0,:,:,i], cmap='gray')
        
        
def plot_batch(images, batch_function):
    train_images = batch_function(images.resolve())
    num_imgs = train_images.shape[-1]
    plt.figure(figsize=(10, 10*num_imgs))
    
    for i in range(train_images.shape[-1]):
        plt.subplot(num_imgs,1,i+1)
        plt.imshow(train_images[:,:,i], cmap='gray')


def plot_prediction(model=None, im_stack=None, **kwargs):
    predictions = model.predict(im_stack)
    num_imgs = predictions.shape[-1]
    shape = im_stack.shape
    tot_pixels = shape[1]+shape[2]
    x_frac = shape[2]/tot_pixels
    y_frac = shape[1]/tot_pixels
    
    plt.figure(figsize=(15*x_frac, 15*num_imgs*y_frac))
    for i in range(num_imgs):
        plt.subplot(num_imgs,1,i+1)
        plt.imshow(predictions[0,:,:,i], cmap='gray')


def plot_net_vs_gross_distance(list_of_plankton=None, **kwargs):
    net_distances, gross_distances = get_mean_net_and_gross_distance(list_of_plankton, **kwargs)
    plt.figure(figsize=(8,8))
    plt.axis([0, max(gross_distances[gross_distances!=0])*1.1, 0, max(net_distances[net_distances!=0])*1.1])
    plt.plot(gross_distances[gross_distances!=0], net_distances[net_distances!=0])
    plt.xlabel('mean gross distance')
    plt.ylabel('mean net distance')


def load_and_plot_folder_image(folder_path, frame):
    list_paths = os.listdir(folder_path)   
    image = cv2.imread(folder_path +'\\' + list_paths[frame], 0)
    plt.figure(figsize=(15,15))

    plt.imshow(image, cmap='gray')
    return image


def plot_and_save_track(no_of_frames=10,
               plankton_track=None,
               plankton_dont_track=None,
               folder_path=None,
               frame_im0=0,
               save_images=False,
               show_plankton_track=True,
               show_plankton_dont_track=True,
               show_specific_plankton=None,
               show_numbers_track=True,
               show_numbers_dont_track=True,
               show_numbers_specific_plankton=True,
               specific_plankton=None,
               color_plankton_track='b',
               color_plankton_dont_track='r',
               color_specific_plankton='g',
               im_size_width=640, 
               im_size_height=512,
               x_axis_label='pixels',
               y_axis_label='pixels',
               pixel_length_ratio=1,
               save_path=None,
               frame_name='track',
               file_type='.jpg',
               **kwargs):

    list_paths = os.listdir(folder_path)
    for i, j in enumerate(range(frame_im0, frame_im0 + no_of_frames)):
        fig, ax = plt.subplots(figsize=(10, 10), constrained_layout=True)
        im = cv2.imread(folder_path +'\\' + list_paths[j], 0)
        dims = im.shape

        scale_height = dims[0]/im_size_height
        scale_width = dims[1]/im_size_width
        
        if show_plankton_track:
            for key in plankton_track:
                ax.plot(scale_width*plankton_track[key].positions[max(i-5,0):i+1, 1],scale_height*plankton_track[key].positions[max(i-5,0):i+1, 0], c=color_plankton_track,linewidth=1)
                ax.scatter(scale_width*plankton_track[key].positions[i,1], scale_height*plankton_track[key].positions[i,0], s=100, marker='.', facecolor='none', edgecolors=color_plankton_track)

        if show_plankton_dont_track:    
            for key in plankton_dont_track:
                ax.plot(scale_width*plankton_dont_track[key].positions[max(i-5,0):i+1, 1],scale_height*plankton_dont_track[key].positions[max(i-5,0):i+1, 0], c=color_plankton_dont_track, linewidth=1)
                ax.scatter(scale_width*plankton_dont_track[key].positions[i,1], scale_height*plankton_dont_track[key].positions[i,0], s=100, marker='.', facecolor='none', edgecolors=color_plankton_dont_track)

        if show_specific_plankton:
            for num in specific_plankton:
                ax.plot(scale_width*plankton_track['plankton%d' % num].positions[max(i-5,0):i+1, 1], scale_height*plankton_track['plankton%d' % num].positions[max(i-5,0):i+1, 0], c=color_specific_plankton, linewidth=1)
                ax.scatter(scale_width*plankton_track['plankton%d' % num].positions[i,1], scale_height*plankton_track['plankton%d' % num].positions[i,0], s=100, marker='.', facecolor='none', edgecolors=color_specific_plankton)

        
        ax.imshow(im, cmap="gray")

        if show_numbers_track:
            for key in plankton_track:
                ax.annotate(key.replace('plankton',''), (scale_width*plankton_track[key].positions[i,1]-27, scale_height*plankton_track[key].positions[i,0]-10), color=color_plankton_track, fontsize=10)

        if show_numbers_dont_track:
            for key in plankton_dont_track:
                ax.annotate(key.replace('plankton',''), (scale_width*plankton_dont_track[key].positions[i,1]-27, scale_height*plankton_dont_track[key].positions[i,0]-10), color=color_plankton_dont_track, fontsize=10)

        if show_numbers_specific_plankton:
            for num in specific_plankton:
                ax.annotate(num, (scale_width*plankton_track['plankton%d' % num].positions[i,1]-27, scale_height*plankton_track['plankton%d' % num].positions[i,0]-10), color=color_specific_plankton, fontsize=10)
        
        locs, labels = plt.xticks()
        labels = [int(float(item)*pixel_length_ratio) for item in locs]
        plt.xticks(locs[1:-1], labels[1:-1])
        plt.xlabel(x_axis_label)
        plt.xlabel(y_axis_label)
        
        
        plt.title('Planktons')
        if save_images: 
            plt.savefig(save_path + '\\' + frame_name + '%0{}d'.format(len(str(no_of_frames))) % j + file_type)
            plt.close(fig)
        else:
            
            plt.show()
            
            
def plot_found_positions(positions, width=1280, height=1024):
    plt.figure(figsize=(10,10))
    plt.scatter(positions[0][:,1], positions[0][:,0], cmap='gray', marker='.', facecolor='white')
    plt.gca().set_xlim([0, width])
    plt.gca().set_ylim([0, height])
    plt.gca().invert_yaxis()
    plt.gca().set_facecolor('xkcd:black')
    plt.gca().set_aspect('equal', adjustable='box')