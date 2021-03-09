from deeptrack.scatterers import Sphere
from deeptrack.optics import Brightfield, IlluminationGradient
from deeptrack.noises import Poisson
from deeptrack.math import NormalizeMinMax, Clip
from deeptrack.augmentations import FlipLR, FlipUD
import matplotlib.pyplot as plt
import numpy as np
from utils import Normalize_image, RemoveRunningMean
import cv2
import os

def stationary_spherical_plankton(im_size_height, im_size_width, radius, label=0):
    plankton = Sphere(
    position_unit="pixel",          # Units of position (default meter)
    position=lambda: np.random.rand(2) * np.array([im_size_height, im_size_width]),
    z= lambda:  -1.0 + np.random.rand() * 0.5,
    radius=lambda: ((radius) + np.random.rand() * 0.5 * radius), # Dimensions of the principal axes of the ellipsoid
    refractive_index=lambda: 0.9 + 1*(0.1j + np.random.rand() * 0.00j),
    upsample=4,                      # Amount the resolution is upsampled for accuracy
    particle_type = -1 + label
    )
    return plankton
    

def moving_spherical_plankton(im_size_height, im_size_width, radius, label=0, speed=1):
    plankton = Sphere(
    position_unit="pixel",          # Units of position (default meter)
    position=lambda: np.random.rand(2) * np.array([im_size_height, im_size_width]),
    z= lambda:  -1.5 + np.random.rand() * 0.5,
    radius=lambda: ((20e-6) + np.random.rand() * (3.0e-6)) * radius, # Dimensions of the principal axes of the ellipsoid
    refractive_index=lambda: 0.9 + 1*(0.1j + np.random.rand() * 0.00j),
    upsample=4,                      # Amount the resolution is upsampled for accuracy
    particle_type = 0,
    diffusion_constant=lambda: (1 + np.random.rand() * 9) * 2e-14 * speed,
    alpha=lambda: np.random.rand() * np.pi * 2, # yaw, rotatin about z-axis
    beta=lambda: -np.random.rand() * np.pi * 2, # pitch, rotation about y-axis
    phi=lambda: np.random.rand() * 2 * np.pi, # initial angle
    helix_radius=lambda: (1+np.random.rand()) * 1, # unit: "pixels"
    phi_dot=lambda: (1+np.random.rand()) * np.pi # angular velocity
    )
    return plankton


def get_position_plankton(sequence_step, previous_value, diffusion_constant, 
                          alpha, beta, phi, phi_dot, helix_radius, dt=1/7):
    Rz = np.array([[np.cos(alpha), -np.sin(alpha), 0],[np.sin(alpha), np.cos(alpha), 0],[0 ,0 ,1]])
    Ry = np.array([[np.cos(beta), 0, np.sin(beta)],[0, 1, 0],[-np.sin(beta), 0, np.cos(beta)]])
    R = np.matmul(Rz,Ry)
    position_circle = np.array([0,helix_radius * np.sin(phi + phi_dot * sequence_step * dt), 
                                helix_radius*np.cos(phi + phi_dot * sequence_step * dt)])
    position_tilted_circle = np.matmul(R,position_circle)
    orientation = np.matmul(R,np.array([1, 0, 0]))
    translation = np.sqrt(diffusion_constant * dt) * 1e7
    position_helix = position_tilted_circle + translation * orientation
    return previous_value + position_helix[0:2]

def plankton_brightfield(im_size_height, im_size_width, gradient_amp):
    spectrum = np.linspace(400e-9, 700e-9, 3)

    illumination_gradient = IlluminationGradient(gradient=lambda: np.random.randn(2) * 0.0008 * gradient_amp)

    brightfield_microscope = [Brightfield(
                            wavelength=wavelength,
                            NA=0.9,
                            resolution= 2.5e-6,
                            magnification=8, #0.5/scaling
                            refractive_index_medium=1.33,
                            upscale=1,                    # Upscales the pupil function for accuracy
                            illumination=illumination_gradient,
                            output_region=(0, 0, im_size_height, im_size_width))
                          for wavelength 
                          in spectrum]
    return brightfield_microscope

def create_sample(*arg):
    sample = 0
    for i in range(0, len(arg), 2):
        no_of_plankton = lambda: np.random.randint(int(arg[i+1]*0.66), int(arg[i+1]*1.33))
        sample += arg[i]**no_of_plankton
    return sample

def create_image(noise_amp, sample, microscope, norm_min, norm_max):
    noise = Poisson(snr=lambda: (60 + np.random.rand() * 30) * 1/(max(0.01,noise_amp)))
    incoherently_illuminated_sample = sum([brightfield_microscope_one_wavelegth(sample) 
                                        for brightfield_microscope_one_wavelegth 
                                        in microscope])
    augmented_image = FlipUD(FlipLR(incoherently_illuminated_sample))

    image = augmented_image + noise + NormalizeMinMax(min=norm_min, max=norm_max) + Clip(min=0, max=1) #+ BlurCV2()
    
    return image

def plot_image(image):
    plt.figure(figsize=(11, 11))
    image.update()
    image.plot(cmap="gray")

def get_target_sequence(sequence_of_particles):
    label = np.zeros((*np.asarray(sequence_of_particles).shape[1:3], 4))
    
    X, Y = np.meshgrid(
        np.arange(0, np.asarray(sequence_of_particles).shape[2]), 
        np.arange(0, np.asarray(sequence_of_particles).shape[1])
    )
    indices = np.asarray(sequence_of_particles).shape[0]
    for i in range(indices):
        for property in sequence_of_particles[i].properties:
            if "position" in property:
                position = property["position"]
                distance_map = (X - position[1])**2 + (Y - position[0])**2

                label[distance_map < 3, (property["particle_type"] + 1) * (i+1)] = 1
    label[..., 0] = 1 - np.max(label[..., 1:], axis=-1)
    
    return label


def get_target_image(image_of_particles):
    no_of_types = 1
    for property in image_of_particles.properties:
        if "particle_type" in property:
            no_of_types = max(property['particle_type'], no_of_types)
    label = np.zeros((*image_of_particles.shape[:2], no_of_types + 1))
    X, Y = np.meshgrid(
        np.arange(0, image_of_particles.shape[1]), 
        np.arange(0, image_of_particles.shape[0])
    )
    for property in image_of_particles.properties:
        if "position" in property:
            position = property["position"]
            distance_map = (X - position[1])**2 + (Y - position[0])**2
            label[distance_map < 3, property["particle_type"] + 1] = 1
            
    label[..., 0] = 1 - np.max(label[..., 1:], axis=-1)
    
    return label

def batch_function0(image):
    return np.squeeze(image)




def batch_function1(imaged_particle_sequence):
    images = np.array(np.concatenate(imaged_particle_sequence,axis=-1))
    train_images = np.zeros((images.shape[0],images.shape[1],images.shape[2]-1))
    
    for i in range(images.shape[2]-1):
        train_images[:,:,i] = Normalize_image(images[:,:,i]-images[:,:,i+1])
    return train_images

def batch_function2(imaged_particle_sequence):
    images = np.array(np.concatenate(imaged_particle_sequence,axis=-1))
    train_images = np.zeros((images.shape[0],images.shape[1],images.shape[2]))
    
    for i in range(images.shape[2]-1):
        train_images[:,:,i] = Normalize_image(images[:,:,i]-images[:,:,i+1])
        
    train_images[:,:,-1] = images[:,:,np.int((images.shape[2])/2)]
    return train_images

def batch_function3(imaged_particle_sequence):
    images = np.array(np.concatenate(imaged_particle_sequence,axis=-1))
    train_images = np.zeros((images.shape[0],images.shape[1],images.shape[2]+2))
    
    for i in range(images.shape[2]-1):
        train_images[:,:,i] = Normalize_image(images[:,:,i]-images[:,:,i+1])
        
    for i in range(images.shape[2]):
        train_images[:,:,images.shape[2]-1+i] = images[:,:,i]
    return train_images


