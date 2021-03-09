import matplotlib.pyplot as plt


def plot_label(label_function, image):
    resolved_image = image.resolve()
    labels = label_function(resolved_image)
    no_of_labels = labels.shape[-1]
    
    plt.figure(figsize=(7,7*no_of_labels))
    for i in range(no_of_labels):
        plt.subplot(no_of_labels,1,i+1)
        plt.imshow(labels[..., i], cmap="gray")
    
def plot_im_stack(im_stack):
    num_imgs = im_stack.shape[-1]
    plt.figure(figsize=(7, 7*num_imgs))
    for i in range(num_imgs):
        plt.subplot(num_imgs,1,i+1)
        plt.imshow(im_stack[0,:,:,i], cmap='gray')