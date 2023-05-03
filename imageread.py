# This script uses skimage. Use conda install scikit-image if not installed yet
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from skimage import io 

# Define the path to the images
base_path = "C:\\Users\\20203531\\OneDrive\\Marcus_RemoteCodeStash\\RCS_BEP\\Data\\"
path = base_path + "Case1Pack\\Images\\"
images=[]
for filename in os.listdir(path):
    print('hi', filename)
    img=mpimg.imread(path + filename)
    # plt.imshow(path + 'case1_T00_s.img')
    # images.append(os.listdir(path + filename))


img=mpimg.imread('myimage.png')
# end
# from now on you can use img as an image, but make sure you know what you are doing!
imgplot=plt.imshow(img)
plt.show()

#%%
fig, axs = plt.subplots(2, len(no_metastasis_images), figsize=(2*nr_images,3))
    for i in range(len(no_metastasis_images)):
        axs[0, i].imshow(no_metastasis_images[i])
        axs[0, i].set_title("No Metastasis")
        axs[0, i].axis("off")

#%%
# Load the images from the two classes into arrays
no_metastasis_images = []
metastasis_images = []

# Change the number of images you want to compare
nr_images = 50

# Add imaged to lists
for filename in os.listdir(path + "/0/")[0:nr_images]:
    no_metastasis_images.append(io.imread(path + "/0/" + filename))
    
for filename in os.listdir(path + "/1/")[0:nr_images]:
    metastasis_images.append(io.imread(path + "/1/" + filename))


# Plot the images from the two classes
# Vertical=True will plot the images underneath each other
vertical=True
if vertical==True:
    fig, axs = plt.subplots(len(no_metastasis_images), 2, figsize=(3, 2*nr_images))
    for i in range(len(no_metastasis_images)):
        axs[i, 0].imshow(no_metastasis_images[i])
        axs[i, 0].set_title("No Metastasis")
        axs[i, 0].axis("off")

    for i in range(len(metastasis_images)):
        axs[i, 1].imshow(metastasis_images[i])
        axs[i, 1].set_title("Metastasis")
        axs[i, 1].axis("off")
else:
    fig, axs = plt.subplots(2, len(no_metastasis_images), figsize=(2*nr_images,3))
    for i in range(len(no_metastasis_images)):
        axs[0, i].imshow(no_metastasis_images[i])
        axs[0, i].set_title("No Metastasis")
        axs[0, i].axis("off")

    for i in range(len(metastasis_images)):
        axs[1, i].imshow(metastasis_images[i])
        axs[1, i].set_title("Metastasis")
        axs[1, i].axis("off")
plt.show()