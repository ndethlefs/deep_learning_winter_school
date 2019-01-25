from skimage.io import imread, imsave
from skimage.color import rgb2gray, rgb2grey, rgb2hsv
 
# read the image
inp_image = imread("./1.jpg")
# replace the path above with the absolute path of the image you want to read
 
#write image back to file
 
#parameter 1: path where the image has to be saved.
#parameter 2: the array of the image.
imsave("./1-new.jpg",inp_image)

# turn the image into a grey image

img_grey = rgb2grey(inp_image) # rgb2gray(inp_image) can also be used
print(img_grey.shape)
 
imsave("./1_grey.jpg",img_grey)

# turn to hsv

hsv_img = rgb2hsv(inp_image)
imsave("./1_hsv.jpg",hsv_img)

inp_image_flip = inp_image[:, ::-1]

imsave("./1_flip.jpg",inp_image_flip)