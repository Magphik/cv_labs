import matplotlib.pyplot as plt
import cv2
from skimage.feature import greycomatrix, greycoprops
from skimage import data

PATCH_SIZE = 21

# open the camera image
# image = data.camera()
image = cv2.imread('Data/nasa4.jpg', cv2.IMREAD_GRAYSCALE)


# select some patches from grassy areas of the image
nasa_locations = [(338, 234), (139, 328), (37, 337), (345, 379)]
nasa_patches = []
for loc in nasa_locations:
    nasa_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,
                                loc[0]:loc[0] + PATCH_SIZE])



# select some patches from sky areas of the image
water_locations = [(190, 54), (242, 53), (444, 72), (155, 85)]
water_patches = []
for loc in water_locations:
    water_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,
    loc[1]:loc[1] + PATCH_SIZE])



    # compute some GLCM properties each patch

xs = []
ys= []

for patch in (nasa_patches + water_patches):
    glcm = greycomatrix(patch, distances=[5], angles= [0],
                        levels = 256, symmetric = True, normed = True)

    xs.append(greycoprops(glcm, 'dissimilarity')[0, 0])
    ys.append(greycoprops(glcm, 'correlation')[0, 0])

fig = plt.figure(figsize = (8,8))



# display original image with locations of patches
ax = fig.add_subplot(3, 2, 1)
ax.imshow(image, cmap=plt.cm.gray,
          vmin=0, vmax=255)

for (y, x) in nasa_locations:
    ax.plot(x + PATCH_SIZE/2, y + PATCH_SIZE/2, 'gs')

for (y, x) in water_locations:
    ax.plot(x + PATCH_SIZE/2, y + PATCH_SIZE/2, 'bs')
ax.set_xlabel('Original Image')
ax.set_xticks([])
ax.set_yticks([])
ax.axis('image')

# for each patch, plot (dissimilarity, correlation)

ax = fig.add_subplot(3, 2, 2)

ax.plot(xs[:len(nasa_patches)], ys[:len(nasa_patches)], 'go',
        label = 'Earth')

ax.plot(xs[len(nasa_patches):], ys[len(nasa_patches):], 'bo',
        label = 'Water')

ax.set_xlabel('GLCM Dissimilarity')

ax.set_ylabel('GLCM Correlation')

ax.legend()

# display the image patches
for i, patch in enumerate(nasa_patches):
    ax = fig.add_subplot(3, len(nasa_patches), len(nasa_patches) * 1 + i + 1)
    ax.imshow(patch, cmap=plt.cm.gray,vmin = 0, vmax = 255)
    ax.set_xlabel("Earth %d" % (i + 1))



for i, patch in enumerate(water_patches):
    ax = fig.add_subplot(3, len(water_patches), len(water_patches) * 2 + i + 1)
    ax.imshow(patch, cmap=plt.cm.gray, vmin = 0, vmax = 255)
    ax.set_xlabel("Water %d" % (i + 1))

    # display the patches and plot
fig.suptitle('Grey level co-occurrence matrix features', fontsize=14, y=1.05)
plt.tight_layout()

plt.show()