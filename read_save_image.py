import os
import matplotlib.pyplot as plt
import glob
image_path = '/p/qdata/zw6sg/VLCS'
import matplotlib.image as mpimg

subdatasets = glob.glob(image_path+'/*')

for sets in subdatasets:
    classes = glob.glob(os.path.join(*[image_path, sets, '*']))
    for single_class in classes:
        all_imgs = glob.glob(os.path.join(*[single_class, '*']))[:2]
        i = 0
        for image in all_imgs:
            img = mpimg.imread(image)
            plt.imshow(img)
            plt.axis('off')
            plt.savefig('./example_img/'+image.split('/')[5]+image.split('/')[6]+str(i)+'.jpg')
            i+=1



