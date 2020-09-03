import random
import pickle
import argparse
from keras.datasets import cifar10
import numpy as np
import cv2 as cv


global cifar_labels
cifar_labels= {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck'
}

#Variables for filterbank, two binarization filterbank currently implemented, can be changed as wished between GAUSSIAN, ADAPTIVE, CANNY
global fbin_1
global fbin_2
fbin_1 = 'GAUSSIAN'
fbin_2 = 'CANNY'


#Adding functionality for argument parsing into script
parser = argparse.ArgumentParser()
parser.add_argument('-c', default='RGB')
parser.add_argument('-b', default='GAUSSIAN')
parser.add_argument('-t', type=int, default=3)
parser.add_argument('--SHOW', default='No')
parser.add_argument('--DB', default='No')
parser.add_argument('--LABELS', nargs='+', help='Change this to change dataset',
                    default = [1,3])
args=parser.parse_args()

#Selected labels to use, currently only tested with 2 datasets
LABELS = list(map(int, args.LABELS))
print("Labels:", LABELS)
for label in LABELS:
    print(cifar_labels[label])

#RGB, HSV, HLS, GRAYSCALE
COLORTYPE = args.c
#Actually BGR

#OTSU, ADAPTIVE, GAUSSIAN
BINARIZATION_FLAG = args.b

#Show small extract of test images with the given colortype and binarization
#Will only show images as RGB, thus scewing the images for HLS AND HSV "slightly"
SHOW_IMAGES = args.SHOW

#Choose amount of dimensions final binarized image will be presented in
#1 for standard tsetlin machine, 3 for convolutional
tsetlin_mode = args.t

#Wheter or not to enable filterbank function
enable_filterbank = args.DB
if enable_filterbank == "Yes":
    print(" ")
    print("!--------------------------------------------------------------------------!")
    print("FILTERBANK ENABLED, IGNORING -b and -t. Variables must be changed in code")
    print("Current Filterbank is set to: ")
    print("Colortype: ", COLORTYPE)
    print("Binarization: ", fbin_1, " + ", fbin_2)
    print("!--------------------------------------------------------------------------!")
    print(" ")
else:
    print(" ")
    print("!--------------------------------------------------------------------------!")
    print('Extracting with current parameters: ')
    print('Colortype: ',args.c)
    print('Binarization Method: ',args.b)
    print('Tsetlin Mode: ', args.t)
    print("!--------------------------------------------------------------------------!")
    print(" ")

#Stupid variablename, but idk.. For use in pickle file
labellabel = '-'.join(map(str,LABELS))
labellabel += '-'+COLORTYPE+'-'+BINARIZATION_FLAG+'-'+str(tsetlin_mode)+'C'+'-'+'fbank='+str(enable_filterbank)

print(labellabel)

#Load in datasets
(x_train, y_train), (x_test, y_test) = cifar10.load_data()


#!-------------------------!#
#
#
#
#!-------------------------!#
def extract_images_from_label(image_array, label_array, label_to_extract):
    extracted_image_array = []
    extracted_label_array = []
    for i in range(len(label_to_extract)):
        for j in range(len(image_array)):
            if(label_array[j] == label_to_extract[i]):
                extracted_image_array.append(image_array[j])
                extracted_label_array.append(int(i))

    return (extracted_image_array, extracted_label_array)


#!-------------------------!#
#
#
#
#!-------------------------!#
def single_color(image, color):
    p_array = []
    i_array = []
    for row in image:
        for pixel in row:
            p_array.append(pixel[color])
        i_array.append(p_array)
        p_array=[]
    return i_array

#!-------------------------!#
#
#
#
#!-------------------------!#
def grayscale_single_color(image):
    p_array = []
    i_array = []
    for row in image:
        for pixel in row:
            p_array.append(pixel)
        i_array.append(p_array)
        p_array = []
    return i_array

#!-------------------------!#
#
# 
#
#!-------------------------!#
def binarization(image, bin_flag):
    if bin_flag == 'NONE':
        return image
    elif bin_flag == 'GAUSSIAN':
        return cv.adaptiveThreshold(image, 255,
                                    cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,
                                    11, 2)
    elif bin_flag == 'ADAPTIVE':
        return cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_MEAN_C,
                                    cv.THRESH_BINARY, 11, 2)
    elif bin_flag == 'OTSU':
        ret2,th2 = cv.threshold(image, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
        return th2
    elif bin_flag == 'GRANMO':
        ret2,th2 = cv.threshold(image, 127, 255, cv.THRESH_BINARY)
        return th2
    elif bin_flag == 'CANNY':
        return cv.Canny(image, 100, 200)
    else:
        return image


#!-------------------------!#
#
#
#
#!-------------------------!#
def image_to_1d_binarized_array(image, bin_flag):
    stacked = np.empty((0,32),int)
    for color in range(len(image[0][0])):
        img = np.asarray(single_color(image, color))
        img = binarization(img, bin_flag)
        stacked = np.append(stacked, img, axis=0)
    return stacked.ravel().clip(max=1)


#!-------------------------!#
#
#
#
#!-------------------------!#
def image_array_to_1d_binarized_array(image_array, bin_flag):
    stacked_1d = []
    for i in range(len(image_array)):
        bin_arr = image_to_1d_binarized_array(image_array[i], bin_flag).tolist()
        stacked_1d.append(bin_arr)
        if(((i/len(image_array))*100)%10 == 0):
            print("Binarizing ", len(image_array), " images", (i/len(image_array))*100, "% done")
    return np.asarray(stacked_1d)


#!-------------------------!#
#
#
#
#!-------------------------!#
def grayscale_array_to_1d_binarized_array(image_array, bin_flag):
    stacked_1d = []
    for i in range(len(image_array)):
        stacked = np.empty((0,32),int)
        img = np.asarray(grayscale_single_color(image_array[i]))
        img = binarization(img, bin_flag)
        stacked = np.append(stacked, img, axis=0)
        bin_arr = stacked.ravel().clip(max=1).tolist()
        stacked_1d.append(bin_arr)
        if(((i/len(image_array))*100)%10 == 0):
            print("Binarizing ", len(image_array), " images", (i/len(image_array))*100, "% done")
    return np.asarray(stacked_1d)


#!-------------------------!#
#
#
#
#!-------------------------!#
def change_image_type(image, col_flag):
    if col_flag == 'GRAYSCALE':
        return cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    elif col_flag == 'HLS':
        return cv.cvtColor(image, cv.COLOR_BGR2HLS)
    elif col_flag =='HSV':
        return cv.cvtColor(image, cv.COLOR_BGR2HSV)
    elif col_flag == 'RGB':
        return image
    else:
        print("Unknown color flag:")
        return image


#!-------------------------!#
#
#
#
#!-------------------------!#
def change_image_color_scheme(image_array, col_flag):
    changed_color_array = []
    for i in image_array:
        changed_color_array.append(change_image_type(i,col_flag))
    return changed_color_array


#!-------------------------!#
#
#
#
#!-------------------------!#
def binarization_stack(image, bin_flag):
    stacked = image
    for color in range(len(image[0][0])):
        img = np.asarray(single_color(image, color))
        img = binarization(img, bin_flag)
        img_3_channel = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        stacked = np.hstack((stacked, img_3_channel))
    return stacked


#!-------------------------!#
#
#
#
#!-------------------------!#
def show_images_stacked(image_array, batch_size, bin_flag):
    img = binarization_stack(image_array[0], bin_flag)
    img_stack = img
    for i in range(1,batch_size):
        img_stack = np.vstack((img_stack, binarization_stack(image_array[i], bin_flag)))
    cv.imshow('image',img_stack)
    cv.waitKey()

#!-------------------------!#
#
#
#
#!-------------------------!#
def shuffle_dataset(x_dataset, y_dataset):
    pairs = list(zip(x_dataset, y_dataset))
    random.shuffle(pairs)
    x_rand=[]
    y_rand=[]
    for i in range(len(pairs)):
        x_rand.append(pairs[i][0])
        y_rand.append(pairs[i][1])
    return (x_rand, y_rand)


#!-------------------------!#
#
#
#
#!-------------------------!#
def change_tsetlin_mode(train_x, COLORTYPE):
    transposed_x = []
    if COLORTYPE == 'GRAYSCALE':
        for i in range(len(train_x)):
            transposed_x.append(train_x[i].reshape(32,32))
    else:
        for i in range(len(train_x)):
            transposed_x.append(np.transpose(train_x[i].reshape(3,32,32),(1,2,0)))
    return transposed_x


#!-------------------------!#
#
#
#
#!-------------------------!#
def databank_image(image_1, image_2):
    databank = []
    #print(len(image_1))
    for i in range(len(image_1)):
        databank.append(image_1[i])
        if (i % 1024) == 0 and i != 0:
            for j in range(i-1024,i):
                databank.append(image_2[j])
    for i in range(2048,3072):
        databank.append(image_2[i])
    return databank


#!-------------------------!#
#
#
#
#!-------------------------!#
def pickling(train_x_binarized, train_y, test_x_binarized, test_y,
             tsetlin_mode, fbank):
    if tsetlin_mode == 1:
        shuffled_train_x, shuffled_train_y = shuffle_dataset(train_x_binarized, train_y)
        with open('dataset(%s).pkl' %labellabel, 'wb') as f:
            pickle.dump([shuffled_train_x, shuffled_train_y, test_x_binarized, test_y], f)

    elif tsetlin_mode == 3 and fbank == "No":
        shuffled_train_x, shuffled_train_y = shuffle_dataset(change_tsetlin_mode(train_x_binarized, COLORTYPE), train_y)
        with open('dataset(%s).pkl' %labellabel, 'wb') as f:
            pickle.dump([shuffled_train_x, shuffled_train_y, change_tsetlin_mode(test_x_binarized, COLORTYPE), test_y], f)

    else:
        with open('dataset(%s).pkl' %labellabel, 'wb') as f:
            pickle.dump([train_x_binarized, train_y, test_x_binarized, test_y],
                        f)

#!-------------------------!#
#
#
#
#!-------------------------!#
def no_databank_binarization(bin_flag, col_typ, labels):
    #Extract target classes from dataset
    (train_x, train_y) = extract_images_from_label(x_train, y_train, labels)
    (test_x, test_y) = extract_images_from_label(x_test, y_test, labels)

    #Change color scheme
    color_train_x = change_image_color_scheme(train_x, col_typ)
    color_test_x = change_image_color_scheme(test_x, col_typ)

    if COLORTYPE == 'GRAYSCALE':
        train_x_binarized = grayscale_array_to_1d_binarized_array(color_train_x, BINARIZATION_FLAG)
        test_x_binarized = grayscale_array_to_1d_binarized_array(color_test_x, BINARIZATION_FLAG)
    else:
        train_x_binarized = image_array_to_1d_binarized_array(color_train_x, BINARIZATION_FLAG)
        test_x_binarized = image_array_to_1d_binarized_array(color_test_x, BINARIZATION_FLAG)

    return train_x_binarized, train_y, test_x_binarized, test_y




#GRAYSCALE NOT IMPLEMENTED, !!DO NOT USE!!
#!-------------------------!#
#
#
#
#!-------------------------!#
def databank_binarization(bin_flag_1, bin_flag_2, col_typ, labels):
    (train_x, train_y) = extract_images_from_label(x_train, y_train, labels)
    (test_x, test_y) = extract_images_from_label(x_test, y_test, labels)

    color_train_x = change_image_color_scheme(train_x, col_typ)
    color_test_x = change_image_color_scheme(test_x, col_typ)

    train_x_bin_1 = image_array_to_1d_binarized_array(color_train_x,
                                                      bin_flag_1)
    test_x_bin_1 = image_array_to_1d_binarized_array(color_test_x,
                                                      bin_flag_1)
    train_x_bin_2 = image_array_to_1d_binarized_array(color_train_x,
                                                      bin_flag_2)
    test_x_bin_2 = image_array_to_1d_binarized_array(color_test_x,
                                                      bin_flag_2)

    train_databank = []
    test_databank = []

    for i in range(len(train_x_bin_1)):
        train_databank.append(databank_image(train_x_bin_1[i], train_x_bin_2[i]))
    for j in range(len(test_x_bin_1)):
        test_databank.append(databank_image(test_x_bin_1[j], test_x_bin_2[j]))

    return train_databank, train_y, test_databank, test_y


#!-------------------------!#
#
#
#
#!-------------------------!#
def databank_reshape(train_x):
    transposed_x = []
    for i in range(len(train_x)):
        transposed_x.append(np.transpose(np.asarray(train_x[i]).reshape(6,32,32),(1,2,0)))
    return transposed_x



#!-------------------------!#
#
#
#
#!-------------------------!#
def run(col_type, bin_flag, labels, tsetlin_mode, enable_filterbank):
    if enable_filterbank == "Yes":
        fbank_1, train_y, fbank_2, test_y = databank_binarization(fbin_1, fbin_2, col_type, labels)
        filterbank_1 = databank_reshape(fbank_1)
        filterbank_2 = databank_reshape(fbank_2)
        shuffled_train_x, shuffled_train_y = shuffle_dataset(filterbank_1, train_y)
        shuffled_test_x, shuffled_test_y = shuffle_dataset(filterbank_2, test_y)
        pickling(shuffled_train_x, shuffled_train_y, shuffled_test_x,shuffled_test_y, tsetlin_mode, enable_filterbank)
    else:
        train_x, train_y, test_x, test_y = no_databank_binarization(bin_flag,
                                                                    col_type,
                                                                    labels)
        pickling(train_x, train_y, test_x, test_y, tsetlin_mode, enable_filterbank)



#Yes.....
run(COLORTYPE, BINARIZATION_FLAG, LABELS, tsetlin_mode, enable_filterbank)


# CURRENTLY BROKEN
# Nasty addition to view images in both tsetlin modes
# if SHOW_IMAGES == 'YES':
#     for i in range(0,11):
#         if tsetlin_mode == 1:
#             img = binarization_stack(test_x[0], BINARIZATION_FLAG)
#             img_stack = img
#             for i in range(1,11):
#                 img_stack = np.vstack((img_stack, binarization_stack(test_x[i], BINARIZATION_FLAG)))
#             cv.imshow('image',img_stack)
#             cv.waitKey()
#             break
#         elif tsetlin_mode == 3:
#             tsetlin_x=change_tsetlin_mode(test_x_binarized, COLORTYPE)
#             test1 = np.asarray((tsetlin_x[i]),np.uint8)
#             test1 = np.where(test1 == 1,255,0)
#             test2 = np.asarray(test1,np.uint8)
#             cv.imshow('image', test2)
#             print(train_y[i])
# #            print(test1)
#             cv.waitKey()

