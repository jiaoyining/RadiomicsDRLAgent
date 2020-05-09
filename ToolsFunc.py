import SimpleITK as sitk
import six
import sys, os
import numpy as np
import csv
from skimage.measure import regionprops, label
import cv2
from cv2 import distanceTransform
from SimpleITK import DanielssonDistanceMap
from PIL import Image
import imageio

def startWith(*startstring):
    starts = startstring

    def run(s):
        f = map(s.startswith, starts)
        if True in f: return s

    return run


def endWith(*endstring):
    ends = endstring

    def run(s):
        f = map(s.endswith, ends)
        if True in f: return s

    return run


def write_csv(dict_1, file_1):
    with open(file_1, 'a') as f:
        for key, value in dict_1.items()[2:]:
            f.write(str(key) + ',')
            f.write(str(value) + '\n')

def single_mask(image, label):
    array = sitk.GetArrayFromImage(image)
    array[array != label] = 0
    array[array == label] = 1
    after_image = sitk.GetImageFromArray(array)
    after_image.CopyInformation(image)
    return after_image


def fuse_mask(image):
    array = sitk.GetArrayFromImage(image)
    array[array > 0] = 1
    after_image = sitk.GetImageFromArray(array)
    after_image.CopyInformation(image)
    return after_image


def select_from_start(file_address, selected):
    listed_file = os.listdir(file_address)
    selector = startWith(selected) #('_p2.nii')
    file_name = list(filter(selector, listed_file))
    file_selected = []

    if not len(file_name) == 0:
        for i_fn in file_name:
            file_selected.append(file_address + i_fn)
        return file_name, file_selected
    else:
        return 0, 0

def select_from_end(file_address, selected):
    listed_file = os.listdir(file_address)
    selector = endWith(selected) #('_p2.nii')
    file_name = list(filter(selector, listed_file))
    file_selected = []

    if not len(file_name) == 0:
        for i_fn in file_name:
            file_selected.append(file_address + i_fn)
        return file_name, file_selected
    else:
        return 0, 0


def normalize(img_arr, scale=255):

    #labelArr=sitk.GetArrayFromImage(label)
    ''''
    min_value = np.percentile(img_arr, 0.1).astype('float')
    max_value = np.percentile(img_arr, 99.9).astype('float')
    img_arr[img_arr > max_value] = max_value
    img_arr[img_arr < min_value] = min_value   #-outliers
    new_arr = (img_arr-min_value)/(max_value-min_value)*scale
    '''
    pos_pos = np.where(img_arr>0)

    hist, bins = np.histogram(img_arr.flatten(), img_arr.max()+1)
    cdf = hist.cumsum()  # 计算累积直方图
    cdf_m = np.ma.masked_equal(cdf, 0)  # 除去直方图中的0值
    cdf_m = (cdf_m - cdf_m.min()) * bins.max()/ (cdf_m.max() - cdf_m.min())  # 等同于前面介绍的lut[i] = int(255.0 *p[i])公式
    cdf = np.ma.filled(cdf_m, 0).astype('int')

    new_arr = cdf[img_arr]
    new_arr = (new_arr-new_arr.min())*255/(new_arr.max()-new_arr.min()).astype('float')

    return new_arr


def order_times(patient_times):

    time_array = np.array(patient_times)
    times_from_previous = np.sort(time_array)

    return times_from_previous


def VisualSegBox(im_boxed, msk_name, dir = './data/IM_RESULT/'):

    color_tab = {1:[0, 0, 255], 5:[0, 255, 0], 6:[255, 0, 0]}
    arr_rst = normalize(np.copy(im_boxed), 255)
    arr_color = np.zeros((np.size(arr_rst, 0), np.size(arr_rst, 1), np.size(arr_rst, 2)))

    msk_addr = './data/VESSEL/SegmentationClass/' + msk_name
    arr_mask = sitk.GetArrayFromImage(sitk.ReadImage(msk_addr))
    regions = regionprops(label(arr_mask))

    for i_region in range(len(regions)):
        pos = regions[i_region].coords[0]
        label_value = arr_mask[int(pos[0]), int(pos[1])]
        if label_value == 2 or label_value == 4 or label_value == 0:
            print(msk_name + ', y:' + str(pos[0]) + ', x:' + str(pos[1]) + ', color:' + str(label_value) + ', wrong' + '\n')
            continue
        color = color_tab[label_value]

        for i_coord in regions[i_region].coords:
            arr_color[int(i_coord[0]), int(i_coord[1]), :] = np.array(color)
            arr_rst[int(i_coord[0]), int(i_coord[1]), :] = \
                arr_rst[int(i_coord[0]), int(i_coord[1]), :]*0.6 + \
                arr_color[int(i_coord[0]), int(i_coord[1]), :]*0.4

    #im_rst = sitk.GetImageFromArray(arr_rst.transpose(1, 2, 0))
    cv2.imwrite(dir + msk_name[:-4] + '_rst.png', arr_rst)

    return 0




def ReadImageWithDist(addr_im, addr_mask, name, id):
    # a = cv2.imread(addr_im)
    im_im = sitk.ReadImage(name)
    arr_im = sitk.GetArrayFromImage(im_im)
    arr_out = np.zeros((4, np.size(arr_im, 1), np.size(arr_im, 2)))

    DisMap = GenDistMap(addr_mask)

    if id == 0:
        arr_out[0, :, :] = arr_im[id, :, :]
        arr_out[1, :, :] = arr_im[id, :, :]
        arr_out[2, :, :] = arr_im[id + 1, :, :]
    elif id == (np.size(DisMap, 0)-1):
        arr_out[0, :, :] = arr_im[id - 1, :, :]
        arr_out[1, :, :] = arr_im[id, :, :]
        arr_out[2, :, :] = arr_im[id, :, :]
    else:
        arr_out[0, :, :] = arr_im[id - 1, :, :]
        arr_out[1, :, :] = arr_im[id, :, :]
        arr_out[2, :, :] = arr_im[id + 1, :, :]

    arr_out[3, :, :] = DisMap[id, :, :]

    return np.transpose(arr_out, (1, 2, 0))

def cut_id(name_id):

    pos = name_id.find('-')
    name = name_id[0:pos]
    id = name_id[pos+1:]
    return name, id


def Scaling(img_arr):
    arr_mean = np.mean(img_arr)
    arr_std = np.std(img_arr)
    arr_out = np.copy(img_arr)
    return arr_out


def GenDistMap(addr_mask):

    im_mask = sitk.ReadImage(addr_mask)
    arr_mask = sitk.GetArrayFromImage(im_mask)
    arr_rec = np.zeros((np.size(arr_mask, 0), np.size(arr_mask, 1), np.size(arr_mask, 2)))
    arr_rec[np.where(arr_mask == 1)] = 1
    im_rec = sitk.GetImageFromArray(arr_rec.astype('uint8'))
    im_DistMap = sitk.DanielssonDistanceMap(im_rec)
    arr_DistMap = sitk.GetArrayFromImage(im_DistMap)

    return arr_DistMap

def draw_gif_sequences_test(step, region_mask, image_name, save_boolean=1):
    # addressing
    image_addr = '../data/BRATS/ALL/t1ce/JPEGImages/' + image_name + '.png'
    mask_addr = '../data/BRATS/ALL/t1ce/SegmentationClass0/' + image_name + '.png'
    if not os.path.exists('../gif/png/'):
        os.mkdir('../gif/png/')

    sav_dir = '../gif/png/' + image_name + '/'
    if not os.path.exists(sav_dir):
        os.makedirs(sav_dir)

    mask = np.array(cv2.imread(mask_addr))
    image = np.array(cv2.imread(image_addr))

    # get bounding boxes

    pos = np.where(region_mask > 0)
    xmin = np.min(pos[0])
    xmax = np.max(pos[0])
    ymin = np.min(pos[1])
    ymax = np.max(pos[1])
    bbox = [ymin, xmin, ymax, xmax]

    # get rgb color of different masks
    Label2RGB = {1: (255, 0, 0, ), 2: (0, 255, 0), 4: (255, 255, 0), 0: (0, 0, 0)}
    # get the background of the a result and annotations
    background = Image.new('RGBA', (image.shape[1]+10, image.shape[0]+10), (255, 255, 255, 255))
    # get images labeled with masks
    mask_platte = np.zeros((mask.shape))
    for i in np.arange(mask.shape[0]):
        for j in np.arange(mask.shape[1]):
            mask_platte[i, j, :] = Label2RGB[mask[i, j, 0]]
    image_masked = 0.6 * image + 0.4*mask_platte

    image_masked = np.asarray(image_masked, np.uint8)
    # draw the bounding boxes
    cv2.rectangle(image_masked, tuple(bbox[0:2]), tuple(bbox[2:4]), (150, 50, 100), 2)
    # set the positions to put the images and the annotations
    img_offset = (5, 5)
    #footnote_offset = (0, 280)
    #q_predictions_offset = (0, 250)

    #paste the images and the annotations to the background
    img_for_paste = Image.fromarray(image_masked)
    background.paste(img_for_paste, img_offset)

    #draw.text(footnote_offset, footnote, (0, 0, 0), font=font)
    #draw.text(q_predictions_offset, q_val_predictions_text, (0, 0, 0), font=font)

    file_name = sav_dir + image_name + '-s' + str(step) + '.png'
    if save_boolean == 1:
        background.save(file_name)

    return background


def drawing_gif(image_name):
    if not os.path.exists('../gif/gif/'):
        os.mkdir('../gif/gif/')
    sav_addr = '../gif/gif/' + image_name + '/'
    if not os.path.exists(sav_addr):
        os.mkdir(sav_addr)

    obj_selector = image_name
    addr = '../gif/png/'

    file_name, file_selected = select_from_start(addr, obj_selector)
    if file_name!=0:
        images = []
        for i_filename in file_name:
            pngs = os.listdir(addr + i_filename + '/')
            max_step = ordering(pngs, image_name)
            for i_step in range(max_step):
                i_png_name = addr + i_filename + '/' + image_name + '-s' + str(i_step) + '.png'
                images.append(imageio.imread(i_png_name))
        # making gif
        imageio.mimsave(sav_addr + obj_selector + '.gif', images, duration=0.5)

        print(image_name + '.gif ---generated' + '\n')
    else:
        print(image_name + '.gif ---not-generated' + '\n')

    return


def ordering(strs, selector):
    length = len(selector)
    steps = []
    for i_str in strs:
        step = int(i_str[(length+2):-4])
        steps.append(step)
    return max(steps)