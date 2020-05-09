
import numpy as np

import numpy
import pywt
import SimpleITK as sitk
import six
from six.moves import range


def normalize(img_arr, scale=255):

    #labelArr=sitk.GetArrayFromImage(label)
    new_arr = img_arr.copy()
    min_value = np.percentile(img_arr, 0.1).astype('float')
    max_value = np.percentile(img_arr, 99.9).astype('float')
    if (max_value-min_value) == 0:
        new_arr[:, :] = min_value
    else:
        img_arr[img_arr > max_value] = max_value
        img_arr[img_arr < min_value] = min_value   #-outliers
        new_arr = (img_arr-min_value)/(max_value-min_value)*scale

    return new_arr

def get_all_ids(annotations):
    all_ids = []
    for i in range(len(annotations)):
        all_ids.append(get_ids_objects_from_annotation(annotations[i]))
    return all_ids


def get_all_images(image_names, path_voc):
    images = []
    for j in range(np.size(image_names)):
        image_name = image_names[0][j]
        string = path_voc + '/JPEGImages/' + image_name + '.jpg'
        images.append(image.load_img(string, False))
    return images


def get_all_images_pool(image_names, path_voc):
    images = []
    for j in range(np.size(image_names)):
        image_name = image_names[j]
        string = path_voc + '/JPEGImages/' + image_name + '.jpg'
        images.append(image.load_img(string, False))
    return images


def load_images_names_in_data_set(data_set_name, path_voc):
    file_path = path_voc + '/ImageSets/Main/' + data_set_name + '.txt'
    f = open(file_path)
    image_names = f.readlines()
    image_names = [x.strip('\n') for x in image_names]
    if data_set_name.startswith("aeroplane") | data_set_name.startswith("bird") | data_set_name.startswith("cow"):
        return [x.split(None, 1)[0] for x in image_names]
    else:
        return [x.strip('\n') for x in image_names]


def load_images_labels_in_data_set(data_set_name, path_voc):
    file_path = path_voc + '/ImageSets/Main/' + data_set_name + '.txt'
    f = open(file_path)
    images_names = f.readlines()
    images_names = [x.split(None, 1)[1] for x in images_names]
    images_names = [x.strip('\n') for x in images_names]
    return images_names


def mask_image_with_mean_background(mask_object_found, image):
    new_images = []
    for i_image in image:
        new_image = i_image
        size_image = np.shape(mask_object_found)
        for j in range(size_image[0]):
            for i in range(size_image[1]):
                if mask_object_found[j][i] == 1:
                    new_image[j, i, 0] = 0
                    new_image[j, i, 1] = 0
                    new_image[j, i, 2] = 0
        new_images.append(new_image)
    return new_images


def update_image(image, offset, size_mask):

    region_image = np.zeros(np.shape(image)) + -1
    region_image[int(offset[0]):int(offset[0] + size_mask[0]),
                 int(offset[1]):int(offset[1] + size_mask[1]), :] \
                = image[int(offset[0]):int(offset[0] + size_mask[0]),
                        int(offset[1]):int(offset[1] + size_mask[1]), :]
    return region_image





def cut_nameid(name_id):
  pos = name_id.find('-')
  name = name_id[0:pos]
  id = name_id[pos + 1:-4]
  return name, id


def get_multi_modality_images(addr0, name_id):

  name, i_slice = cut_nameid(name_id)
  i_slice = int(i_slice)
  addr = addr0 + name + '/'

  volume = []
  pos_xmin = 100000
  pos_ymin = 100000
  pos_xmax = 0
  pos_ymax = 0

  # get initial arr
  flair_addr = addr + name + '_flair.nii.gz'
  arr_img = sitk.GetArrayFromImage(sitk.ReadImage(flair_addr))

  # cropping
  available_pos = np.where(arr_img>0)
  pos_xmin = np.min((pos_xmin, np.min(available_pos[1])))
  pos_ymin = np.min((pos_ymin, np.min(available_pos[2])))
  pos_xmax = np.max((pos_xmax, np.max(available_pos[1])))
  pos_ymax = np.max((pos_ymax, np.max(available_pos[2])))

  # get image of falir
  arr_flair = arr_img[:, pos_xmin:pos_xmax, pos_ymin:pos_ymax]
  mean_flair = np.mean(arr_flair[np.where(arr_flair != 0)])
  std_flair = np.std(arr_flair[np.where(arr_flair != 0)])
  arr_flair[np.where(arr_flair != 0)] = (arr_flair[np.where(arr_flair != 0)]-mean_flair)/std_flair
  volume0 = np.zeros((np.size(arr_flair, 1), np.size(arr_flair, 2), 3))
  volume0[:, :, 0] = arr_flair[i_slice, :, :].astype(np.float32, copy=False)
  volume0[:, :, 1] = arr_flair[i_slice, :, :].astype(np.float32, copy=False)
  volume0[:, :, 2] = arr_flair[i_slice, :, :].astype(np.float32, copy=False)

  # get arr of other modlaity

  t1_addr = addr + name + '_t1.nii.gz'
  arr_t1 = sitk.GetArrayFromImage(sitk.ReadImage(t1_addr))
  arr_t1 = arr_t1[:, pos_xmin:pos_xmax, pos_ymin:pos_ymax]
  mean_t1 = np.mean(arr_t1[np.where(arr_t1 != 0)])
  std_t1 = np.std(arr_t1[np.where(arr_t1 != 0)])
  arr_t1[np.where(arr_t1 != 0)] = (arr_t1[np.where(arr_t1 != 0)]-mean_t1)/std_t1
  volume1 = np.zeros((np.size(arr_t1, 1), np.size(arr_t1, 2), 3))
  volume1[:, :, 0] = arr_t1[i_slice, :, :].astype(np.float32, copy=False)
  volume1[:, :, 1] = arr_t1[i_slice, :, :].astype(np.float32, copy=False)
  volume1[:, :, 2] = arr_t1[i_slice, :, :].astype(np.float32, copy=False)

  t1ce_addr = addr + name + '_t1ce.nii.gz'
  arr_t1ce = sitk.GetArrayFromImage(sitk.ReadImage(t1ce_addr))
  arr_t1ce = arr_t1ce[:, pos_xmin:pos_xmax, pos_ymin:pos_ymax]
  mean_t1ce = np.mean(arr_t1ce[np.where(arr_t1ce != 0)])
  std_t1ce = np.std(arr_t1ce[np.where(arr_t1ce != 0)])
  arr_t1ce[np.where(arr_t1ce != 0)] = (arr_t1ce[np.where(arr_t1ce != 0)]-mean_t1ce)/std_t1ce
  volume2 = np.zeros((np.size(arr_t1ce, 1), np.size(arr_t1ce, 2), 3))
  volume2[:, :, 0] = arr_t1ce[i_slice, :, :].astype(np.float32, copy=False)
  volume2[:, :, 1] = arr_t1ce[i_slice, :, :].astype(np.float32, copy=False)
  volume2[:, :, 2] = arr_t1ce[i_slice, :, :].astype(np.float32, copy=False)

  t2_addr = addr + name + '_t2.nii.gz'
  arr_t2 = sitk.GetArrayFromImage(sitk.ReadImage(t2_addr))
  arr_t2 = arr_t2[:, pos_xmin:pos_xmax, pos_ymin:pos_ymax]
  mean_t2 = np.mean(arr_t2[np.where(arr_t2 != 0)])
  std_t2 = np.std(arr_t2[np.where(arr_t2 != 0)])
  arr_t2[np.where(arr_t2 != 0)] = (arr_t2[np.where(arr_t2 != 0)]-mean_t2)/std_t2
  volume3 = np.zeros((np.size(arr_t2, 1), np.size(arr_t2, 2), 3))
  volume3[:, :, 0] = arr_t1ce[i_slice, :, :].astype(np.float32, copy=False)
  volume3[:, :, 1] = arr_t1ce[i_slice, :, :].astype(np.float32, copy=False)
  volume3[:, :, 2] = arr_t1ce[i_slice, :, :].astype(np.float32, copy=False)

  return (volume0, volume1, volume2, volume3), (arr_flair, arr_t1, arr_t1ce, arr_t2)


def get_normalized_images(normalized, i_slice):

    arr_flair = normalized[0]
    arr_t1 = normalized[1]
    arr_t1ce = normalized[2]
    arr_t2 = normalized[3]

    volume0 = np.zeros((np.size(arr_flair, 1), np.size(arr_flair, 2), 3))
    volume0[:, :, 0] = arr_flair[i_slice, :, :].astype(np.float32, copy=False)
    volume0[:, :, 1] = arr_flair[i_slice, :, :].astype(np.float32, copy=False)
    volume0[:, :, 2] = arr_flair[i_slice, :, :].astype(np.float32, copy=False)

    volume1 = np.zeros((np.size(arr_t1, 1), np.size(arr_t1, 2), 3))
    volume1[:, :, 0] = arr_t1[i_slice, :, :].astype(np.float32, copy=False)
    volume1[:, :, 1] = arr_t1[i_slice, :, :].astype(np.float32, copy=False)
    volume1[:, :, 2] = arr_t1[i_slice, :, :].astype(np.float32, copy=False)

    volume2 = np.zeros((np.size(arr_t1ce, 1), np.size(arr_t1ce, 2), 3))
    volume2[:, :, 0] = arr_t1ce[i_slice, :, :].astype(np.float32, copy=False)
    volume2[:, :, 1] = arr_t1ce[i_slice, :, :].astype(np.float32, copy=False)
    volume2[:, :, 2] = arr_t1ce[i_slice, :, :].astype(np.float32, copy=False)

    volume3 = np.zeros((np.size(arr_t2, 1), np.size(arr_t2, 2), 3))
    volume3[:, :, 0] = arr_t1ce[i_slice, :, :].astype(np.float32, copy=False)
    volume3[:, :, 1] = arr_t1ce[i_slice, :, :].astype(np.float32, copy=False)
    volume3[:, :, 2] = arr_t1ce[i_slice, :, :].astype(np.float32, copy=False)


    return (volume0, volume1, volume2, volume3)



def getWaveletImage(inputImage):
    """
    Applies wavelet filter to the input image and yields the decompositions and the approximation.

    Following settings are possible:

    - start_level [0]: integer, 0 based level of wavelet which should be used as first set of decompositions
    from which a signature is calculated
    - level [1]: integer, number of levels of wavelet decompositions from which a signature is calculated.
    - wavelet ["coif1"]: string, type of wavelet decomposition. Enumerated value, validated against possible values
    present in the ``pyWavelet.wavelist()``. Current possible values (pywavelet version 0.4.0) (where an
    aditional number is needed, range of values is indicated in []):

    - haar
    - dmey
    - sym[2-20]
    - db[1-20]
    - coif[1-5]
    - bior[1.1, 1.3, 1.5, 2.2, 2.4, 2.6, 2.8, 3.1, 3.3, 3.5, 3.7, 3.9, 4.4, 5.5, 6.8]
    - rbio[1.1, 1.3, 1.5, 2.2, 2.4, 2.6, 2.8, 3.1, 3.3, 3.5, 3.7, 3.9, 4.4, 5.5, 6.8]

    Returned filter name reflects wavelet type:
    wavelet[level]-<decompositionName>

    N.B. only levels greater than the first level are entered into the name.

    :return: Yields each wavelet decomposition and final approximation, corresponding imaget type name and ``kwargs``
    (customized settings).
    """

    axes = [1, 0]
    rst = _swt3(inputImage[:, :, None], tuple(axes))

    return rst



def _swt3(matrix, axes):  # Stationary Wavelet Transform 3D
    wavelet = 'coif1'
    level = 1
    start_level = 0

    # This function gets a numpy array from the SimpleITK Image "inputImage"
    matrix = numpy.asarray(matrix) # The function np.asarray converts "matrix" (which could be also a tuple) into an array.
    if matrix.ndim != 3:
        raise ValueError('Expected 3D data array')

    original_shape = matrix.shape
    # original_shape becomes a tuple (?,?,?) containing the number of rows, columns, and slices of the image
    padding = tuple([(0, 1 if dim % 2 != 0 else 0) for dim in original_shape])
    # padding is necessary because of pywt.swtn (see function Notes)
    data = matrix.copy()  # creates a modifiable copy of "matrix" and we call it "data"
    data = numpy.pad(data, padding, 'wrap')  # padding the tuple "padding" previously computed

    for i in range(0, start_level):  # if start_level = 0 this for loop never gets executed
        dec = pywt.swtn(data, wavelet, level=1, start_level=0, axes=axes)[0] # computes all decompositions and saves them in "dec" dict
        data = dec['a' * len(axes)].copy()  # copies in "data" just the "aaa" decomposition (if len(axes) = 3)

    ret = []  # initialize empty list
    for i in range(start_level, start_level + level):
        dec = pywt.swtn(data, wavelet, level=1, start_level=0, axes=axes)[0]  # computes the n-dimensional stationary wavelet transform
        data = dec['a' * len(axes)].copy()

        dec_im = {}  # initialize empty dict
        for decName, decImage in six.iteritems(dec):

            decTemp = decImage.copy()
            decTemp = decTemp[tuple(slice(None, -1 if dim % 2 != 0 else None) for dim in original_shape)]
            sitkImage = decTemp
            #print(str(decName).replace('a', 'L').replace('d', 'H') + "\n")
            dec_im[str(decName).replace('a', 'L').replace('d', 'H')] = sitkImage

    rst = np.zeros((original_shape[0], original_shape[1], 4))
    rst[:, :, 0] = dec_im['LL'][:, :, 0]
    rst[:, :, 1] = dec_im['LH'][:, :, 0]
    rst[:, :, 2] = dec_im['HL'][:, :, 0]
    rst[:, :, 3] = dec_im['HH'][:, :, 0]

    return rst  # returns the approximation and the detail (ret) coefficients of the stationary wavelet decomposition



