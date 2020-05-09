import SimpleITK as sitk
import numpy as np
import os, sys
from ToolsFunc import select_from_start, select_from_end, normalize
from sklearn.model_selection import KFold, train_test_split
from skimage.measure import regionprops, label
from PIL import Image
import pprint
from xml.dom.minidom import parseString
#from xml.etree.ElementTree import SubElement, Element, tostring
from lxml.etree import Element, SubElement, tostring, ElementTree
import xml.etree.cElementTree as et
import xml.etree.ElementTree as ET

def ToPNG(addr, sav_addr):

    Modality = ('flair', 't1', 't2', 't1ce', )

    sub = os.listdir(addr)
    name_list = []
    for i_sub in sub:
        sub_addr = addr + i_sub + '/'
        name_list.append(i_sub)

        for i_mol in Modality:

            # get array of image
            img_name, img_addr_name = select_from_end(sub_addr, i_mol + '.nii.gz')
            im_img = sitk.ReadImage(img_addr_name[0])
            arr_img = sitk.GetArrayFromImage(im_img)

            # get array of mask
            mask_name, mask_addr_name = select_from_end(sub_addr, 'seg.nii.gz')
            im_mask = sitk.ReadImage(mask_addr_name[0])
            arr_mask = sitk.GetArrayFromImage(im_mask)


            for i_slice in range(np.size(arr_mask, 0)):

                # cropping
                img_slice = normalize(arr_img[i_slice, :, :])
                mask_slice = arr_mask[i_slice, :, :]

                # new combined mask
                mask_whole0 = np.zeros(mask_slice.shape)
                mask_whole1 = np.zeros(mask_slice.shape)
                mask_whole2 = np.zeros(mask_slice.shape)

                mask_whole0[np.where(mask_slice > 0)] = 1
                mask_whole1[np.where(mask_slice == 1)] = 1
                mask_whole1[np.where(mask_slice == 4)] = 1
                mask_whole2[np.where(mask_slice == 1)] = 1

                # calculating regions of combined mask and
                regions = regionprops(label(mask_slice))
                print(str(img_name) + str(i_slice) + '\n')
                #regions_whole0 = regionprops(label(mask_whole0))
                #regions_whole1 = regionprops(label(mask_whole1))
                #regions_whole2 = regionprops(label(mask_whole2))

                # condition
                if len(regions) == 0:
                    continue

                # set saving addr
                img_slice_addr = sav_addr + '/' + i_mol + '/JPEGImages/'
                mask_slice_addr = sav_addr + '/' + i_mol + '/SegmentationClass/'
                mask_whole0_addr = sav_addr + '/' + i_mol + '/SegmentationClass0/'
                mask_whole1_addr = sav_addr + '/' + i_mol + '/SegmentationClass1/'
                mask_whole2_addr = sav_addr + '/' + i_mol + '/SegmentationClass2/'

                if not os.path.exists(img_slice_addr):
                    os.mkdir(sav_addr + '/' + i_mol + '/')
                    os.mkdir(img_slice_addr)
                    os.mkdir(mask_slice_addr)
                    os.mkdir(mask_whole0_addr)
                    os.mkdir(mask_whole1_addr)
                    os.mkdir(mask_whole2_addr)

                # save cropped images and masks
                img_slice_name = img_slice_addr + i_sub + '-' + str(i_slice) + '.png'
                mask_slice_name = mask_slice_addr + i_sub + '-' + str(i_slice) + '.png'
                mask_whole0_name = mask_whole0_addr + i_sub + '-' + str(i_slice) + '.png'
                mask_whole1_name = mask_whole1_addr + i_sub + '-' + str(i_slice) + '.png'
                mask_whole2_name = mask_whole2_addr + i_sub + '-' + str(i_slice) + '.png'

                im_img = sitk.GetImageFromArray(img_slice.astype('uint8'))
                im_mask = sitk.GetImageFromArray(mask_slice.astype('uint8'))
                im_mask_whole0 = sitk.GetImageFromArray(mask_whole0.astype('uint8'))
                im_mask_whole1 = sitk.GetImageFromArray(mask_whole1.astype('uint8'))
                im_mask_whole2 = sitk.GetImageFromArray(mask_whole2.astype('uint8'))

                sitk.WriteImage(im_img, img_slice_name)
                sitk.WriteImage(im_mask, mask_slice_name)
                sitk.WriteImage(im_mask_whole0, mask_whole0_name)
                sitk.WriteImage(im_mask_whole1, mask_whole1_name)
                sitk.WriteImage(im_mask_whole2, mask_whole2_name)

    return name_list


def generating_xmls(addr):

    Modality = ('flair', 't1', 't1ce', 't2')


    for i_mol in Modality:
        PNG_MD_addr = addr + i_mol + '/JPEGImages/'
        SEG_MD_addr = addr + i_mol + '/SegmentationClass/'
        SEG0_MD_addr = addr + i_mol + '/SegmentationClass0/'
        SEG1_MD_addr = addr + i_mol + '/SegmentationClass1/'
        SEG2_MD_addr = addr + i_mol + '/SegmentationClass2/'

        img_list = os.listdir(PNG_MD_addr)

        xml_addr = addr + i_mol + '/Annotations/'
        if not os.path.exists(xml_addr):
            os.mkdir(xml_addr)

        for i_img in img_list:

            # get array of images and segmentation
            arr_img = sitk.GetArrayFromImage(sitk.ReadImage(PNG_MD_addr + i_img))
            arr_mask = sitk.GetArrayFromImage(sitk.ReadImage(SEG_MD_addr + i_img))
            arr0_mask = sitk.GetArrayFromImage(sitk.ReadImage(SEG0_MD_addr + i_img))
            arr1_mask = sitk.GetArrayFromImage(sitk.ReadImage(SEG1_MD_addr + i_img))
            arr2_mask = sitk.GetArrayFromImage(sitk.ReadImage(SEG2_MD_addr + i_img))

            # get region props of segmentations
            regions = regionprops(label(arr_mask))
            regions_whole0 = regionprops(label(arr0_mask))
            regions_whole1 = regionprops(label(arr1_mask))
            regions_whole2 = regionprops(label(arr2_mask))

            xml_name = xml_addr + str(i_img[:-4]) + '.xml'

            savedStdout = sys.stdout
            f = open(xml_name, "w")
            sys.stdout = f

            node_root = Element('annotation')
            node_folder = SubElement(node_root, 'folder')
            node_folder.text = 'BRATS'
            node_filename = SubElement(node_root, 'filename')
            node_filename.text = i_img

            height = np.size(arr_img, 0)
            width = np.size(arr_img, 1)
            node_size = SubElement(node_root, 'size')
            node_height = SubElement(node_size, 'height')
            node_height.text = str(height)
            node_width = SubElement(node_size, 'width')
            node_width.text = str(width)
            node_depth = SubElement(node_size, 'depth')
            node_depth.text = str(3)


            # recording different kind of lesions
            for i_region in range(len(regions)):

                pos = regions[i_region].coords[0]
                label_value = arr_mask[int(pos[0]), int(pos[1])]

                if label_value == 2:
                    name = 'tumour'
                elif label_value == 1:
                    name = 'core'
                elif label_value == 4:
                    name = 'necrosis'
                else:
                    continue

                node_object = SubElement(node_root, 'object')
                node_objname = SubElement(node_object, 'name')
                node_objname.text = name
                node_difficult = SubElement(node_object, 'difficult')
                node_difficult.text = '0'

                nodebndbox = SubElement(node_object, 'bndbox')

                xmin = np.min(regions[i_region].coords[:, 1])
                xmax = np.max(regions[i_region].coords[:, 1])

                ymin = np.min(regions[i_region].coords[:, 0])
                ymax = np.max(regions[i_region].coords[:, 0])

                node_xmin = SubElement(nodebndbox, 'xmin')
                node_xmin.text = str(xmin)
                node_xmax = SubElement(nodebndbox, 'xmax')
                node_xmax.text = str(xmax)
                node_ymin = SubElement(nodebndbox, 'ymin')
                node_ymin.text = str(ymin)
                node_ymax = SubElement(nodebndbox, 'ymax')
                node_ymax.text = str(ymax)


            # recording combined lesions
            for i_region in range(len(regions_whole0)):

                pos = regions[i_region].coords[0]
                label_value = arr_mask[int(pos[0]), int(pos[1])]

                if label_value == 1:
                    name = 'whole0'
                else:
                    continue

                node_object = SubElement(node_root, 'object')
                node_objname = SubElement(node_object, 'name')
                node_objname.text = name
                node_difficult = SubElement(node_object, 'difficult')
                node_difficult.text = '0'

                nodebndbox = SubElement(node_object, 'bndbox')

                xmin = np.min(regions[i_region].coords[:, 1])
                xmax = np.max(regions[i_region].coords[:, 1])

                ymin = np.min(regions[i_region].coords[:, 0])
                ymax = np.max(regions[i_region].coords[:, 0])

                node_xmin = SubElement(nodebndbox, 'xmin')
                node_xmin.text = str(xmin)
                node_xmax = SubElement(nodebndbox, 'xmax')
                node_xmax.text = str(xmax)
                node_ymin = SubElement(nodebndbox, 'ymin')
                node_ymin.text = str(ymin)
                node_ymax = SubElement(nodebndbox, 'ymax')
                node_ymax.text = str(ymax)

            # recording red and yellow lesions
            for i_region in range(len(regions_whole1)):

                pos = regions[i_region].coords[0]
                label_value = arr_mask[int(pos[0]), int(pos[1])]

                if label_value == 1:
                    name = 'whole1'
                else:
                    continue

                node_object = SubElement(node_root, 'object')
                node_objname = SubElement(node_object, 'name')
                node_objname.text = name
                node_difficult = SubElement(node_object, 'difficult')
                node_difficult.text = '0'

                nodebndbox = SubElement(node_object, 'bndbox')

                xmin = np.min(regions[i_region].coords[:, 1])
                xmax = np.max(regions[i_region].coords[:, 1])

                ymin = np.min(regions[i_region].coords[:, 0])
                ymax = np.max(regions[i_region].coords[:, 0])

                node_xmin = SubElement(nodebndbox, 'xmin')
                node_xmin.text = str(xmin)
                node_xmax = SubElement(nodebndbox, 'xmax')
                node_xmax.text = str(xmax)
                node_ymin = SubElement(nodebndbox, 'ymin')
                node_ymin.text = str(ymin)
                node_ymax = SubElement(nodebndbox, 'ymax')
                node_ymax.text = str(ymax)

            # recording red lesions
            for i_region in range(len(regions_whole2)):

                pos = regions[i_region].coords[0]
                label_value = arr_mask[int(pos[0]), int(pos[1])]

                if label_value == 1:
                    name = 'whole2'
                else:
                    continue

                node_object = SubElement(node_root, 'object')
                node_objname = SubElement(node_object, 'name')
                node_objname.text = name
                node_difficult = SubElement(node_object, 'difficult')
                node_difficult.text = '0'

                nodebndbox = SubElement(node_object, 'bndbox')

                xmin = np.min(regions[i_region].coords[:, 1])
                xmax = np.max(regions[i_region].coords[:, 1])

                ymin = np.min(regions[i_region].coords[:, 0])
                ymax = np.max(regions[i_region].coords[:, 0])

                node_xmin = SubElement(nodebndbox, 'xmin')
                node_xmin.text = str(xmin)
                node_xmax = SubElement(nodebndbox, 'xmax')
                node_xmax.text = str(xmax)
                node_ymin = SubElement(nodebndbox, 'ymin')
                node_ymin.text = str(ymin)
                node_ymax = SubElement(nodebndbox, 'ymax')
                node_ymax.text = str(ymax)


            xml = tostring(node_root, pretty_print=True).decode()
            dom = parseString(xml)
            print (xml)
            sys.stdout = savedStdout
    return


def spliting(name_list, addr):

    Modality = ('flair', 't1', 't1ce', 't2')

    for i_mol in Modality:

        # NumofSubject = 49
        split_addr = addr + i_mol + '/ImageSets/Main/'
        annotation_addr = addr + i_mol + '/Annotations/'

        # We ensure i_mol folder exists
        if not os.path.exists(split_addr):
            os.mkdir(addr + i_mol + '/ImageSets/')
            os.mkdir(split_addr)

        cv = KFold(10)


        # filename tumour
        fn_tumour_train = []
        fn_tumour_trainval = []
        fn_tumour_val = []
        fn_tumour_test = []

        # filename core
        fn_core_train = []
        fn_core_trainval = []
        fn_core_val = []
        fn_core_test = []

        # filename necrosis
        fn_necrosis_train = []
        fn_necrosis_trainval = []
        fn_necrosis_val = []
        fn_necrosis_test = []

        # filename whole0
        fn_whole0_train = []
        fn_whole0_trainval = []
        fn_whole0_val = []
        fn_whole0_test = []

        # filename whole1
        fn_whole1_train = []
        fn_whole1_trainval = []
        fn_whole1_val = []
        fn_whole1_test = []

        # filename whole2
        fn_whole2_train = []
        fn_whole2_trainval = []
        fn_whole2_val = []
        fn_whole2_test = []

        #
        fn_train = []
        fn_test = []
        fn_trainval = []
        fn_val = []

        FD = 0
        in_FD = 0

        for trainval, test in cv.split(name_list):


            # filename tumour
            fn_tumour_trainval.append(split_addr + 'cv' + str(FD) + '_tumour_trainval.txt')
            fn_tumour_test.append(split_addr + 'cv' + str(FD) + '_tumour_test.txt')
            fn_tumour_train.append(split_addr + 'cv' + str(FD) + '_tumour_train.txt')
            fn_tumour_val.append(split_addr + 'cv' + str(FD) + '_tumour_val.txt')

            # filename core
            fn_core_trainval.append(split_addr + 'cv' + str(FD) + '_core_trainval.txt')
            fn_core_test.append(split_addr + 'cv' + str(FD) + '_core_test.txt')
            fn_core_train.append(split_addr + 'cv' + str(FD) + '_core_train.txt')
            fn_core_val.append(split_addr + 'cv' + str(FD) + '_core_val.txt')

            # filename necrosis
            fn_necrosis_trainval.append(split_addr + 'cv' + str(FD) + '_necrosis_trainval.txt')
            fn_necrosis_test.append(split_addr + 'cv' + str(FD) + '_necrosis_test.txt')
            fn_necrosis_train.append(split_addr + 'cv' + str(FD) + '_necrosis_train.txt')
            fn_necrosis_val.append(split_addr + 'cv' + str(FD) + '_necrosis_val.txt')

            # filename whole0
            fn_whole0_trainval.append(split_addr + 'cv' + str(FD) + '_whole0_trainval.txt')
            fn_whole0_test.append(split_addr + 'cv' + str(FD) + '_whole0_test.txt')
            fn_whole0_train.append(split_addr + 'cv' + str(FD) + '_whole0_train.txt')
            fn_whole0_val.append(split_addr + 'cv' + str(FD) + '_whole0_val.txt')

            # filename whole1
            fn_whole1_trainval.append(split_addr + 'cv' + str(FD) + '_whole1_trainval.txt')
            fn_whole1_test.append(split_addr + 'cv' + str(FD) + '_whole1_test.txt')
            fn_whole1_train.append(split_addr + 'cv' + str(FD) + '_whole1_train.txt')
            fn_whole1_val.append(split_addr + 'cv' + str(FD) + '_whole1_val.txt')

            # filename whole2
            fn_whole2_trainval.append(split_addr + 'cv' + str(FD) + '_whole2_trainval.txt')
            fn_whole2_test.append(split_addr + 'cv' + str(FD) + '_whole2_test.txt')
            fn_whole2_train.append(split_addr + 'cv' + str(FD) + '_whole2_train.txt')
            fn_whole2_val.append(split_addr + 'cv' + str(FD) + '_whole2_val.txt')

            #
            fn_trainval.append(split_addr + 'cv' + str(FD) + '_trainval.txt')
            fn_test.append(split_addr + 'cv' + str(FD) + '_test.txt')
            fn_train.append(split_addr + 'cv' + str(FD) + '_train.txt')
            fn_val.append(split_addr + 'cv' + str(FD) + '_val.txt')


            # writing files

            trainval_list = np.array(name_list)[trainval]
            test_list = np.array(name_list)[test]

            # writing files for trainval set
            for i_train_val_sub in trainval_list:
                fn_trval_sub, slt_trval_sub = select_from_start(annotation_addr, i_train_val_sub)
                #print(i_train_val_sub)

                for i_slice_fn in fn_trval_sub:

                    i_slice_addr_fn = annotation_addr + i_slice_fn
                    #print(i_slice_addr_fn)
                    tree = et.parse(i_slice_addr_fn)
                    root = tree.getroot()
                    filename = root.find('filename').text
                    #print(filename)

                    have_tumour = -1
                    have_core = -1
                    have_necrosis = -1
                    have_whole0 = -1
                    have_whole1 = -1
                    have_whole2 = -1


                    for obj in root.findall('object'):
                        name = obj.find('name').text
                        if name == 'tumour':
                            have_tumour = 1
                        if name == 'core':
                            have_core = 1
                        if name == 'necrosis':
                            have_necrosis = 1
                        if name == 'whole0':
                            have_whole0 = 1
                        if name == 'whole1':
                            have_whole1 = 1
                        if name == 'whole2':
                            have_whole2 = 1


                    with open(fn_trainval[FD], 'a') as f:
                        f.write(i_slice_fn[:-4] + ' ' + '\n')
                    with open(fn_tumour_trainval[FD], 'a') as f:
                        f.write(i_slice_fn[:-4] + ' ' + str(have_tumour) + '\n')
                    with open(fn_core_trainval[FD], 'a') as f:
                        f.write(i_slice_fn[:-4] + ' ' + str(have_core) + '\n')
                    with open(fn_necrosis_trainval[FD], 'a') as f:
                        f.write(i_slice_fn[:-4] + ' ' + str(have_necrosis) + '\n')
                    with open(fn_whole0_trainval[FD], 'a') as f:
                        f.write(i_slice_fn[:-4] + ' ' + str(have_whole0) + '\n')
                    with open(fn_whole1_trainval[FD], 'a') as f:
                        f.write(i_slice_fn[:-4] + ' ' + str(have_whole1) + '\n')
                    with open(fn_whole2_trainval[FD], 'a') as f:
                        f.write(i_slice_fn[:-4] + ' ' + str(have_whole2) + '\n')


            # writing files for testing set
            for i_test_sub in test_list:
                fn_test_sub, slt_test_sub = select_from_start(annotation_addr, i_test_sub)

                for i_slice_fn in fn_test_sub:

                    i_slice_addr_fn = annotation_addr + i_slice_fn
                    tree = et.parse(i_slice_addr_fn)
                    root = tree.getroot()
                    filename = root.find('filename').text
                    #print(filename)

                    have_tumour = -1
                    have_core = -1
                    have_necrosis = -1
                    have_whole0 = -1
                    have_whole1 = -1
                    have_whole2 = -1

                    for obj in root.findall('object'):
                        name = obj.find('name').text
                        if name == 'tumour':
                            have_tumour = 1
                        if name == 'core':
                            have_core = 1
                        if name == 'necrosis':
                            have_necrosis = 1
                        if name == 'whole0':
                            have_whole0 = 1
                        if name == 'whole1':
                            have_whole1 = 1
                        if name == 'whole2':
                            have_whole2 = 1

                    with open(fn_test[FD], 'a') as f:
                        f.write(i_slice_fn[:-4] + ' ' + '\n')
                    with open(fn_tumour_test[FD], 'a') as f:
                        f.write(i_slice_fn[:-4] + ' ' + str(have_tumour) + '\n')
                    with open(fn_core_test[FD], 'a') as f:
                        f.write(i_slice_fn[:-4] + ' ' + str(have_core) + '\n')
                    with open(fn_necrosis_test[FD], 'a') as f:
                        f.write(i_slice_fn[:-4] + ' ' + str(have_necrosis) + '\n')
                    with open(fn_whole0_test[FD], 'a') as f:
                        f.write(i_slice_fn[:-4] + ' ' + str(have_whole0) + '\n')
                    with open(fn_whole1_test[FD], 'a') as f:
                        f.write(i_slice_fn[:-4] + ' ' + str(have_whole1) + '\n')
                    with open(fn_whole2_test[FD], 'a') as f:
                        f.write(i_slice_fn[:-4] + ' ' + str(have_whole2) + '\n')


            for A in range(1):
                train_and_val = train_test_split(range(len(trainval)), test_size=0.1)
                train_list = np.array(trainval_list)[train_and_val[0]]
                val_list = np.array(trainval_list)[train_and_val[1]]

                # writing files for training set

                for i_train_sub in train_list:
                    fn_tr_sub, slt_tr_sub = select_from_start(annotation_addr, i_train_sub)
                    #print(fn_tr_sub)

                    for i_slice_fn in fn_tr_sub:

                        i_slice_addr_fn = annotation_addr + i_slice_fn
                        tree = et.parse(i_slice_addr_fn)
                        root = tree.getroot()
                        filename = root.find('filename').text
                        #print(filename)

                        have_tumour = -1
                        have_core = -1
                        have_necrosis = -1
                        have_whole0 = -1
                        have_whole1 = -1
                        have_whole2 = -1

                        for obj in root.findall('object'):
                            name = obj.find('name').text
                            if name == 'tumour':
                                have_tumour = 1
                            if name == 'core':
                                have_core = 1
                            if name == 'necrosis':
                                have_necrosis = 1
                            if name == 'whole0':
                                have_whole0 = 1
                            if name == 'whole1':
                                have_whole1 = 1
                            if name == 'whole2':
                                have_whole2 = 1

                        with open(fn_train[FD], 'a') as f:
                            f.write(i_slice_fn[:-4] + ' ' + '\n')
                        with open(fn_tumour_train[FD], 'a') as f:
                            f.write(i_slice_fn[:-4] + ' ' + str(have_tumour) + '\n')
                        with open(fn_core_train[FD], 'a') as f:
                            f.write(i_slice_fn[:-4] + ' ' + str(have_core) + '\n')
                        with open(fn_necrosis_train[FD], 'a') as f:
                            f.write(i_slice_fn[:-4] + ' ' + str(have_necrosis) + '\n')
                        with open(fn_whole0_train[FD], 'a') as f:
                            f.write(i_slice_fn[:-4] + ' ' + str(have_whole0) + '\n')
                        with open(fn_whole1_train[FD], 'a') as f:
                            f.write(i_slice_fn[:-4] + ' ' + str(have_whole1) + '\n')
                        with open(fn_whole2_train[FD], 'a') as f:
                            f.write(i_slice_fn[:-4] + ' ' + str(have_whole2) + '\n')

                # writing files for val set

                for i_val_sub in val_list:
                    fn_val_sub, slt_val_sub = select_from_start(annotation_addr, i_val_sub)

                    for i_slice_fn in fn_val_sub:

                        i_slice_addr_fn = annotation_addr + i_slice_fn
                        tree = et.parse(i_slice_addr_fn)
                        root = tree.getroot()
                        filename = root.find('filename').text
                        #print(filename)

                        have_tumour = -1
                        have_core = -1
                        have_necrosis = -1
                        have_whole0 = -1
                        have_whole1 = -1
                        have_whole2 = -1

                        for obj in root.findall('object'):
                            name = obj.find('name').text
                            if name == 'tumour':
                                have_tumour = 1
                            if name == 'core':
                                have_core = 1
                            if name == 'necrosis':
                                have_necrosis = 1
                            if name == 'whole0':
                                have_whole0 = 1
                            if name == 'whole1':
                                have_whole1 = 1
                            if name == 'whole2':
                                have_whole2 = 1

                        with open(fn_val[FD], 'a') as f:
                            f.write(i_slice_fn[:-4] + ' ' + '\n')
                        with open(fn_tumour_val[FD], 'a') as f:
                            f.write(i_slice_fn[:-4] + ' ' + str(have_tumour) + '\n')
                        with open(fn_core_val[FD], 'a') as f:
                            f.write(i_slice_fn[:-4] + ' ' + str(have_core) + '\n')
                        with open(fn_necrosis_val[FD], 'a') as f:
                            f.write(i_slice_fn[:-4] + ' ' + str(have_necrosis) + '\n')
                        with open(fn_whole0_val[FD], 'a') as f:
                            f.write(i_slice_fn[:-4] + ' ' + str(have_whole0) + '\n')
                        with open(fn_whole1_val[FD], 'a') as f:
                            f.write(i_slice_fn[:-4] + ' ' + str(have_whole1) + '\n')
                        with open(fn_whole2_val[FD], 'a') as f:
                            f.write(i_slice_fn[:-4] + ' ' + str(have_whole2) + '\n')
            FD += 1


    return

def dilate_recta(mask_addr, sav_addr):

    sub = os.listdir(mask_addr)
    for i_sub in sub:
        name = mask_addr + i_sub
        sav_name = sav_addr + i_sub[:-12] + '_dilated.nii.gz'
        im_mask = sitk.ReadImage(name)
        im_arr = sitk.GetArrayFromImage(im_mask)
        arr = np.zeros((np.size(im_arr, 0), np.size(im_arr, 1), np.size(im_arr, 2)))
        arr[np.where(im_arr == 1)] = 1
        im = sitk.GetImageFromArray(arr.astype('uint8'))
        im.CopyInformation(im_mask)

        dilator = sitk.BinaryDilateImageFilter()
        dilator.SetKernelRadius(100)
        dilator.SetKernelType(sitk.sitkBall)
        dilated = dilator.Execute(im)
        dilated.CopyInformation(im_mask)
        sitk.WriteImage(dilated, sav_name)

    return


def cut_id(name_id):

    pos = name_id.find('-')
    name = name_id[0:pos]
    id = name_id[pos+1:]

    return name, id


if __name__ == '__main__':

    # dilating recta
    addr = '/media/jyn/A0CAA102CAA0D5B6/RL_Detection/MICCAI_BraTS_2018_Data_Training/ALL/'
    #dilate_recta(addr, dilated_mask_addr)

    # cropping eccording to dilation
    db_addr = '/media/jyn/A0CAA102CAA0D5B6/BRATS/ALL/'
    name_list = ToPNG(addr, db_addr)

    # generatinf xml files
    png_img_addr = '/media/jyn/A0CAA102CAA0D5B6/BRATS/ALL/JPEGImages/'
    png_mask_addr = '/media/jyn/A0CAA102CAA0D5B6/BRATS/ALL/SegmentationClass/'
    generating_xmls(db_addr)

    # spliting training and testing data, writ to files
    #addr = '../data/BRATS/LGG/'
    spliting(name_list, db_addr)
