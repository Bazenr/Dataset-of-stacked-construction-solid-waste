import json
import os
import numpy as np
import cv2
from tqdm import tqdm
import time
import random
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.polys import PolygonsOnImage
from multiprocessing import Pool, cpu_count, Manager


def aug(image, poly, seq):
        images = np.expand_dims(image, axis=0)
        polygons = [ia.Polygon(poly)]

        new_polygons = PolygonsOnImage(polygons, shape=image.shape)
        images_aug, polygons_aug = seq(images=images, polygons=new_polygons)
        images_aug = np.squeeze(images_aug)       

        pt = polygons_aug.to_xy_array().astype(int)
        pt[pt < 0] = 0
        w = images_aug.shape[1]
        h = images_aug.shape[0]

        pt[:, 1:2][pt[:, 1:2] > w] = w
        pt[:, 0:1][pt[:, 0:1] > h] = h

        return images_aug, pt  


def RapidGenerate(pool, back_path, img_path, json_path, seq, choose, copy_num):
    json_filenames = [json_filename for json_filename in os.listdir(json_path) if json_filename.endswith('.json')]
    shapes_all = []  # The dict that stores all object contour labels
    croped_all = []  # The dict that stores all clipped objects

    for daluan in range(copy_num):                                                           
        random.shuffle(json_filenames)           
        for json_filename in tqdm(json_filenames):
            img_filename = json_filename.replace('.json', '.png')   # Filename
            jsonfile_path = os.path.join(json_path, json_filename)  # Full json path
            imgfile_path = os.path.join(img_path, img_filename)     # Full image path
            img = cv2.imread(imgfile_path)  # Open a single image
            img_height = img.shape[0]
            img_width = img.shape[1]

            j = open(jsonfile_path).read()  # Read json file into str format
            jj = json.loads(j)   
            for i in range(len(jj['shapes'])): 
                if jj["shapes"][i]["label"] in choose:  
                    pts = jj["shapes"][i]["points"]
                    pts = np.array(pts).astype(int)
                    pts[pts < 0] = 0             
                    rect = cv2.boundingRect(pts)

                    x, y, w, h = rect
                    
                    # # Remove objects located at the border of the image
                    if (x <= 3 or y <= 3 or x+w >= (img_width-3) or y+h >= (img_height-3)) and (w*h < 10000):
                        continue

                    croped = img[y:h+y, x:w+x].copy()                                                         
                    dr = int((h**2 + w**2)**0.5)
                    dx = int(0.5*dr - 0.5*w)
                    dy = int(0.5*dr - 0.5*h)
                    # dw = int(dr - w)
                    # dh = int(dr - h)                                              
                    x = x - dx
                    y = y - dy 
                    # wx = (img_width-1) if w+x+dw+1 > img_width else w+x+dw
                    # hy = (img_height-1) if h+y+dh+1 > img_height else h+y+dh

                    # The object's label point set points[numpy type],
                    # translate the object to the upper left corner of the entire image
                    pt = pts - [x, y]
                    black = [0, 0, 0]
                    croped = cv2.copyMakeBorder(croped, dy, dy, dx, dx, cv2.BORDER_CONSTANT, value=black)

                    croped, pt = aug(croped, pt.tolist(), seq)  # Data augmentation 

                    if len(pt) < 4:
                        continue

                    aa = {}
                    aa.update({'label': jj["shapes"][i]["label"]})
                    # The points at this time are in np format, and
                    # need to be converted to list format eventually
                    aa.update({"points":pt})
                    aa.update({"group_id":None})
                    aa.update({"shape_type": "polygon"})
                    aa.update({"flags": {}})

                    shapes_all.append(aa)
                    croped_all.append(croped)

    randnum = random.randint(0, 10000)
    random.seed(randnum)
    random.shuffle(shapes_all)
    random.seed(randnum)
    random.shuffle(croped_all)

    # ## Generation ## #

    # ## The file to store the background
    back_imgs = [back_img for back_img in os.listdir(back_path) if back_img.endswith('.png')]
    # ## Automatically generated folder
    generated_root = os.path.dirname(back_path)
    new_path = generated_root + '/tmp/'  # The generated path to modify

    if not os.path.exists(new_path):                  
        os.makedirs(new_path)  
 
    cout = 0
    m = 0
    while(cout < len(shapes_all)):

        # ## Randomly choose an image as background
        back_ImgChoice = random.randint(0, len(back_imgs) - 1)
        back_imgpath = back_path+'/' + back_imgs[back_ImgChoice]
        back_img = cv2.imread(back_imgpath)

        # ## Get the size of the background image
        back_height = back_img.shape[0]
        back_width = back_img.shape[1]

        # ## Create new json file name and image file name
        t = time.strftime('2022%m%d_%H%M%S', time.localtime(time.time())) 
        New_jsonfile = t + str(pool) + str(m) + '.json'
        New_imgfile = New_jsonfile.replace('.json', '.png')
        New_jsonpath = new_path + New_jsonfile             

        # ## Create json file
        json_background = {
            "version": "4.5.6",
            "flags": {},
            "shapes": [],
            "imagePath": New_imgfile,
            "imageData": None,
            "imageHeight": back_height,
            "imageWidth": back_width
        }

        # ## Create distribution plot
        per=random.randint(3, 5)
        n_list=random.sample(range(0, per**2),random.randint(per*2, per**2))
        # The last image needs special handling
        if (cout + len(n_list)) > len(shapes_all):
            per = 5
            n_list = random.sample(range(0, per**2), len(shapes_all) - cout)
        pixel_X = int(back_width / per)
        pixel_Y = int(back_height / per)
        def f(x):
            return [pixel_X*(x%per), pixel_Y*int(x/per)]
        center_all = list(map(f,n_list))

        # ## Generate a map mask_img with a value of 1
        # mask_img = np.ones((back_height, back_width), dtype=np.uint8) 
        mask_img = np.ones(back_img.shape[:3], np.uint8)
        
        mask_dict = []
        zuobiao = []
        for n in range(len(center_all)):

            aa = shapes_all[cout+n]
            pt = aa["points"]
            croped = croped_all[cout+n]

            # ## Get the size of the clipping object
            # The height h of the cropped object, which corresponds to
            # the coordinates in the y direction
            rows = int(croped.shape[0])
            # The width w of the cropped object, which corresponds to
            # the x-direction coordinates
            cols = int(croped.shape[1])  # 

            # ## c0,c1 represent the reference point (x,y) to stick to the position on the image
            c0,c1=center_all[n][0],center_all[n][1]

            # ## The corresponding label is marked with mask number
            mask_num = 10 + n_list[n] * 2 
            mask_dict.append([mask_num,aa])

            # ## Store coordinate information for subsequent cropping
            zuobiao.append([rows,cols,c1,c0])

            # ## Dealing with boundary issues
            gap_rows, gap_cols = 0, 0
            if c1+rows > back_height:
                gap_rows = c1 + rows - back_height
            if c0+cols > back_width:
                gap_cols = c0 + cols - back_width

            # ## Generate a mask with a background == 0 and object == 1
            mask0 = np.zeros(croped.shape[:3], np.uint8)    
            cv2.drawContours(mask0, [pt], -1, (1, 1, 1), -1, cv2.LINE_AA)

            # ## Generate a mask whose background == 1 and object == 0
            mask1 = np.ones(croped.shape[:3], np.uint8)
            cv2.drawContours(mask1, [pt], -1, (0, 0, 0), -1, cv2.LINE_AA)

            # ## The cropped object background is set to 0
            croped0 = croped * mask0  
            # ## The object to be generated in the background image is set to 0
            back_img[c1:c1+rows-gap_rows, c0:c0+cols-gap_cols] = \
                back_img[c1:c1+rows-gap_rows, c0:c0+cols-gap_cols] * mask1[:rows-gap_rows, :cols-gap_cols]  

            # ## Paste the object to the background
            back_img[c1:c1+rows-gap_rows, c0:c0+cols-gap_cols] = \
                back_img[c1:c1+rows-gap_rows, c0:c0+cols-gap_cols] + croped0[:rows-gap_rows, :cols-gap_cols]


            # ##  Generate a mask_0 with background == 0 and object == mask_num
            mask_0 = mask0.copy()
            mask_0[mask_0==1] = mask_num

            # ## Set the region of interest in "mask_img" to zero
            mask_img[c1:c1+rows-gap_rows, c0:c0+cols-gap_cols] = \
                mask_img[c1:c1+rows-gap_rows, c0:c0+cols-gap_cols] * mask1[:rows-gap_rows, :cols-gap_cols]

            # ## Paste "mask_0" onto "mask_img"
            mask_img[c1:c1+rows-gap_rows, c0:c0+cols-gap_cols] = \
                mask_img[c1:c1+rows-gap_rows, c0:c0+cols-gap_cols] + mask_0[:rows-gap_rows, :cols-gap_cols]


        for mm, item in enumerate(mask_dict):

            # ## Extract the coordinates and perform cropping
            h, w, y, x = zuobiao[mm]
            mask_cro = mask_img[y:h+y, x:w+x].copy()

            # ## Set the object's pixel-values with the value of "mask_num" to 0
            _mask_cro = mask_cro.copy()
            _mask_cro[_mask_cro==1] = 0
            _mask_cro[_mask_cro==item[0]] = 0

            pt = item[1]["points"]
            if type(pt) != np.ndarray:
                pt = np.array(pt)
            pt = pt + [x, y]
            xx, yy, ww, hh = cv2.boundingRect(pt)

            if np.all(_mask_cro==0) and (xx > 0
                                         and yy > 0
                                         and xx+ww < (back_width)
                                         and yy+hh < (back_height)):
                
                pt[(pt - [back_width, 10000]) >= 0] = back_width - 1
                pt[(pt - [10000, back_height]) >= 0] = back_height - 1
    
                if ww < 20 or hh < 20:
                    continue
                item[1].update({"points":pt.tolist()})
                json_background["shapes"].append(item[1])
            else:
                mask_cro[mask_cro!=item[0]] = 0

                # ## cv2.findContours only supports images with CV_8UC1 type
                mask_cro = mask_cro[:, :, 0]
                # ## RETR_EXTERNAL only detects external contours, while
                # ## CHAIN_APPROX_TC89_KCOS is the contour approximation method
                # ## that aims to reduce the number of points in the contour as
                # ## much as possible.
                contours, hierarchy = cv2.findContours(mask_cro, cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_TC89_KCOS) 
            
                for contour in contours:
                    aaa = item[1].copy()
                    if cv2.contourArea(contour) > 800:
                        # Due to some of the reasons, cv2.findContours returns each
                        # contour as a 3D NumPy array with an redundant dimension,
                        # which need to be removed.
                        contour = np.squeeze(contour)    
                        pt = contour + [x, y]
                        pt[(pt - [back_width, 10000]) >= 0] = back_width - 1
                        pt[(pt - [10000, back_height]) >= 0] = back_height - 1
                        
                        xx, yy, ww, hh = cv2.boundingRect(pt)
                        if ww < 20 or hh < 20:
                            continue
                        aaa.update({"points":pt.tolist()})
                        # Add the outline points of the object to the JSON file
                        json_background["shapes"].append(aaa)
        cout = cout + len(center_all)
        m += 1

        with open(New_jsonpath, 'w') as f:
            json.dump(json_background, f, indent=2)
        # mask_imgs.append(mask_img)
        # cv2.imwrite(os.path.join(new_path, 'mask_'+New_imgfile), mask_img)
        cv2.imwrite(new_path + New_imgfile, back_img)


if __name__ == '__main__':
    # ## Data augmentation
    seq = iaa.Sometimes(0.833, 
    iaa.Sequential([iaa.Flipud(0.5),
    iaa.Fliplr(0.5), 
    iaa.Multiply((0.8, 1.3)),  # Increase and decrease the brightness of the image
    iaa.Sometimes(0.7, iaa.Affine(rotate=(-90, 90))),
    # iaa.Affine(
    # scale = {"x":(0.9,1.2), "y":(0.9,1.2)},  # Scale transformation
    # # Position transformation
    # translate_percent = {"x":(-0.2,0.2), "y":(-0.2,0.2)},
    #  rotate = (-45, 45))  # Rotation
    ]))

    # ## The paths for each folder can be set arbitrarily, and the generated
    # ## images are stored in the "generated" folder, which is saved by default 
    # ## in the root directory that is the same as the background folder.
    root = 'D:/paper_dataset'               # dataset root
    back_path = root + '/background_dirty'  # background image path
    img_path = root + '/Bauto'              # source dataset image path
    json_path = root + '/Bauto'             # source dataset label file path
    choose = ['concrete', 'brick', 'rubber', 'wood']  # category
    copy_num = 1  # Num to control the quantity of generated images.

    start = time.time()

    # # Generate without threshold
    # RapidGenerate(back_path, img_path, json_path, seq, choose, copy_num)

    # # Generate with threshold
    pool_num = 18  # The number of parallel independent threads
    p = Pool(pool_num)
    for i in range(pool_num):    
        # apply_async() is used to add processes asynchronously (concurrently)
        p.apply_async(RapidGenerate, args=(i, back_path, img_path, json_path, seq, choose, copy_num))
    p.close()
    p.join() 
    end = time.time()
    print("Total time cost: {}s".format((end - start)))
