import json
import os
import numpy as np
import cv2
from tqdm import tqdm
import math
import shutil
import time
import random
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.polys import PolygonsOnImage
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count, Manager


def aug(image, poly, seq):
        images = np.expand_dims(image, axis=0)
        polygons = [ia.Polygon(poly)]

        new_polygons = PolygonsOnImage(polygons, shape = image.shape)
        images_aug, polygons_aug = seq(images = images, polygons = new_polygons)
        images_aug = np.squeeze(images_aug)       

        pt = polygons_aug.to_xy_array().astype(int)
        pt[pt < 0] = 0
        w=images_aug.shape[1]
        h=images_aug.shape[0]

        pt[:,1:2][pt[:,1:2]>w]=w
        pt[:,0:1][pt[:,0:1]>h]=h

        return images_aug,pt  


def CopyPaste(pool, back_path, img_path, json_path, seq, choose, copy_num):
    json_filenames=[json_filename for json_filename in os.listdir(json_path) if json_filename.endswith('.json')]
    shapes_all=[]     # 存放所有的物体轮廓标签点集
    croped_all=[]     # 存放所有的裁剪物体

    for daluan in range(copy_num):                                                           
        random.shuffle(json_filenames)           
        for json_filename in tqdm(json_filenames):
            img_filename = json_filename.replace('.json', '.png')      # 单个图像的文件名
            jsonfile_path = os.path.join(json_path, json_filename)  # 完整的json路径
            imgfile_path = os.path.join(img_path, img_filename)    # 完整的图像路径
            img = cv2.imread(imgfile_path)     # 打开单个图像
            img_height = img.shape[0]
            img_width = img.shape[1]
            
            j = open(jsonfile_path).read()      # json文件读入成字符串格式
            jj = json.loads(j)   
            for i in range(len(jj['shapes'])): 
                if jj["shapes"][i]["label"] in choose:  
                    pts=jj["shapes"][i]["points"]
                    pts=np.array(pts).astype(int)
                    pts[pts<0]=0             
                    rect = cv2.boundingRect(pts)

                    x, y, w, h = rect
                    
                    ## 这是图像边界物体去除的
                    if (x<=3 or y<=3 or x+w>=(img_width-3) or y+h>=(img_height-3)) and (w*h <10000):
                        continue
                    
                    croped = img[y:h+y, x:w+x].copy()                                                         
                    dr = int((h**2+w**2)**0.5)
                    dx = int(0.5*dr-0.5*w)
                    dy = int(0.5*dr-0.5*h)
                    dw = int(dr-w)
                    dh = int(dr-h)                                              
                    x = x - dx
                    y = y - dy 
                    # wx = (img_width-1) if w+x+dw+1>img_width else w+x+dw
                    # hy = (img_height-1) if h+y+dh+1>img_height else h+y+dh
                    pt = pts - [x, y]                                                    # 物体的标注点集points[numpy类型]，将物体平移到整张图像的左上角
                    black = [0, 0, 0]
                    croped = cv2.copyMakeBorder(croped, dy, dy, dx, dx, cv2.BORDER_CONSTANT, value=black)

                    # new_path=root+'/paste/' 
                    # dst=mask1-mask
                    # crop_name=new_path+'beforecrop_'+str(i)+'_'+img_filename
                    # cv2.imwrite(crop_name, croped)
                    
                    croped,pt = aug(croped,pt.tolist(),seq)         ##数据增强 
                    # cv2.drawContours(croped, [pt], -1, 255, 3)
                    # aug_name=new_path+'aug_'+str(i)+'_'+img_filename
                    ## 
                    # cv2.imwrite(aug_name, croped)
                    # pt = pt - pt.min(axis=0)
                    
                    if len(pt) < 4:
                        continue

                    aa={}
                    aa.update({'label': jj["shapes"][i]["label"]})
                    aa.update({"points":pt})        #此时的点是np格式 ，最终需要转换为列表格式
                    aa.update({"group_id":None})
                    aa.update({"shape_type": "polygon"})
                    aa.update({"flags": {}})

                    shapes_all.append(aa)
                    croped_all.append(croped)
                    

    
    randnum = random.randint(0,10000)
    random.seed(randnum)
    random.shuffle(shapes_all)
    random.seed(randnum)
    random.shuffle(croped_all)
                
    ###### 粘贴  ########  

    ### 存放背景的文件  ###                                                     
    back_imgs = [back_img for back_img in os.listdir(back_path) if back_img.endswith('.png')]
    ### 粘贴的文件夹，没有会自动生成  ###
    paste_root = os.path.dirname(back_path)
    new_path = paste_root + '/tmp/'       # 要修改的paste路径
    # new_path= "H:/detectron2_data/munal_4_copy56/"
    if not os.path.exists(new_path):                  
        os.makedirs(new_path)  

                
    cout = 0
    m = 0
    while(cout < len(shapes_all)):

        ###  随机选择一张图像作为背景  ###
        back_ImgChoice = random.randint(0, len(back_imgs) - 1)
        back_imgpath = back_path+'/' + back_imgs[back_ImgChoice]
        back_img = cv2.imread(back_imgpath)

        ###  获取背景图的尺寸  ###
        back_height = back_img.shape[0]
        back_width = back_img.shape[1]

        ###  新建json文件名和图像文件名  ###
        t = time.strftime('2022%m%d_%H%M%S', time.localtime(time.time())) 
        New_jsonfile = t + str(pool) + str(m) + '.json'  # t+str(pool)+str(m)+'.json'
        New_imgfile = New_jsonfile.replace('.json', '.png')
        New_jsonpath = new_path + New_jsonfile             

        ###  创建json文件  ###
        json_background = {
            "version": "4.5.6",
            "flags": {},
            "shapes": [],
            "imagePath": New_imgfile,
            "imageData": None,
            "imageHeight": back_height,
            "imageWidth": back_width
        }
        ###  创建分布  ###
        per=random.randint(3, 5)
        n_list=random.sample(range(0,per**2),random.randint(per*2,per**2))
        if (cout+len(n_list)) > len(shapes_all):                                 # 最后一张图像要特殊处理
            per=5
            n_list=random.sample(range(0,per**2),len(shapes_all)-cout)
        pixel_X=int(back_width/per)
        pixel_Y=int(back_height/per)
        def f(x):
            return [pixel_X*(x%per),pixel_Y*int(x/per)]
        center_all=list(map(f,n_list))

        ###  生成一张值为1的图mask_img  ###
        # mask_img=np.ones((back_height,back_width),dtype=np.uint8) 
        mask_img=np.ones(back_img.shape[:3], np.uint8)
        
        mask_dict=[]
        zuobiao=[]
        for n in range(len(center_all)):

            aa=shapes_all[cout+n]
            pt=aa["points"]
            croped=croped_all[cout+n]

            ###  获取裁剪物体的尺寸  ###
            rows=int(croped.shape[0])      #裁剪的物体的高h ,对应y方向坐标
            cols=int(croped.shape[1])      #裁剪的物体的宽w ,对应x方向坐标

            ###  c0,c1 表示粘到图像上位置的参考点(x,y)  ###
            c0,c1=center_all[n][0],center_all[n][1]

            ###  对应的label用mask数字标记   ###
            mask_num=10+n_list[n]*2 
            mask_dict.append([mask_num,aa])

            ###  存储坐标信息，便于后续的裁剪  ###
            zuobiao.append([rows,cols,c1,c0])

            ###  处理边界问题   ###
            gap_rows,gap_cols=0,0
            if c1+rows>back_height:
                gap_rows=c1+rows-back_height
            if c0+cols>back_width:
                gap_cols=c0+cols-back_width

            ###  生成一个背景是0，物体是1的mask
            mask0 = np.zeros(croped.shape[:3], np.uint8)    
            cv2.drawContours(mask0, [pt], -1, (1, 1, 1), -1, cv2.LINE_AA)

            ### 生成一个背景是1，物体是0的mask
            mask1= np.ones(croped.shape[:3], np.uint8)
            cv2.drawContours(mask1, [pt], -1, (0, 0, 0), -1, cv2.LINE_AA)

            ###  裁剪下来的物体背景变成0  ###
            croped0=croped*mask0  
            ###  背景图中要粘贴的物体变成0  ###
            back_img[c1:c1+rows-gap_rows,c0:c0+cols-gap_cols]=back_img[c1:c1+rows-gap_rows,c0:c0+cols-gap_cols]*mask1[:rows-gap_rows,:cols-gap_cols]  

            ###  把物体粘到背景中去 ###
            back_img[c1:c1+rows-gap_rows,c0:c0+cols-gap_cols]=back_img[c1:c1+rows-gap_rows,c0:c0+cols-gap_cols]+croped0[:rows-gap_rows,:cols-gap_cols]


            ###  生成一个背景是0，物体是mask_num的mask_0  ###
            mask_0=mask0.copy()
            mask_0[mask_0==1]=mask_num

            ###  将mask_img中要粘贴的物体区域变成0  ###
            mask_img[c1:c1+rows-gap_rows,c0:c0+cols-gap_cols]=mask_img[c1:c1+rows-gap_rows,c0:c0+cols-gap_cols]*mask1[:rows-gap_rows,:cols-gap_cols]

            ### 将mask_0粘到mask_img中去  ###
            mask_img[c1:c1+rows-gap_rows,c0:c0+cols-gap_cols]=mask_img[c1:c1+rows-gap_rows,c0:c0+cols-gap_cols]+mask_0[:rows-gap_rows,:cols-gap_cols]


        for mm,item in enumerate(mask_dict):

            ###  取出坐标进行裁剪
            h,w,y,x=zuobiao[mm]
            mask_cro = mask_img[y:h+y, x:w+x].copy()
            

            ###  值为mask_num的物体像素值置为0  ###
            _mask_cro=mask_cro.copy()
            _mask_cro[_mask_cro==1]=0
            _mask_cro[_mask_cro==item[0]]=0

            pt=item[1]["points"]
            if type(pt) != np.ndarray:
                pt=np.array(pt)
            pt=pt+[x,y]
            xx,yy,ww,hh = cv2.boundingRect(pt)

            if np.all(_mask_cro==0) and (xx>0 and yy>0 and xx+ww<(back_width) and yy+hh<(back_height)):
                # print('true')
                
                pt[(pt-[back_width,10000])>=0]=back_width-1
                pt[(pt-[10000,back_height])>=0]=back_height-1
    
                if ww <20 or hh<20:
                    continue
                item[1].update({"points":pt.tolist()})
                json_background["shapes"].append(item[1])
            else:
                mask_cro[mask_cro!=item[0]]=0

                ###  findContours只支持CV_8UC1的图像  ###
                mask_cro = mask_cro[:, :, 0]
                ###  RETR_EXTERNAL只检测外轮廓，CHAIN_APPROX_TC89_KCOS点集尽可能少的逼近方式  ###
                contours, hierarchy = cv2.findContours(mask_cro,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_TC89_KCOS) 
            
                for contour in contours:
                    aaa=item[1].copy()
                    if cv2.contourArea(contour)>800:
                        # 由于某种原因，cv2.findContours将每个轮廓作为一个具有一个冗余维度的3D NumPy数组返回，所以要去除冗余
                        contour=np.squeeze(contour)    
                        pt=contour+[x,y]
                        pt[(pt-[back_width,10000])>=0]=back_width-1
                        pt[(pt-[10000,back_height])>=0]=back_height-1
                        
                        xx,yy,ww,hh = cv2.boundingRect(pt)
                        if ww <20 or hh<20:
                            continue
                        aaa.update({"points":pt.tolist()})
                        json_background["shapes"].append(aaa)  #把物体的轮廓点添加进json文件
        cout=cout+len(center_all)    
        m+=1

        with open(New_jsonpath, 'w') as f:
            json.dump(json_background, f,indent=2)#indent=4缩进保存json文件
        # mask_imgs.append(mask_img)
        # cv2.imwrite(os.path.join(new_path,'mask_'+New_imgfile),mask_img)
        cv2.imwrite(new_path + New_imgfile, back_img)


if __name__ == '__main__':
    seq = iaa.Sometimes(0.833, 
    iaa.Sequential([iaa.Flipud(0.5),
    iaa.Fliplr(0.5), 
    iaa.Multiply((0.8, 1.3)),  #用  让一些图片变的更亮,一些图片变得更暗
    iaa.Sometimes(0.7, iaa.Affine(rotate=(-90, 90))),
    # iaa.Affine(
    # #缩放变换
    # scale={"x":(0.9,1.2),"y":(0.9,1.2)},
    # # # #平移变换
    # # # translate_percent={"x":(-0.2,0.2),"y":(-0.2,0.2)},
    # # # #旋转
    # # # rotate=(-45,45)
    # )
    ]))
    

    ###  各个文件夹的路径可以随意设置，生成的图像存放在paste文件夹中，默认保存在与背景文件夹相同的根目录中  ###
    root = 'D:/paper_dataset' 
                                             # [Acp_2386, Bcp_1130]
    back_path = root + '/background_dirty'   # ['background_clean','background_dirty']                                                    
    img_path = root + '/Bauto_302'   #          # ['auto_4','Bauto_302']  
    json_path = root + '/Bauto_302'   #         # ['auto_4','Bauto_302'] 
    choose = ['concrete', 'brick', 'rubber', 'wood']
    copy_num = 1  # 控制生成的图像数量  

    start = time.time()
        
    # CopyPaste(back_path,img_path,json_path,seq,choose,copy_num)
    pool_num = 18  # 控制生成的图像数量 [13, 18]
    p = Pool(pool_num)
    for i in range(pool_num):    
        # apply_async()  以异步的方式来添加进程 (并发执行)
        p.apply_async(CopyPaste, args=(i, back_path, img_path, json_path, seq, choose, copy_num))
    p.close()
    p.join() 
    end = time.time()
    print("总共用时{}秒".format((end - start)))


       