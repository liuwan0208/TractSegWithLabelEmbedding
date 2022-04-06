import numpy as np
import torch
import os
import os.path as osp
import cv2
import time
import scipy.misc as misc
from skimage import measure
# import pydensecrf.densecrf as dcrf

def visualize_result(imt, ant, pred, save_dir,view='Z',gap_num=8):

    if view == 'Z':
        vis_size = (imt.shape[0],256,256)
        imt = image_tensor_resize(imt,vis_size,method='B')
        ant = image_tensor_resize(ant,vis_size,method='N').astype('uint8')
        pred = image_tensor_resize(pred,vis_size,method='N').astype('uint8')
    elif view == 'Y':
        vis_size = (imt.shape[1],256,256)
        imt = image_tensor_resize(imt.transpose(2,0,1),vis_size,method='B')
        ant = image_tensor_resize(ant.transpose(2,0,1),vis_size,method='N').astype('uint8')
        pred = image_tensor_resize(pred.transpose(2,0,1),vis_size,method='N').astype('uint8')
    elif view == 'X':
        vis_size = (imt.shape[1],256,256)
        imt = image_tensor_resize(imt.transpose(1,0,2),vis_size,method='B')
        ant = image_tensor_resize(ant.transpose(1,0,2),vis_size,method='N').astype('uint8')
        pred = image_tensor_resize(pred.transpose(1,0,2),vis_size,method='N').astype('uint8')

    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    for idx in range(pred.shape[0]):
        gap = 1 if view == 'Z' else gap_num
        if idx%gap == 0:
            binary = pred[idx]
            image, contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
            temp_img1 = gray2rgbimage(imt[idx])
            color=(0,0,255)
            cv2.drawContours(temp_img1, contours, -1, color, 1)
            # Draw GT
            binary = ant[idx]
            image, contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
            temp_img2 = gray2rgbimage(imt[idx])
            cv2.drawContours(temp_img2, contours, -1, color, 1)
            cv2.imwrite(osp.join(save_dir, '%02d.png'%idx), np.hstack((temp_img1, temp_img2)).astype('uint8'))

def visualize_single_pred(imt, pred, save_dir = None):
    if save_dir is not None and not osp.exists(save_dir):
        os.makedirs(save_dir)
    vis_imt = []
    for idx in range(pred.shape[0]):
        binary = pred[idx]
        image, contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        temp_img = gray2rgbimage(imt[idx])
        color=(0,0,255)
        cv2.drawContours(temp_img, contours, -1, color, 1)
        temp_img = cv2.resize(temp_img, (512, 512), interpolation=cv2.INTER_LINEAR)
        vis_imt.append(temp_img)
        if(save_dir is not None):
            cv2.imwrite(osp.join(save_dir, '%02d.png'%idx), temp_img.astype('uint8'))
    return vis_imt

# def save_checkpoint(filename, state, is_best=False):
#     torch.save(state, filename)
#     if is_best:
#         shutil.copyfile(filename, 'model_best.pth.tar')

def eval_seg(predict, gt, forground = 255):
    connect_p = measure.label(predict)
    label = np.unique(connect_p)
    n_predict = 0
    n_precision = 0
    thres = 0
    for ii in range(1, label.shape[0]):
        if (connect_p == ii).sum()>thres:
            n_predict += 1
        if ((connect_p == ii)*gt).sum() > 0 and (connect_p == ii).sum()>thres:
            n_precision += 1

    connect_gt = measure.label(gt)
    label = np.unique(connect_gt)
    n_gt = label.shape[0]-1
    n_recall = 0
    for ii in range(1, n_gt+1):
        if((connect_gt == ii)*predict).sum() > thres*255:
            n_recall += 1
    score = 0
    count = 0
    assert(predict.shape == gt.shape)
    overlap = ((predict == forground)*(gt == forground)).sum()
    if(overlap > 0):
        dice = 2.0*overlap / ((predict == forground).sum() + (gt == forground).sum())
        jaccard = 1.0*overlap / ((predict == forground).sum() + (gt == forground).sum() - overlap)
        # recall = 1.0*n_recall / n_gt
        # precision = 1.0*n_precision / n_predict
        #recall = 1.0*overlap / (gt == forground).sum()
        #precision = 1.0*overlap / (predict == forground).sum()
    else:
        dice = 0.0
        jaccard = 0.0
    return dice, jaccard, n_recall, n_gt, n_precision, n_predict, 1.0*overlap/(gt == forground).sum()

def compute_dice_score(predict, gt, forground = 1):
    # score = 0
    # count = 0
    assert(predict.shape == gt.shape)
    #a=(predict==forground).sum()
    #b=(gt==forground).sum()
    #print("a,b=",a,b)
    overlap = 2.0 * ((predict == forground)*(gt == forground)).sum()
    #print("overlap=",overlap)
    if(overlap > 0):
        return overlap / ((predict == forground).sum() + (gt == forground).sum())
    else:
        return 0

# def postprocess_crf(res, imt):
#     pred_label = np.zeros(res.shape, dtype=np.uint8)
#     d, h, w = res.shape
#     for dix in range(d):
#         if((res[dix]>0.5).sum() == 0):
#             continue
#         crf = dcrf.DenseCRF2D(h, w, 2)  # width, height, nlabels
#         U = np.zeros((2, h, w), dtype = np.float32)
#         U[0] = 1 - res[dix]
#         U[1] = res[dix]
#         U = U.reshape((2, -1))
#         crf.setUnaryEnergy(-np.log(U+10**(-6)))
#
#         im = np.zeros((h, w, 3), dtype = np.uint8)
#         im[:, :, 0] = imt[dix]
#         im[:, :, 1] = imt[dix]
#         im[:, :, 2] = imt[dix]
#         crf.addPairwiseBilateral(sxy=8, srgb=32, rgbim=im, compat=10)
#
#         Q = crf.inference(5)
#         pred_label[dix] = np.argmax(Q, axis=0).reshape((h, w))
#     return pred_label.astype(np.uint8)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# Image Operation
def image_tensor_resize(img_tensor, size=(-1, -1, -1), method='B'):
    new_d, new_h, new_w = size
    d, h, w = img_tensor.shape
    new_shape = (new_d, new_h, new_w)
    if d == new_d and h == new_h and w == new_w:
        return img_tensor
    tmp_img_tensor = np.zeros([img_tensor.shape[0], new_shape[1], new_shape[2]], dtype=img_tensor.dtype)
    new_img_tensor = np.zeros(new_shape, dtype=img_tensor.dtype)
    for idx in range(img_tensor.shape[0]):
        if method == 'B':
            tmp_img_tensor[idx, :, :] = cv2.resize(img_tensor[idx, :, :], (new_shape[2], new_shape[1]), interpolation=cv2.INTER_CUBIC)
        elif method == 'L':
            tmp_img_tensor[idx, :, :] = cv2.resize(img_tensor[idx, :, :], (new_shape[2], new_shape[1]), interpolation=cv2.INTER_LINEAR)
        else:
            tmp_img_tensor[idx, :, :] = cv2.resize(img_tensor[idx, :, :], (new_shape[2], new_shape[1]), interpolation=cv2.INTER_NEAREST)

    for idx in range(new_shape[1]):
        if method == 'B':
            new_img_tensor[:, idx, :] = cv2.resize(tmp_img_tensor[:, idx, :], (new_shape[2], new_shape[0]),
                                                   interpolation=cv2.INTER_CUBIC)
        elif method == 'L':
            new_img_tensor[:, idx, :] = cv2.resize(tmp_img_tensor[:, idx, :], (new_shape[2], new_shape[0]),
                                                   interpolation=cv2.INTER_LINEAR)
        else:
            new_img_tensor[:, idx, :] = cv2.resize(tmp_img_tensor[:, idx, :], (new_shape[2], new_shape[0]),
                                                   interpolation=cv2.INTER_NEAREST)
    return new_img_tensor


def load_image_tensor(src_dir, img_ext='png', start_id=-1, end_id=-1):
    from glob import glob
    img_list = sorted(glob(src_dir + '/*.' + img_ext))
    if len(img_list) == 0:
        img_ext = 'png' if img_ext == 'jpg' else 'jpg'
        img_list = sorted(glob(src_dir + '/*.' + img_ext))
    if len(img_list) == 0:
        print ('Error:', src_dir)
        return None
    img_base = misc.imread(img_list[0])
    num_img = len(img_list)
    if start_id < 0:
        start_id = 0
    if end_id < 0:
        end_id = num_img
    end_id = min(end_id, num_img)

    img_tensor = np.zeros([end_id - start_id, img_base.shape[0], img_base.shape[1]], dtype=img_base.dtype)
    for idx in range(start_id, end_id):
        img_path = img_list[idx]
        img = misc.imread(img_path)
        if img.shape != img_base.shape:
            print ('Error: Not match.', img_base.shape, img.shape)
            print (img_path)
            break
        img_tensor[idx - start_id, :, :] = img[:, :]
    return img_tensor

def imsave_tensor(img_tensor, src_dir, direction=0, ext='png'):
    for idx in range(img_tensor.shape[direction]):
        if direction == 0:
            cv2.imwrite(src_dir + '/%03d.' % idx + ext, img_tensor[idx, :, :])
        elif direction == 1:
            cv2.imwrite(src_dir + '/%03d.' % idx + ext, img_tensor[:, idx, :])
        elif direction == 2:
            cv2.imwrite(src_dir + '/%03d.' % idx + ext, img_tensor[:, :, idx])

def gray2rgbimage(image):
    a,b = image.shape
    new_img = np.ones((a,b,3))
    new_img[:,:,0] = image.reshape((a,b)).astype('uint8')
    new_img[:,:,1] = image.reshape((a,b)).astype('uint8')
    new_img[:,:,2] = image.reshape((a,b)).astype('uint8')
    return new_img


def mkdir_safe(d):
    """
    Make Multi-Directories safety and thread friendly.
    """
    sub_dirs = d.split('/')
    cur_dir = ''
    max_check_times = 5
    sleep_seconds_per_check = 0.001
    for i in range(len(sub_dirs)):
        cur_dir += sub_dirs[i] + '/'
        for check_iter in range(max_check_times):
            if not os.path.exists(cur_dir):
                try:
                    os.mkdir(cur_dir)
                except Exception as e:
                    print ('[WARNING] ', str(e))
                    time.sleep(sleep_seconds_per_check)
                    continue
            else:
                break

###### 2D ############

def imt_2_3Dimage(imt):
    '''
    transfer a batch 2D image to a batch of 3D image 
    input: (20,512,512) tensor
    output:(20,3,512,512) tensor
    '''
    img_list = []
    for i in range(imt.size()[0]):
        if i == 0:
            tensor = torch.cat((imt[0],imt[0],imt[1]),0)
            tensor = tensor.view(3,imt.size()[1],imt.size()[2])
            img_list.append(tensor)
        elif i == imt.size()[0]-1:
            tensor = torch.cat((imt[-2],imt[-1],imt[-1]),0)
            tensor = tensor.view(3,imt.size()[1],imt.size()[2])
            img_list.append(tensor)
        else:
            tensor = torch.cat((imt[i-1],imt[i],imt[i+1]),0)
            tensor = tensor.view(3,imt.size()[1],imt.size()[2])
            img_list.append(tensor)

    image3D = torch.cat([img_list[i] for i in range(len(img_list))],0)
    image3D = image3D.view(imt.size()[0],3,imt.size()[1],imt.size()[2])

    return image3D

def rebuild_tensor_v2():
    try:
        torch._utils._rebuild_tensor_v2
    except AttributeError:
        def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
            tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
            tensor.requires_grad = requires_grad
            tensor._backward_hooks = backward_hooks
            return tensor
        torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2


