import torch.optim.lr_scheduler
from torch.utils.data import DataLoader
import torch.nn as nn
# from geoseg.datasets.vaihingen_dataset import *
from geoseg.datasets.potsdam_dataset import *
from tqdm import tqdm
import os
import cv2
import numpy as np
import random
from tools.metric import Evaluator
import multiprocessing.pool as mpp
import multiprocessing as mp

from geoseg.losses.dice import DiceLoss
from geoseg.models.DLGCNet import DLGCNet

ignore_index = len(CLASSES)
train_batch_size = 4
val_batch_size = 4
learning_rate = 1e-4###1e-4  ###2e-4 ####4e-3
weight_decay = 2e-5##2e-5#1e-5 85.67##5e-5 85.70###1e-4
num_epoch = 50  ###50


seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


#
def label2rgb(mask):
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [255, 255, 255]
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [255, 0, 0]
    mask_rgb[np.all(mask_convert == 2, axis=0)] = [255, 255, 0]
    mask_rgb[np.all(mask_convert == 3, axis=0)] = [0, 255, 0]
    mask_rgb[np.all(mask_convert == 4, axis=0)] = [0, 204, 255]
    mask_rgb[np.all(mask_convert == 5, axis=0)] = [0, 0, 255]
    return mask_rgb


def img_writer(inp):
    (mask, mask_id, rgb) = inp
    if rgb:
        mask_name_tif = mask_id + '.png'
        mask_tif = label2rgb(mask)
        cv2.imwrite(mask_name_tif, mask_tif)
    else:
        mask_png = mask.astype(np.uint8)
        mask_name_png = mask_id + '.png'
        cv2.imwrite(mask_name_png, mask_png)


train_dataset = PotsdamDataset(data_root='/train', mode='train',
                                 mosaic_ratio=0, transform=train_aug)

val_dataset = PotsdamDataset(data_root='/val', transform=val_aug)

test_dataset = PotsdamDataset(data_root='/test')

train_loader = DataLoader(dataset=train_dataset,
                            batch_size=train_batch_size,
                            num_workers=4,
                            pin_memory=True,
                            shuffle=True,
                            drop_last=True)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=val_batch_size,
                        num_workers=4,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False)

test_loader = DataLoader(dataset=test_dataset,
                        batch_size=val_batch_size,
                        num_workers=4,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DLGCNet().to(device)
if os.path.exists("/output.pth"):
    model.load_state_dict(torch.load("/output.pth"))
    print('weight model loaded!')


optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate ,weight_decay=weight_decay)#momentum=0.9,
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epoch,last_epoch=-1)

criterion_CE = nn.CrossEntropyLoss(ignore_index=ignore_index)
criterion_Dice = DiceLoss(ignore_index=ignore_index)
evaluator = Evaluator(num_class=len(CLASSES))
evaluator.reset()



def train():
    best_miou = 0.0
    for epoch in range(num_epoch):
        torch.cuda.empty_cache()
        model.train(mode=True)
        loop = tqdm(enumerate(train_loader), total=len(train_loader))
        ############ train  #######################################################################################
        for index, data_s in loop:
            optimizer.zero_grad()
            # ############ train  ################################################################################
            data_img = data_s['img'].to(device)
            input = data_img[:, 0:3, :, :]
            ndsm = data_img[:, 3, :, :]
            y = model(input,ndsm)#
            mask_s = data_s['gt_semantic_seg'].to(device)
            loss = criterion_CE(y, mask_s)+criterion_Dice(y, mask_s)
            loss.backward()
            optimizer.step()


            loop.set_description(f'Epoch [{epoch + 1}/{num_epoch}]')
            loop.set_postfix(loss_s=loss.item())
        torch.save(model.state_dict(), '/output_new.pth')
        scheduler.step()
        ###### val ################################################################################################
        model.eval()
        evaluator.reset()
        with torch.no_grad():
            for data_test in tqdm(val_loader):
                # raw_prediction NxCxHxW
                data_img = data_test['img'].to(device)
                input = data_img[:, 0:3, :, :]
                ndsm = data_img[:, 3, :, :]
                raw_predictions = model(input,ndsm)#
                masks_true = data_test['gt_semantic_seg']
                raw_predictions = nn.Softmax(dim=1)(raw_predictions)
                predictions = raw_predictions.argmax(dim=1)
                evaluator.add_batch(gt_image=masks_true.cpu().numpy(), pre_image=predictions.cpu().numpy())
            iou_per_class = evaluator.Intersection_over_Union()
            f1_per_class = evaluator.F1()
            OA = evaluator.OA()
            torch.cuda.empty_cache()
            for class_name, class_iou, class_f1 in zip(CLASSES, iou_per_class, f1_per_class):
                print('F1_{}:{}, IOU_{}:{}'.format(class_name, class_f1, class_name, class_iou))
            if np.nanmean(iou_per_class[:-1]) >= best_miou:
                best_miou = np.nanmean(iou_per_class[:-1])
                model.train()
                torch.save(model.state_dict(), '/output.pth')
            print('F1:{}, mIOU:{}, OA:{}'.format(np.nanmean(f1_per_class[:-1]), np.nanmean(iou_per_class[:-1]), OA))



def test():
    #  ##### val ################################################################################################
    model.train()
    model.load_state_dict(torch.load('output.pth'))
    model.eval()
    if not os.path.isdir('/mask'):
        os.mkdir('/mask')
    with torch.no_grad():
        evaluator.reset()
        results = []
        for data_test in tqdm(test_loader):
            # raw_prediction NxCxHxW
            data_img = data_test['img'].to(device)
            input = data_img[:, 0:3, :, :]
            ndsm = data_img[:, 3, :, :]
            raw_predictions = model(input,ndsm)#
            masks_true = data_test['gt_semantic_seg']
            image_ids = data_test["img_id"]
            raw_predictions = nn.Softmax(dim=1)(raw_predictions)
            predictions = raw_predictions.argmax(dim=1)
            for i in range(raw_predictions.shape[0]):
                mask = predictions[i].cpu().numpy()
                evaluator.add_batch(gt_image=masks_true[i].cpu().numpy(), pre_image=mask)
                mask_name = image_ids[i]
                results.append(
                    (mask, '/mask/{}'.format(mask_name), True))
        iou_per_class = evaluator.Intersection_over_Union()
        f1_per_class = evaluator.F1()
        OA = evaluator.OA()
        torch.cuda.empty_cache()

        for class_name, class_iou, class_f1 in zip(CLASSES, iou_per_class, f1_per_class):
            print('F1_{}:{}, IOU_{}:{}'.format(class_name, class_f1, class_name, class_iou))
        print('F1:{}, mIOU:{}, OA:{}'.format(np.nanmean(f1_per_class[:-1]), np.nanmean(iou_per_class[:-1]), OA))
        mpp.Pool(processes=mp.cpu_count()).map(img_writer, results)


if __name__ == "__main__":
    train()
    test()