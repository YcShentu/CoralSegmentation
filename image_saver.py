import numpy as np
from skimage import io
import cv2
import os


COLORS = np.array([[0, 0, 0], [255, 0, 0], [0, 0, 255],
                   [0, 255, 0], [255, 255, 0], [255, 0, 255],
                   [0, 255, 255], [255, 228, 255], [244, 255, 255],
                   [255, 255, 220],
                   ],
                  dtype=np.uint8)


def image_saver(images, masks, out_dir, phase, steps, epoch=0, id=1):
    '''
    image saver for train and evaluation
    save masks (bools) and fused RGB images

    only 2 output-channel results are accepted
    '''

    mask_path = os.path.join(out_dir, phase, 'masks', str(epoch))
    fuse_path = os.path.join(out_dir, phase, 'fuse', str(epoch))
    os.makedirs(mask_path, exist_ok=True)
    os.makedirs(fuse_path, exist_ok=True)

    B, C, H, W = images.shape
    images = images.cpu().numpy()
    masks = masks.cpu().numpy()

    for i in range(B):
        image = np.array((images[i, :, :, :] + 0.5)*255, dtype=np.uint8)
        image = np.tile(np.transpose(image, (1, 2, 0)), (1, 1, 3))

        # create a color mask
        mask = np.array(masks[i, :, :], dtype=np.uint8)
        fused = COLORS[mask]
        fused = cv2.addWeighted(image, 1.0, fused, 0.9, gamma=0)
        io.imsave(os.path.join(fuse_path, str(steps)+'_'+str(i)+'.jpg'), fused)

        # create white mask for save
        white_mask = np.where(mask == id, 255, 0).astype(np.uint8)
        io.imsave(os.path.join(mask_path, str(steps)+'_'+str(i)+'.jpg'), white_mask)


def image_saver_inference(images, masks, out_dir, filenames, epoch=0, id=1):

    mask_path = os.path.join(out_dir, 'masks', str(epoch))
    fuse_path = os.path.join(out_dir, 'fuse', str(epoch))

    B, C, H, W = images.shape
    images = images.cpu().numpy()
    masks = masks.cpu().numpy()

    for i in range(B):
        os.makedirs(os.path.dirname(os.path.join(mask_path, filenames[i])), exist_ok=True)
        os.makedirs(os.path.dirname(os.path.join(fuse_path, filenames[i])), exist_ok=True)

        image = np.array((images[i, :, :, :] + 0.5)*255, dtype=np.uint8)
        image = np.tile(np.transpose(image, (1, 2, 0)), (1, 1, 3))

        # create a color mask
        mask = np.array(masks[i, :, :], dtype=np.uint8)
        fused = COLORS[mask]
        fused = cv2.addWeighted(image, 1.0, fused, 0.9, gamma=0)
        io.imsave(os.path.join(fuse_path, filenames[i]), fused)

        # create white mask for save
        white_mask = np.where(mask == id, 255, 0).astype(np.uint8)
        io.imsave(os.path.join(mask_path, filenames[i]), white_mask)


def image_saver_cam(images, heatmaps, probs, out_dir, phase, steps, epoch=0, id=1):

    mask_path = os.path.join(out_dir, phase, 'masks', str(epoch))
    fuse_path = os.path.join(out_dir, phase, 'fuse', str(epoch))
    cam_path = os.path.join(out_dir, phase, 'cam', str(epoch))

    os.makedirs(mask_path, exist_ok=True)
    os.makedirs(fuse_path, exist_ok=True)
    os.makedirs(cam_path, exist_ok=True)

    B, C, H, W = images.shape
    images = images.cpu().numpy()
    C_out, C_in = probs.shape
    probs = np.exp(probs) / np.tile(np.sum(np.exp(probs), axis=1, keepdims=True), C_in)
    heatmaps = heatmaps.detach().cpu().numpy()
    _, _, H_heat, W_heat = heatmaps.shape

    for i in range(B):
        image = np.array((images[i, :, :, :] + 0.5)*255, dtype=np.uint8)
        image = np.tile(np.transpose(image, (1, 2, 0)), (1, 1, 3))

        # class activation map
        cam_all = np.dot(probs, np.reshape(heatmaps[i], (C_in, H_heat*W_heat)))
        cam_all = np.reshape(cam_all, (C_out, H_heat, W_heat)).transpose((1, 2, 0))
        cam_all = cv2.resize(cam_all, (H, W), interpolation=cv2.INTER_LINEAR)
        cam1 = cam_all[:, :, id]
        # cam0 = cam_all[:, :, 0]  # extract background layer
        # cam1 = cam_all[:, :, 1]  # extract coral layer
        # cam0 = (cam0 - np.min(cam0)) / (np.max(cam0) - np.min(cam0))
        cam1 = (cam1 - np.min(cam1)) / (np.max(cam1) - np.min(cam1))

        # create fused image
        thresh = 0.5
        mask_red = COLORS[np.where(cam1 > thresh, 1, 0).astype(np.uint8)]
        mask_fused = cv2.addWeighted(image, 1.0, mask_red, 0.9, gamma=0)
        io.imsave(os.path.join(fuse_path, str(steps) + '_' + str(i) + '.jpg'), mask_fused)

        # create white mask
        mask_white = mask_red[:, :, 0]
        io.imsave(os.path.join(mask_path, str(steps) + '_' + str(i) + '.jpg'), mask_white)

        # vis cam
        # cam0 = (cam0 * 255.0).astype(np.uint8)
        cam1 = (cam1 * 255.0).astype(np.uint8)
        # cam0 = cv2.applyColorMap(cam0, cv2.COLORMAP_JET)[:, :, ::-1]  # convert into RGB
        cam1 = cv2.applyColorMap(cam1, cv2.COLORMAP_JET)[:, :, ::-1]
        # cam0 = cv2.addWeighted(image, 0.3, cam0, 0.8, gamma=0)
        cam1 = cv2.addWeighted(image, 0.3, cam1, 0.8, gamma=0)
        # cam_concat = np.concatenate((cam0, cam1, image), axis=0)
        io.imsave(os.path.join(cam_path, str(steps) + '_' + str(i) + '.jpg'), cam1)


def image_saver_cam_inference(images, heatmaps, probs, out_dir, filenames, epoch=0):

    mask_path = os.path.join(out_dir, 'masks', str(epoch))
    fuse_path = os.path.join(out_dir, 'fuse', str(epoch))
    cam_path = os.path.join(out_dir, 'cam', str(epoch))

    os.makedirs(mask_path, exist_ok=True)
    os.makedirs(fuse_path, exist_ok=True)
    os.makedirs(cam_path, exist_ok=True)

    B, C, H, W = images.shape
    images = images.cpu().numpy()
    C_out, C_in = probs.shape
    probs = np.exp(probs) / np.tile(np.sum(np.exp(probs), axis=1, keepdims=True), C_in)
    heatmaps = heatmaps.detach().cpu().numpy()
    _, _, H_heat, W_heat = heatmaps.shape

    for i in range(B):
        os.makedirs(os.path.dirname(os.path.join(mask_path, filenames[i])), exist_ok=True)
        os.makedirs(os.path.dirname(os.path.join(fuse_path, filenames[i])), exist_ok=True)
        os.makedirs(os.path.dirname(os.path.join(cam_path, filenames[i])), exist_ok=True)

        image = np.array((images[i, :, :, :] + 0.5) * 255, dtype=np.uint8)
        image = np.tile(np.transpose(image, (1, 2, 0)), (1, 1, 3))

        # class activation map
        cam_all = np.dot(probs, np.reshape(heatmaps[i], (C_in, H_heat * W_heat)))
        cam_all = np.reshape(cam_all, (C_out, H_heat, W_heat)).transpose((1, 2, 0))
        cam_all = cv2.resize(cam_all, (H, W), interpolation=cv2.INTER_LINEAR)
        cam1 = cam_all[:, :, id]
        cam1 = (cam1 - np.min(cam1)) / (np.max(cam1) - np.min(cam1))

        # create fused image
        thresh = 0.5
        mask_red = COLORS[np.where(cam1 > thresh, 1, 0).astype(np.uint8)]
        mask_fused = cv2.addWeighted(image, 1.0, mask_red, 0.9, gamma=0)
        io.imsave(os.path.join(fuse_path, filenames[i]), mask_fused)

        # create white mask
        mask_white = mask_red[:, :, 0]
        io.imsave(os.path.join(mask_path, filenames[i]), mask_white)

        # vis cam
        cam1 = (cam1 * 255.0).astype(np.uint8)
        cam1 = cv2.applyColorMap(cam1, cv2.COLORMAP_JET)[:, :, ::-1]
        cam1 = cv2.addWeighted(image, 0.3, cam1, 0.8, gamma=0)
        io.imsave(os.path.join(cam_path, filenames[i]), cam1)
