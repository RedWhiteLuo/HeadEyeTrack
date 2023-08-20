import cv2
import numpy as np
from sklearn.metrics import mean_squared_error


def save_img_and_coords(img, coords, saved_img_index):
    save_path = './dataset/' + '%d_%d_%d.png' % (saved_img_index, coords[0], coords[1])
    cv2.imwrite(save_path, img)
    saved_img_index += 1


def return_boundary(eye_bbox):
    """
    :param eye_bbox:
    :return:
    """
    max_index = np.argmax(eye_bbox, axis=0)
    min_index = np.argmin(eye_bbox, axis=0)
    left, top = eye_bbox[min_index[0]][0], eye_bbox[min_index[1]][1]
    right, down = eye_bbox[max_index[0]][0], eye_bbox[max_index[1]][1]
    return left - 2, right + 2, top - 2, down + 2


def trim_eye_img(image, face_kp):
    l_l, l_r, l_t, l_b = return_boundary(face_kp[60:68])
    r_l, r_r, r_t, r_b = return_boundary(face_kp[68:76])
    left_eye_img = image[int(l_t):int(l_b), int(l_l):int(l_r)]
    right_eye_img = image[int(r_t):int(r_b), int(r_l):int(r_r)]
    left_eye_img = cv2.resize(left_eye_img, (64, 32), interpolation=cv2.INTER_AREA)
    right_eye_img = cv2.resize(right_eye_img, (64, 32), interpolation=cv2.INTER_AREA)
    return np.concatenate((left_eye_img, right_eye_img), axis=1)


def transform_fn(data_item):
    # convert input tensor into float format
    images = data_item.float()
    # scale input
    images = images / 255
    # convert torch tensor to numpy array
    images = images.cpu().detach().numpy()
    return images


def transform_fn_with_annot(data_item):
    images = data_item.float()[0]
    images = images / 255
    images = images.cpu().detach().numpy()
    return images


def validate(model, validation_loader) -> float:
    """
    :param model: model that need to be quantized
    :param validation_loader: the func 'create_data_source(path=None, with_annot=True)', return (image, target) in 'for'
    :return: return float: MSE between annot and predict
    """
    predictions, references = [], []
    output = model.outputs[0]
    for images, target in validation_loader:
        pred = model(images)[output][0]
        predictions.append(pred)
        references.append(target)
    predictions = np.concatenate(predictions, axis=0)
    references = np.concatenate(references, axis=0)
    return mean_squared_error(predictions, references)
