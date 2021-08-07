import numpy as np
from torch import nn
import torch
import utils
import supervisely_lib as sly
from torchvision.utils import draw_segmentation_masks


def inference(model: nn.Module, classes, input_height, input_width, image_path, save_path):
    with torch.no_grad():
        model.eval()
    image = sly.image.read(image_path)  # RGB
    input = utils.prepare_image_input(image, input_width, input_height)
    input = torch.unsqueeze(input, 0)
    input = utils.cuda(input)
    output = model(input)

    #sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(classes)}

    #normalized_masks = torch.nn.functional.softmax(output, dim=1)  # each mask for every class (probabilities)
    predicted_classes = output.data.cpu().numpy().argmax(axis=1)
    model_classes = [sly.ObjClass.from_json(data) for data in classes]
    colors = np.array([cls.color for cls in model_classes])
    colored_mask = colors[predicted_classes[0]]
    sly.image.write(save_path, colored_mask)

    # class_colors = [tuple(obj_class.color) for obj_class in model_classes]
    # draw_segmentation_masks(image, )
    # dog_and_boat_masks = [
    #     normalized_masks[img_idx, sem_class_to_idx[cls]]
    #     for img_idx in range(batch.shape[0])
    #     for cls in ('dog', 'boat')
    # ]

    #dog_with_all_masks = draw_segmentation_masks(image, masks=dog1_all_classes_masks, alpha=.6)
    # mask = COLORS[classMap]
    # resize the mask such that its dimensions match the original size
    # of the input frame
    # mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]),
    #                   interpolation=cv2.INTER_NEAREST)
    # # perform a weighted combination of the input frame with the mask
    # # to form an output visualization
    # output = ((0.3 * frame) + (0.7 * mask)).astype("uint8")



