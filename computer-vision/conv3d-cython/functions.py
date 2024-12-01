import torch


def iou(box_1: torch.Tensor, box_2: torch.Tensor) -> float:
    """
    Determines de Intesection over Union (IoU) of two boxes. The tensors have both four
    elements and they refer to (start_width, start_height, end_width, end_height).

    Parameters
    ----------
    box_1 : Coordinates of the first box.
    box_2 : Coordinates of the second box.

    Returns
    -------
    IoU of the two boxes.
    ñlasdfñlkjasdñfkljsdañlfkjdsñlkj
    """

    # TODO

    first_box = torch.tensor(box_1)
    second_box = torch.tensor(box_2)
    x1 = torch.max(first_box[0], second_box[0])
    y1 = torch.max(first_box[1], second_box[1])
    x2 = torch.min(first_box[2], second_box[2])
    y2 = torch.min(first_box[3], second_box[3])
    intersection = torch.max(torch.tensor(0), x2 - x1) * torch.max(
        torch.tensor(0), y2 - y1
    )
    area_1 = (first_box[2] - first_box[0]) * (first_box[3] - first_box[1])
    area_2 = (second_box[2] - second_box[0]) * (second_box[3] - second_box[1])
    union = area_1 + area_2 - intersection
    return (intersection / union).item()


def nms(
    boxes: torch.Tensor, scores: torch.Tensor, threshold: float = 0.75
) -> torch.Tensor:
    """
    Implements the Non-Max Suppression (NMS) algorithm.

    Parameters
    ----------
    boxes     : Tensor with all the predicted boxes. Each box must have four elements
                (start_width, start_height, end_width, end_height).
    scores    : Tensor with scores for each box.
    threshold : IoU threshold to discard overlapping boxes (IoU > threshold).

    Returns
    -------
    Tensor with the indices of the boxes that have been kept by NMS.
    """

    # TODO
    indices = torch.argsort(scores, descending=True)
    keep = []
    while len(indices) > 0:
        keep.append(indices[0])
        iou_values = torch.tensor(
            [iou(boxes[indices[0]], boxes[i]) for i in indices[1:]]
        )
        indices = indices[1:][iou_values <= threshold]
    return torch.tensor(keep)


def _pool_one_roi(
    feature_map: torch.Tensor, roi: torch.Tensor, pooled_height: int, pooled_width: int
) -> torch.Tensor:
    """
    Applies ROI pooling to a single image and a single region of interest.

    Parameters
    ----------
    feature_map   : Feature map coming from the CNN. It has dimensions
                    (channels, height, width).
    roi           : Box representing the roi. It must have the components
                    (x1, y1, x2, y2).
    pooled_height : Height of the output tensor.
    pooled_width  : Width of the output tensor.

    Returns
    -------
    pooled_features : Pooled feature map.
    """

    # TODO
    x1, y1, x2, y2 = roi
    h = max(y2 - y1, 1)
    w = max(x2 - x1, 1)
    h_scale = h / pooled_height
    w_scale = w / pooled_width
    pooled_features = torch.zeros(feature_map.shape[0], pooled_height, pooled_width)
    for i in range(pooled_height):
        for j in range(pooled_width):
            y_start = int(y1 + i * h_scale)
            y_end = int(y1 + (i + 1) * h_scale)
            x_start = int(x1 + j * w_scale)
            x_end = int(x1 + (j + 1) * w_scale)
            pooled_features[:, i, j] = torch.max(
                torch.max(feature_map[:, y_start:y_end, x_start:x_end], dim=1)[0], dim=1
            )[0]
    return pooled_features


def roi_pool(
    feature_map: torch.Tensor, roi: torch.Tensor, pooled_height: int, pooled_width: int
) -> torch.Tensor:
    """
    Applies ROI pooling to a single image and a single region of interest.

    Parameters
    ----------
    feature_map   : Feature map coming from the CNN. It has dimensions
                    (channels, height, width).
    roi           : Box representing the rois. Each one must have the components
                    (x1, y1, x2, y2).
    pooled_height : Height of the output tensor.
    pooled_width  : Width of the output tensor.

    Returns
    -------
    pooled_features : Pooled feature map.
    """

    # TODO
    for i in range(roi.shape[0]):
        pooled_features = _pool_one_roi(
            feature_map, roi[i], pooled_height, pooled_width
        )
        if i == 0:
            result = pooled_features.unsqueeze(0)
        else:
            result = torch.cat((result, pooled_features.unsqueeze(0)), dim=0)
    return result
