import torch
import torch.nn.functional as F

def confusion_matrix(input, target, num_classes, ignore_index = None):
    """
    input: torch.LongTensor:(N, H, W)
    target: torch.LongTensor:(N, H, W)
    num_classes: int
    results:Tensor
    """
    assert torch.max(input) < num_classes
    # assert torch.max() < num_classes
    H, W = target.size()[-2:]
    results = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for i, j in zip(target.flatten(), input.flatten()):
        if ignore_index is not None and i == ignore_index:
            continue
        results[i, j] += 1
    print(results)
    return results
    
def pixel_accuracy(input, target, ignore_index = None):
    """
    input: torch.FloatTensor:(N, C, H, W)
    target: torch.LongTensor:(N, H, W)
    return: Tensor
    """
    assert len(input.size()) == 4
    assert len(target.size()) == 3
    N, H, W = target.size()
    input = F.softmax(input, dim=1)
    arg_max = torch.argmax(input, dim=1)
    # (TP + TN) / (TP + TN + FP + FN)
    division_num = N * H * W - torch.sum(target == ignore_index)
    if division_num != 0:
        return torch.sum(arg_max == target) / division_num
    else:
        return None

def mean_pixel_accuarcy(input, target, ignore_index = None):
    """
    input: torch.FloatTensor:(N, C, H, W)
    target: torch.LongTensor:(N, H, W)
    return: Tensor
    """
    N, num_classes, H, W = input.size()
    input = F.softmax(input, dim=1)
    arg_max = torch.argmax(input, dim=1)
    confuse_matrix = confusion_matrix(arg_max, target, num_classes, ignore_index=ignore_index)
    result = 0
    n_classes = 0
    for i in range(num_classes):
        if torch.sum(confuse_matrix[i,:]) != 0:
            result += (confuse_matrix[i, i] / torch.sum(confuse_matrix[i, :]))
            n_classes += 1
    if n_classes == 0:
        return None
    return result / n_classes

def mean_iou(input, target, ignore_index = None):
    """
    input: torch.FloatTensor:(N, C, H, W)
    target: torch.LongTensor:(N, H, W)
    return: Tensor
    """
    assert len(input.size()) == 4
    assert len(target.size()) == 3
    N, num_classes, H, W = input.size()
    input = F.softmax(input, dim=1)
    arg_max = torch.argmax(input, dim=1)
    result = 0
    confuse_matrix = confusion_matrix(arg_max, target, num_classes)
    for i in range(num_classes):
        nii = confuse_matrix[i, i]
        # consider the case where the denominator is zero.
        if nii == 0:
            continue
        else:
            ti, tj = torch.sum(confuse_matrix[i, :]), torch.sum(confuse_matrix[:, i])
            result += (nii / (ti + tj - nii))

    return result / num_classes

def frequency_weighted_iou(input, target):
    """
    input: torch.FloatTensor:(N, C, H, W)
    target: torch.LongTensor:(N, H, W)
    return: Tensor
    """
    assert len(input.size()) == 4
    assert len(target.size()) == 3
    N, num_classes, H, W = input.size()
    input = F.softmax(input, dim=1)
    arg_max = torch.argmax(input, dim=1)
    # get confusion matrix
    result = 0
    confuse_matrix = confusion_matrix(arg_max, target, num_classes)
    for i in range(num_classes):
        nii = confuse_matrix[i, i]
        # consider the case where the denominator is zero.
        if nii == 0:
            continue
        else:
            ti, tj = torch.sum(confuse_matrix[i, :]), torch.sum(confuse_matrix[:, i])
            result += (ti * nii / (ti + tj - nii))

    return result / torch.sum(confuse_matrix)

def all_metric(input, target, ignore_index = None):
    """
    calculate confusion_matrix, pixel_accuracy, mean_pixel_accuracy mean_iou and frequency_weighted_iou for each batch
    input: torch.FloatTensor:(N, C, H, W)
    target: torch.LongTensor:(N, H, W)
    ignore_index: NULL label
    return: torch.Tensor
    """
    assert len(input.size()) == 4
    assert len(target.size()) == 3
    N, num_classes, H, W = input.size()
    pixel_accuracy = 0
    mean_pixel_accuarcy = 0
    mean_iou = 0
    frequency_weighted_iou = 0
    input = F.softmax(input, dim=1)
    arg_max = torch.argmax(input, dim=1)
    confuse_matrix = confusion_matrix(arg_max, target, num_classes, ignore_index=ignore_index)
    #===========calculate pixel_accuracy===============
    division_num = N * H * W - torch.sum(target == ignore_index)
    if division_num != 0:
        pixel_accuracy = torch.sum(arg_max == target) / division_num
    else:
        pixel_accuracy =  torch.nan
    #===========calculate mean_pixel_accuracy==========
    count_valid_classes = 0
    for i in range(num_classes):
        if torch.sum(confuse_matrix[i,:]) != 0:
            mean_pixel_accuarcy += (confuse_matrix[i, i] / torch.sum(confuse_matrix[i, :]))
            count_valid_classes += 1
    if count_valid_classes == 0:
        mean_pixel_accuarcy = torch.nan
    else:
        mean_pixel_accuarcy = mean_pixel_accuarcy/count_valid_classes
    #===========calculate mean_iou=====================
    count_valid_classes = 0
    for i in range(num_classes):
        nii = confuse_matrix[i, i]
        ti, tj = torch.sum(confuse_matrix[i, :]), torch.sum(confuse_matrix[:, i])
        if ti != 0 and tj != 0:
            count_valid_classes += 1
            # consider the case where the denominator is zero.
            if nii == 0:
                continue
            else:
                mean_iou += (nii / (ti + tj - nii))
    if count_valid_classes == 0:
        mean_iou = torch.nan
    else:
        mean_iou = mean_iou/count_valid_classes
    #===========calculate frequency_weighted_iou========
    count_valid_classes = 0
    for i in range(num_classes):
        nii = confuse_matrix[i, i]
        ti, tj = torch.sum(confuse_matrix[i, :]), torch.sum(confuse_matrix[:, i])
        if ti != 0 and tj != 0:
            count_valid_classes += 1
            # consider the case where the denominator is zero.
            if nii == 0:
                continue
            else:
                frequency_weighted_iou += (nii*ti / (ti + tj - nii))
    if count_valid_classes == 0:
        frequency_weighted_iou = torch.nan
    else:
        frequency_weighted_iou = frequency_weighted_iou/torch.sum(confuse_matrix)
    #========================output================================
    batch_metric = torch.tensor([pixel_accuracy,mean_pixel_accuarcy,mean_iou,frequency_weighted_iou])
    return batch_metric
    


