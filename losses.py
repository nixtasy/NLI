from itertools import product
import torch
from torch.nn import BCEWithLogitsLoss, MarginRankingLoss
import torch.nn.functional as F

PAD_Y_VAL = -1

DEFAULT_EPS = 1e-10


def cal_losses(raw_results, id2feature):
    """
    Args:
        raw_results (list[RawResult]):
        id2feature (dict):
    Returns:
        dict:
    """
    with torch.no_grad():
        labels, logits = [], []
        for r in raw_results:
            f = id2feature[r.id]
            labels.append(torch.tensor(f.labels, dtype=torch.float))
            logits.append(r.logits)

        labels = torch.stack(labels)
        logits = torch.stack(logits)

        _losses = dict()
        _losses['list_net'] = list_net(logits, labels).item()  # Likelihood
        _losses['rank_net'] = rank_net(logits, labels).item()  # Hinge

    return _losses
    
def list_net(y_pred, y_true, eps=DEFAULT_EPS, pad_value_indicator=PAD_Y_VAL):
    """
    A listwise approach, calculated the  K-L divergence. 
    
    Args:
        y_pred: predictions from the model, shape [batch_size, list_size]
        y_true: ground truth labels, shape [batch_size, list_size]
        eps: epsilon value, used for numerical stability
        pad_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    Returns:
        torch.Tensor: loss value
    """

    y_pred = y_pred.clone()
    y_true = y_true.clone()

    mask = y_true == pad_value_indicator
    y_pred[mask] = float('-inf')
    y_true[mask] = float('-inf')

    preds_smax = F.softmax(y_pred, dim=1)
    true_smax = F.softmax(y_true, dim=1)

    preds_smax = preds_smax + eps
    preds_log = torch.log(preds_smax)

    return torch.mean(-torch.sum(true_smax * preds_log, dim=1))


def rank_net(y_pred, y_true, padded_value_indicator=PAD_Y_VAL, weight_by_diff=False, weight_by_diff_powed=False):
    """
    A pairwise approach, uses the logistic (cross entropy) loss.

    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param padded_value_indicator:
    :param weight_by_diff: flag indicating whether to weight the score differences by ground truth differences.
    :param weight_by_diff_powed: flag indicating whether to weight the score differences by the squared ground truth differences.
    :return: loss value, a torch.Tensor
    """
    y_pred = y_pred.clone()
    y_true = y_true.clone()

    mask = y_true == padded_value_indicator
    y_pred[mask] = float('-inf')
    y_true[mask] = float('-inf')

    # here we generate every pair of indices from the range of document length in the batch
    document_pairs_candidates = list(product(range(y_true.shape[1]), repeat=2))

    pairs_true = y_true[:, document_pairs_candidates]
    selected_pred = y_pred[:, document_pairs_candidates]

    # here we calculate the relative true relevance of every candidate pair
    true_diffs = pairs_true[:, :, 0] - pairs_true[:, :, 1]
    pred_diffs = selected_pred[:, :, 0] - selected_pred[:, :, 1]

    # here we filter just the pairs that are 'positive' and did not involve a padded instance
    # we can do that since in the candidate pairs we had symetric pairs so we can stick with
    # positive ones for a simpler loss function formulation
    the_mask = (true_diffs > 0) & (~torch.isinf(true_diffs))

    pred_diffs = pred_diffs[the_mask]

    weight = None
    if weight_by_diff:
        abs_diff = torch.abs(true_diffs)
        weight = abs_diff[the_mask]
    elif weight_by_diff_powed:
        true_pow_diffs = torch.pow(pairs_true[:, :, 0], 2) - torch.pow(pairs_true[:, :, 1], 2)
        abs_diff = torch.abs(true_pow_diffs)
        weight = abs_diff[the_mask]

    # here we 'binarize' true relevancy diffs since for a pairwise loss we just need to know
    # whether one document is better than the other and not about the actual difference in
    # their relevancy levels
    true_diffs = (true_diffs > 0).type(torch.float32)
    true_diffs = true_diffs[the_mask]

    return BCEWithLogitsLoss(weight=weight)(pred_diffs, true_diffs)
