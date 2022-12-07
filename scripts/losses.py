import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# set seed for reproducibility
torch.manual_seed(1337)
np.random.seed(3459)
# tf.set_random_seed(3459)

torch.autograd.set_detect_anomaly(True)

EPS = 1e-8

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class SoftHingeMult(nn.Module):

    def __init__(self, num_class=10, reduction="none"):
        super(SoftHingeMult, self).__init__()

        self.reduction = reduction
        self.num_class = num_class

    def forward(self, output, target_label, one_hot=True):

        if one_hot:
            y_true = F.one_hot(target_label.type(torch.LongTensor), num_classes=self.num_class)
            y_true = y_true.type(torch.FloatTensor).to(DEVICE)

        # Compute Margins for multi-class
        marg_lab = torch.sum(output*y_true, dim=1)
        marg_max, _ = torch.max(output - output*y_true, dim=1)
        marg_1 = marg_lab - marg_max
        marg_2 = marg_lab - torch.logsumexp(output, dim=1)

        idx_mask = torch.zeros(output.shape[0]).to(DEVICE)
        idx_tmp = torch.where(marg_1 >= 0)[0]
        idx_mask[idx_tmp] = 1

        loss_val = idx_mask * torch.max(1. -  marg_1,
                                        torch.zeros(output.shape[0]).to(DEVICE)) + \
                                        ((1 - idx_mask) * torch.max(1. -  marg_2,
                                                                    torch.zeros(output.shape[0]).to(
                                                                        DEVICE)))

        if self.reduction == "sum":
            return torch.sum(loss_val)
            
        if self.reduction == "mean":
            return torch.mean(loss_val)
        return loss_val

class MAE(nn.Module):

    def __init__(self, num_class=10, reduction="none"):
        super(MAE, self).__init__()

        self.reduction = reduction
        self.num_class = num_class

    def forward(self, prediction, target_label, one_hot=True):

        if one_hot:
            y_true = F.one_hot(target_label.type(torch.LongTensor), num_classes=self.num_class)
            y_true = y_true.type(torch.FloatTensor).to(DEVICE)
        
        prediction = F.softmax(prediction, dim=1)
        y_pred = torch.clamp(prediction, EPS, 1-EPS)

        if self.reduction == "none":
            return torch.sum(F.l1_loss(y_pred, y_true, reduction="none"), dim=1)
        
        return F.l1_loss(y_pred, y_true, reduction=self.reduction)

class MSE(nn.Module):
    def __init__(self, num_class=10, reduction="none"):
        super(MSE, self).__init__()

        self.reduction = reduction
        self.num_class = num_class

    def forward(self, prediction, target_label, one_hot=True):

        if one_hot:
            y_true = F.one_hot(target_label.type(torch.LongTensor), num_classes=self.num_class)
            y_true = y_true.type(torch.FloatTensor).to(DEVICE)
        
        prediction = F.softmax(prediction, dim=1)
        y_pred = torch.clamp(prediction, EPS, 1-EPS)

        if self.reduction == "none":
            return torch.sum(F.mse_loss(y_pred, y_true, reduction="none"), dim=1)
        else:
            return F.mse_loss(y_pred, y_true, reduction=self.reduction)

class NormMSE(nn.Module):
    def __init__(self, alpha=0.1, beta=1, num_class=10, reduction="none"):
        super(NormMSE, self).__init__()
        self.reduction = reduction
        self.num_class = num_class
        self.alpha = alpha
        self.beta = beta

    def forward(self, prediction, target_label, one_hot=True):
        if one_hot:
            y_true = F.one_hot(target_label.type(torch.LongTensor),
                               num_classes=self.num_class)
            y_true = y_true.type(torch.FloatTensor).to(DEVICE)
        
        prediction = F.softmax(prediction, dim=1)
        y_pred = torch.clamp(prediction, EPS, 1-EPS).to(DEVICE)
        norm_const = (self.num_class*((torch.norm(y_pred,
                                                  None, dim=1))**2)) + self.num_class - 2

        # norm_const = torch.clamp(norm_const, min=1e-8)

        l1_sum = (1./norm_const) * torch.sum(F.mse_loss(y_pred,
                                                        y_true,
                                                        reduction="none"), dim=1)
        ## l2_sum = -1. * torch.sum(y_pred * torch.log(torch.clamp(y_true, EPS, 1.0)), dim=1)
        # l2_sum = torch.sum(F.l1_loss(y_pred, y_true, reduction="none"), dim=1)

        if self.reduction == "mean":
            return self.alpha * torch.mean(l1_sum,
                                           dim=0) # + (self.beta * torch.mean(l2_sum, dim=0))
        elif self.reduction == "sum":
            return self.alpha * torch.sum(l1_sum, dim=0) # + (self.beta * torch.sum(l2_sum, dim=0))
        else:
            return self.alpha * l1_sum # + (self.beta * l2_sum)

class NormCCE(nn.Module):

    def __init__(self, alpha=0.1, beta=1, num_class=10, reduction="none"):
        super(NormCCE, self).__init__()
        self.reduction = reduction
        self.num_class = num_class
        self.alpha = alpha
        self.beta = beta

    def forward(self, prediction, target_label, one_hot=True):
        
        if one_hot:
            y_true = F.one_hot(target_label.type(torch.LongTensor),
                               num_classes=self.num_class)
            y_true = y_true.type(torch.FloatTensor).to(DEVICE)
        
        prediction = F.softmax(prediction, dim=1)
        y_pred = torch.clamp(prediction, EPS, 1-EPS)

        l1_sum = torch.sum(y_true * torch.log(1. + y_pred), dim=1)

        norm_const = torch.sum(torch.log(1. + y_pred.repeat_interleave(self.num_class, dim=0)) *
                               (torch.eye(self.num_class)).repeat(y_true.shape[0], 1).to(DEVICE),
                               dim=1)
        norm_const = torch.reshape(norm_const, (self.num_class, -1))
        norm_const = torch.sum(norm_const, dim=0)

        if torch.min(norm_const) < 1e-8:
            raise SystemExit("Denominator too small.\n")

        l1_sum = (1./norm_const) * l1_sum
        ## l2_sum = -1. * torch.sum(y_pred * torch.log(torch.clamp(y_true, EPS, 1.0)),
        #                           dim=1) # gives denom. < 1e-8
        l2_sum = torch.sum(F.l1_loss(y_pred, y_true, reduction="none"), dim=1)

        if self.reduction == "sum":
            return (self.alpha * torch.sum(l1_sum, dim=0)) + \
                    (self.beta * torch.sum(l2_sum, dim=0))
        elif self.reduction == "mean":
            return (self.alpha * torch.mean(l1_sum, dim=0)) + \
                    (self.beta * torch.mean(l2_sum, dim=0))
        else:
            return (self.alpha * l1_sum) + (self.beta * l2_sum)


class WeightedCCE(nn.Module):
    """
    Implementing BARE with Cross-Entropy (CCE) Loss
    """

    def __init__(self, k=1, num_class=10, reduction="mean"):
        super(WeightedCCE, self).__init__()

        self.k = k
        self.reduction = reduction
        self.num_class = num_class

    def forward(self, prediction, target_label, one_hot=True):
        if one_hot:
            y_true = F.one_hot(target_label.type(torch.LongTensor),
                               num_classes=self.num_class).to(DEVICE)
        y_pred = F.softmax(prediction, dim=1)
        y_pred = torch.clamp(y_pred, EPS, 1-EPS)
        pred_tmp = torch.sum(y_true * y_pred, axis=-1).reshape(-1, 1)

        ## Compute batch statistics
        avg_post = torch.mean(y_pred, dim=0)
        avg_post = avg_post.reshape(-1, 1)
        std_post = torch.std(y_pred, dim=0)
        std_post = std_post.reshape(-1, 1)
        avg_post_ref = torch.matmul(y_true.type(torch.float), avg_post)
        std_post_ref = torch.matmul(y_true.type(torch.float), std_post)
        pred_prun = torch.where((pred_tmp - avg_post_ref >= self.k * std_post_ref),
                                pred_tmp, torch.zeros_like(pred_tmp))

        # prun_idx will tell us which examples are
        # 'trustworthy' for the given batch
        prun_idx = torch.where(pred_prun != 0.)[0]
        if len(prun_idx) != 0:
            prun_targets = torch.argmax(torch.index_select(y_true, 0, prun_idx), dim=1)
            weighted_loss = F.cross_entropy(torch.index_select(prediction, 0, prun_idx), 
                                            prun_targets, reduction=self.reduction)
        else:
            weighted_loss = F.cross_entropy(prediction, target_label)
        return weighted_loss, prun_idx


class RLL(nn.Module):

    def __init__(self, alpha=0.45, num_class=10, reduction="none"):
        super(RLL, self).__init__()
        
        self.alpha = torch.Tensor([alpha]).to(DEVICE)
        self.reduction = reduction
        self.num_class = num_class

    def forward(self, prediction, target_label, one_hot=True):
        if one_hot:
            y_true = F.one_hot(target_label.type(torch.LongTensor),
                               num_classes=self.num_class).to(DEVICE)

        prediction = F.softmax(prediction, dim=1)
        y_pred = torch.clamp(prediction, EPS, 1-EPS)
        y_t = (((1.-y_true)/(self.num_class - 1))*torch.log(self.alpha + y_pred)) \
               - (y_true*torch.log(self.alpha + y_pred)) \
                    + y_true*(torch.log(self.alpha + 1) - torch.log(self.alpha))
        
        temp = torch.sum(y_t, dim=1)
        if self.reduction == "none":
            return temp
        elif self.reduction == "mean":
            return torch.mean(temp, dim=0)
        elif self.reduction == "sum":
            return torch.sum(temp, dim=0)

class DMI(nn.Module):

    def __init__(self, num_class=10):
        super(DMI, self).__init__()

        self.num_class = num_class

    def forward(self, prediction, target_label, one_hot=True):
        """prediction and target_label should be of size [batch_size, num_class]
        """
        if one_hot:
            y_true = F.one_hot(target_label.type(torch.LongTensor), num_classes=self.num_class)
            y_true = y_true.type(torch.FloatTensor).to(DEVICE)

        prediction = F.softmax(prediction, dim=1)
        y_pred = torch.clamp(prediction, EPS, 1-EPS)

        U = torch.matmul(torch.transpose(y_true, 0, 1), y_pred)
        return -1.0 * torch.log(torch.abs(torch.det(U.type(torch.FloatTensor))
                                         ).type(torch.FloatTensor) + 1e-3)


class GCE(nn.Module):
    """
    Implementing Generalised Cross-Entropy (GCE) Loss 
    """
    def __init__(self, q=0.7, num_class=10, reduction="none"):
        super(GCE, self).__init__()
        self.q = q
        self.reduction = reduction
        self.num_class = num_class
    def forward(self, prediction, target_label, one_hot=True):
        """
        Function to compute GCE loss.
        Usage: total_loss = GCE(target_label, prediction)
        Arguments:
            prediction : A 2d tensor of shape (batch_size, num_classes), with
                        each element in the ith row representing the 
                        probability of the corresponding class being present
                        in the ith sample.
            target_label: A 2d tensor of shape (batch_size, num_classes), with 
                        each element in the ith row representing the presence 
                        or absence of the corresponding class in the ith 
                        sample.
        """
        if one_hot:
            y_true = F.one_hot(target_label.type(torch.LongTensor),
                               num_classes=self.num_class).to(DEVICE)

        prediction = F.softmax(prediction, dim=1)
        y_pred = torch.clamp(prediction, EPS, 1-EPS)

        t_loss = (1. - torch.pow(torch.sum(y_true.type(torch.float)
                                           * y_pred, dim=1), self.q)) / self.q
        
        if self.reduction == "mean":
            return torch.mean(t_loss, dim=0)
        elif self.reduction == "sum":
            return torch.sum(t_loss, dim=0)
        else:
            return t_loss

def get_loss(loss_name, num_class, reduction="none", **kwargs):

    if loss_name == "cce":
        loss_fn = nn.CrossEntropyLoss(reduction=reduction)
    elif loss_name == "bare_cce":
        k = 1
        loss_fn = WeightedCCE(k=k, num_class=num_class, reduction=reduction)
    elif loss_name == "norm_cce":
        try:
            alpha = kwargs['alpha']
            beta = kwargs['beta']
        except KeyError:
            alpha = 0.1
            beta = 1
        loss_fn = NormCCE(alpha=alpha, beta=beta, num_class=num_class, reduction=reduction)
    elif loss_name == "gce":
        try:
            q = kwargs['q']
        except KeyError:
            q = 0.7
        loss_fn = GCE(q=q, num_class=num_class, reduction=reduction)
    elif loss_name == "dmi":
        loss_fn = DMI(num_class=num_class)
    elif loss_name == "rll":
        try:
            alpha = kwargs['alpha']
        except KeyError:
            alpha = 0.01 # 0.45 # 0.45/0.5/0.6 => works well with lr = 3e-3 => ADAM
        loss_fn = RLL(alpha=alpha, num_class=num_class, reduction=reduction)
    elif loss_name == "mae":
        loss_fn = MAE(num_class=num_class, reduction=reduction)
    elif loss_name == "mse":
        loss_fn = MSE(num_class=num_class, reduction=reduction)
    elif loss_name == "norm_mse":
        try:
            alpha = kwargs['alpha']
            beta = kwargs['beta']
        except KeyError:
            alpha = 1 # 0.1
            beta = 1
        loss_fn = NormMSE(alpha=alpha, beta=beta, num_class=num_class, reduction=reduction)
    elif loss_name == "soft_hinge":
        loss_fn = SoftHingeMult(num_class=num_class, reduction=reduction)
    else:
        raise NotImplementedError("Loss Function Not Implemented.\n")
    return loss_fn
