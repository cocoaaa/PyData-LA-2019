import torch.nn.functional as F
import torch

def get_recon_losses(x_recon, x, distribution: str = 'bernoulli', **kwargs):
    """
    Computes reconstruction loss between (batch of) x_recon and (batch of) original input x.
    The loss is averaged over both spatial and batch dimension, thus indicating the recon loss
    per pixel
    :param x_recon: Tensor. 1st dim is batch size dim
    :param x: Tensor. Same shape as `x_recon`
    :param distribution: the p.distribution the decoder is parametrizing, must be either 'bernoulli' (for binary classification) or 'gaussian' for regression problem
    :param kwargs: optional argument 'pos_weight' as weight on positive class during loss computation
    :return:
    """
    pos_weight = kwargs.get('pos_weight', None)
    reduction = kwargs.get('reduction', 'sum')
    if pos_weight is not None:
        pos_weight = pos_weight.to(x_recon.device)

    distribution = distribution.lower()
    assert distribution in ['bernoulli', 'gaussian'], f"distribution must be either bernoulli or gaussian: {distribution}"
    batch_size = x.size(0)
    assert batch_size > 0

    if distribution == 'bernoulli':
        #         Use recon_loss_fn of nn.BCEWithLogitsLoss
        # batch_mean_recon_loss = F.binary_cross_entropy_with_logits(x_recon, x, size_average=False).div(batch_size)
        # mean_recon_loss = F.binary_cross_entropy_with_logits(x_recon, x, reduction='mean')
        weighted_recon_loss = F.binary_cross_entropy_with_logits(
            x_recon, x, reduction=reduction ,pos_weight=pos_weight
        )  # avg per-pixel cross-entropy loss (avged over mini-batch)
    else:
        raise NotImplementedError

    # return mean_recon_loss, mean_weighted_recon_loss ,batch_mean_recon_loss
    return weighted_recon_loss / batch_size


def get_kl_divs(mu, logvar):
    "Copied from: https://tinyurl.com/wa5u7w2"
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld
    # return total_kld


def vae_loss_fn(output, x, beta=1.0, decoder_dist: str = 'bernoulli', **kwargs):
    x_recon, mu, logvar = output
    weighted_recon_loss = get_recon_losses(x_recon, x, distribution=decoder_dist, **kwargs)
    total_kld,_,_ = get_kl_divs(mu, logvar)
    # print('mean_recon: ', mean_recon_loss)
    # print('mean_weighted_recon: ', mean_weighted_recon_loss)
    # print('batch_mean_recon: ', batch_mean_recon_loss)
    # print('total kld: ', total_kld)
    # print('mean_kld: ', mean_kld)
    return weighted_recon_loss + beta * total_kld


