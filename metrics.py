import torch

# todo: change it to something like
# metric = accuracy(topk=(1,))
# metric_val = metric(output, target)

def accuracy(topk=(1,)):
    return TopK(topk)

class TopK:
    def __init__(self, topk=(1,)):
        self._topk = topk

    # TODO: make the class call a function, so we don't have to replace topk with self._topk
    # TODO: we want users to provide a function to call straight away
    def __call__(self, output, target):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(self._topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in self._topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res