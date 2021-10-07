# dynamic task balancing
import torch
import torch.nn as nn

class DTBLayer(nn.Module):
    def __init__(self, shared_weights, task_num, alpha=1.25, lr=1e-3, beta=0.998, params_initial=None):
        super().__init__()
        self.shared_weights = shared_weights
        self.task_num = task_num
        self.alpha = alpha
        self.lr = lr
        if params_initial == None:
            params_initial = [1. / task_num] * task_num
        self.params_list = nn.Parameter(torch.tensor(params_initial), requires_grad=True)
        self.mv_loss = torch.tensor([0.] * task_num, requires_grad=False)
        self.beta = beta
        self.global_step = 0
        self.task_difficulty = None

    def get_weight(self, i):
        assert i < self.task_num
        return self.params_list[i]

    def get_task_difficulty(self):
        return self.task_difficulty

    def forward(self, *args):
        assert len(args) == self.task_num
        gradnorm_list = []
        L_list = []
        for i, loss in enumerate(args):
            grad = torch.autograd.grad(loss, self.shared_weights, retain_graph=True)[0]
            gradnorm = torch.norm(grad, 2) * self.params_list[i]
            gradnorm_list.append(gradnorm)
            with torch.no_grad():
                mv_factor = min(1 - 1/(self.global_step+1), self.beta)       
                self.mv_loss[i] = mv_factor * self.mv_loss[i] + (1 - mv_factor) * loss
                L = loss / self.mv_loss[i]
                L_list.append(L)

        with torch.no_grad():
            self.global_step += 1
            grad_norm_avg = sum(gradnorm_list) / len(gradnorm_list)
            L_avg = sum(L_list) / len(L_list)
            r_list = torch.tensor([(L / L_avg) ** self.alpha for L in L_list], requires_grad=False).to(grad_norm_avg.device)
            target_gradnorm = (grad_norm_avg * r_list).detach()

            self.task_difficulty = r_list

        # compute gradients
        L_grad = torch.sum(torch.abs(torch.stack(gradnorm_list) - target_gradnorm)) 
        grads = torch.autograd.grad(L_grad, self.params_list)[0]

        with torch.no_grad():
            # update gradients
            self.params_list.data = self.params_list.data - torch.tensor(self.lr) * grads
            for p in self.params_list:
                if p.data < 0.:
                    p.data.zero_()

            # renormalize
            coef =  12. / sum(self.params_list)
            self.params_list.data = self.params_list * coef