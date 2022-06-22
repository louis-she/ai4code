import torch


class Adversarial:
    def __init__(self, model, optimizer, scaler):
        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler

    def save(self):
        pass

    def attack(self):
        pass

    def restore(self):
        pass


class AWP(Adversarial):
    def __init__(self, model, optimizer, scaler):
        super().__init__(model, optimizer, scaler)
        self.adv_param = "weight"
        self.adv_lr = 1
        self.backup_eps = {}
        self.backup = {}
        self.adv_eps = 0.2

    def attack_step(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(param.data, self.backup_eps[name][0]),
                        self.backup_eps[name][1],
                    )

    def save(self):
        for name, param in self.model.named_parameters():
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}


class FGM(Adversarial):

    def __init__(self, model, optimizer, scaler):
        super().__init__(model, optimizer, scaler)
        self.adv_param = "word_embeddings"
        self.adv_lr = 1
        self.backup = {}

    def save(self):
        for name, param in self.model.named_parameters():
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
                and name not in self.backup
            ):
                self.backup[name] = param.data.clone()

    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.adv_param in name:
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = self.adv_lr * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.adv_param in name:
                param.data = self.backup[name]
            self.backup = {}
