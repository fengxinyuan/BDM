import torch
import copy


class BDM(torch.nn.Module):
    def __init__(self, encoder, predictor, discriminator):
        super().__init__()
        # main GNN encoder
        self.main_encoder = copy.deepcopy(encoder)
        # auxiliary GNN encoder
        self.auxiliary_encoder = encoder
        self.mlp_predictor = predictor
        self.discriminator = discriminator

        # reinitialize weights
        self.main_encoder.reset_parameters()
        # stop gradient
        for param in self.main_encoder.parameters():
            param.requires_grad = False

    def trainable_parameters(self):
        return list(self.auxiliary_encoder.parameters()) + list(self.mlp_predictor.parameters()) + list(
            self.discriminator.parameters())

    @torch.no_grad()
    def update_target_network(self, mm):
        assert 0.0 <= mm <= 1.0, "Momentum needs to be between 0.0 and 1.0(which need to get %.5f)" % mm
        for param_q, param_k in zip(self.auxiliary_encoder.parameters(), self.main_encoder.parameters()):
            param_k.data.mul_(mm).add_(param_q.data, alpha=1. - mm)

    def forward(self, self_x, pos_x):
        # forward auxiliary GNN
        self_y = self.auxiliary_encoder(self_x)
        # MLP_prediction
        self_q = self.mlp_predictor(self_y)
        # forward target network
        with torch.no_grad():
            pos_y = self.main_encoder(pos_x).detach()
        return self_q, pos_y


def load_trained_encoder(encoder, ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location=device)
    encoder.load_state_dict(checkpoint['model'], strict=True)
    return encoder.to(device)


def compute_representations(net, dataset, device):
    net.eval()
    reps = []
    labels = []

    for data in dataset:
        # forward
        data = data.to(device)
        with torch.no_grad():
            reps.append(net(data))
            labels.append(data.y)

    reps = torch.cat(reps, dim=0)
    labels = torch.cat(labels, dim=0)
    return [reps, labels]


