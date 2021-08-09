import argparse
import random
import numpy as np
import torch.backends.cudnn as cudnn
import torch
from opacus import PrivacyEngine
from train_utils import get_device, train, test
from data import get_data, random_sample_loader
from models import MODELS
from dp_utils import ORDERS, get_privacy_spent, get_renyi_divergence


seed = 1234
cudnn.benchmark = True
cudnn.enabled = True
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

def main(dataset, batch_size=2048, mini_batch_size=256, lr=1,
         noise_multiplier=1, max_grad_norm=0.1, epochs=100, max_epsilon=10.,):
    device = get_device()

    bs = batch_size
    assert bs % mini_batch_size == 0
    n_acc_steps = bs // mini_batch_size

    train_data, test_data = get_data(dataset, augment=False)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=mini_batch_size, shuffle=True, num_workers=1, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=mini_batch_size, shuffle=False, num_workers=1, pin_memory=True)
    train_loader = random_sample_loader(train_loader, device, drop_last=True)
    test_loader = random_sample_loader(test_loader, device)

    model = MODELS[dataset]()
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    privacy_engine = PrivacyEngine(
        module=model,
        batch_size=bs,
        sample_size=len(train_data),
        alphas=ORDERS,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    )
    privacy_engine.attach(optimizer)

    results = []
    for epoch in range(0, epochs):
        print(f"\nEpoch: {epoch}")
        train_loss, train_acc = train(model, train_loader, optimizer, n_acc_steps=n_acc_steps)
        test_loss, test_acc = test(model, test_loader)

        if noise_multiplier > 0:
            rdp_sgd = get_renyi_divergence(
                privacy_engine.sample_rate, privacy_engine.noise_multiplier
            ) * privacy_engine.steps
            epsilon, _ = get_privacy_spent(rdp_sgd)
            print(f"Privacy cost: ε = {epsilon:.3f}")
            if max_epsilon is not None and epsilon >= max_epsilon:
                return
        else:
            epsilon = None
        results.append([epsilon, test_acc])

    results = np.array(results)
    print('\n'+'='*60)
    print('Best test accuracy: %.2f for privacy budget ε = %.2f'%(results[:,1].max(), results[-1, 0]))
    print('='*60)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['cifar10', 'fmnist', 'mnist'])
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--mini_batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=2)
    parser.add_argument('--noise_multiplier', type=float, default=1)
    parser.add_argument('--max_grad_norm', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=40)
    args = parser.parse_args()
    print(args)
    main(**vars(args))
