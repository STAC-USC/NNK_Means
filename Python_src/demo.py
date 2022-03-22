__author__="shekkizh"
import os, argparse
from time import time

import numpy as np
import random
from numpy.random import RandomState


from nnk_model import NNK_Means, kmeans_plusplus

import torch
import torchvision.datasets 
import torchvision.transforms as transforms


def train(model, dataloader, args):
    for itr in range(args.epochs - 1):
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
            data, interpolated, label_interpolated = model(batch_x, batch_y,  update_cache=True, update_dict=False)
        model.update_dict()

    mse = 0
    acc = 0
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.cuda()
        batch_y = batch_y.cuda()
        data, interpolated, label_interpolated = model(batch_x, batch_y,  update_cache=True, update_dict=False)
        mse += torch.nn.functional.mse_loss(data, interpolated, reduction='sum')
        acc += torch.sum(100.0*(batch_y == torch.argmax(label_interpolated, 1)))
    
    return mse, acc

def evaluate(model, dataloader):
    mse = 0
    acc = 0
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.cuda()
        batch_y = batch_y.cuda()
        data, interpolated, label_interpolated = model(batch_x, batch_y,  update_cache=False, update_dict=False)
        mse += torch.nn.functional.mse_loss(data, interpolated, reduction='sum')
        acc += torch.sum(100.0*(batch_y == torch.argmax(label_interpolated, 1)))
    
    return mse, acc

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='NNK Means')
    parser.add_argument('--seed', default=4629, type=int, help='seed for initializing training. ')
    parser.add_argument('--epochs', default=0, type=int, help='number of epochs to train dictionary')
    parser.add_argument('-j', '--num_workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch_size', default=8192, type=int,
                        help='batch size (default: 256)')
    parser.add_argument("--data_dir", default="datasets/", help="dataset directory")
    parser.add_argument('--log_dir', default='logs/')
    args = parser.parse_args()
    rng = RandomState(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    
    log_dir = os.path.join( args.log_dir, f"epochs_{args.epochs}_seed_{args.seed}")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    data_transform = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.flatten(x))
    ])
    
    
    train_dataset = torchvision.datasets.MNIST(
            args.data_dir, train=True, transform=data_transform)
            
    image_shape = (28,28)
    X_torch = train_dataset.data.flatten(1).float().cuda()/255.0
    y = train_dataset.targets.cuda()
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, drop_last=False)
    
    top_k = 10
    n_components = 100
    
    
    n_train_points = len(train_dataset)
    init_indices = kmeans_plusplus(X_torch, n_components)
    
    
    model = NNK_Means(n_components, n_nonzero_coefs=top_k, n_classes=10, use_error_based_buffer=False, use_residual_update=False)
    model.initialize_dictionary(X_torch[init_indices], y[init_indices])
    t0 = time()
    mse, acc = train(model, train_loader, args)
    train_time = time() - t0
    
    
    mse /= n_train_points
    acc /= n_train_points

    print(f"#components:{n_components}, #nonzero:{top_k}  Train Error {mse:.2e}, Train Accuracy: {acc:0.2f} time: {train_time:0.3f}s")
    filename = os.path.join(log_dir, f"model_k_{top_k}_components_{n_components}.npz")
    np.savez_compressed(filename, atoms=model.dictionary_atoms.cpu().numpy(), labels=model.atom_labels.cpu().numpy(), error=mse.item(), accuracy=acc.item())


    test_dataset = torchvision.datasets.MNIST(
            args.data_dir, train=False, transform=data_transform)
    n_test_points = len(test_dataset)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, drop_last=False)

    mse, acc = evaluate(model, test_loader)
    mse /= n_test_points
    acc /= n_test_points
    print(f"#components:{n_components}, #nonzero:{top_k}  Test Error {mse:.2e}, Test Accuracy: {acc:0.2f}")
