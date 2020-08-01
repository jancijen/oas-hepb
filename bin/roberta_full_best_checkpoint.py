import argparse
import torch
import os

if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser(description='Construct the best checkpoint file for a RoBERTa model trained on data without validation based on the previous training with validation.')
    parser.add_argument('--valid_best_checkpoint', action='store', help='Patht to the best checkpoint from the training with validation data.')
    parser.add_argument('--out_checkpoint_dir', action='store', help='Path to a directory containing the checkpoints from training without validation data.')
    parser.add_argument('--out_checkpoint_filename', action='store', help='Filename of the constructed output checkpoint.')

    args = parser.parse_args()

    print('Loading the best checkpoint from train+valid training...')
    best_checkpoint = torch.load(args.valid_best_checkpoint, map_location=torch.device('cpu'))
    best_epoch = best_checkpoint['extra_state']['train_iterator']['epoch']
    print(f'Best checkpoint - epoch {best_epoch}')
    
    print('Loading corresponding checkpoint from training without validation...')
    best_out_checkpoint = torch.load(os.path.join(args.out_checkpoint_dir, f'checkpoint{best_epoch}.pt'), map_location=torch.device('cpu'))
    
    print('Saving the output checkpoint...')
    torch.save(best_out_checkpoint, os.path.join(args.out_checkpoint_dir, args.out_checkpoint_filename))
