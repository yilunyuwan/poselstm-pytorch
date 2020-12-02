import numpy as np
import os.path
import transforms3d.quaternions as txq
import argparse

def params():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataroot', type=str, default='../datasets/7Scenes', help='dataset root')

    return parser.parse_args()
    
def is_pose_file(filename):
    return filename.endswith(EXTENSIONS) and filename.startswith(PREFIX)


def get_pose_filenames(dataset_dir, seq_dir):
    pose_filenames = []
    dir = os.path.join(dataset_dir, seq_dir)
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    fnames = sorted(os.listdir(dir))
    for fname in fnames:
        if is_pose_file(fname):
            pose_filenames.append(os.path.join(dir, fname))
    return pose_filenames

def convert_xyzquat(dataset_dir, split_file, write_file):
    with open(split_file, 'r') as f:
        seqs = [int(l.split('sequence')[-1]) for l in f if not l.startswith('#')]
    with open(write_file, 'w') as f:
        f.write('7 Scenes Datasets (convert rotation matrix to translation + quaternion)\n')
        f.write('Image File, Camera Position [X Y Z W P Q R]\n')
        f.write('\n')
        for seq in seqs:
            seq_dir = 'seq-{:02d}'.format(seq)
            p_filenames = get_pose_filenames(dataset_dir, seq_dir)
            assert p_filenames, 'no poses in directory {}'.format(seq_dir)
            ss = p_filenames[0].find(seq_dir)
            se = p_filenames[0].find('.pose')
            pose_out = np.zeros(7)
            for i in range(len(p_filenames)): 
                pose_in = np.loadtxt(p_filenames[i])
                pose_in = np.asarray(pose_in)    
                pose_out[3: ] = txq.mat2quat(pose_in[:3, :3])
                pose_out[0:3] = pose_in[:, 3].flatten()[:3]
                pose_str = p_filenames[i][ss:se] + '.color.png'
                for i in range(7):
                    pose_str += ' {:0.8f}'.format(pose_out[i])
                
                f.write(pose_str + '\n')


def split_train_test(dataset_dir, splitfns_in, splitfns_out):
    for i in range(len(splitfns_in)):
        split_in = splitfns_in[i]
        split_out = splitfns_out[i]
        split_file = os.path.join(dataset_dir, split_in)
        write_file = os.path.join(dataset_dir, split_out)
        if (not os.path.exists(split_file)):
            print('{} does not exist'.format(split_file))
            continue
        if (os.path.exists(write_file)):
            print('{} has existed'.format(write_file))
            continue
        print('start converting', split_file)
        convert_xyzquat(dataset_dir, split_file, write_file)
        print('finish converting', write_file)

EXTENSIONS = ('.txt')
PREFIX = ('frame')
args = params()
dataroot = args.dataroot
dataset_names = ['chess', 'fire', 'heads', 'office', 'pumpkin', 'redkitchen', 'stairs']
# dataset_names = ['chess']
splitfns_in = ['TrainSplit.txt', 'TestSplit.txt']
splitfns_out = ['dataset_train.txt', 'dataset_test.txt']
for name in dataset_names:
    dir = os.path.join(dataroot, name)
    print('processing', dir)
    split_train_test(dir, splitfns_in, splitfns_out)
  