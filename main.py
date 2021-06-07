import logging, os, numpy as np
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

from prettytable import PrettyTable

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tqdm import tqdm
from model import ResUnet
from loss import Tanimoto_dual_loss
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

patch_size = 256
batch_size = 8
num_classes = 6
epochs = 100

rows = patch_size
cols = patch_size
channels = 3

loss = Tanimoto_dual_loss()
loss_color = Tanimoto_dual_loss()
loss_reg = Tanimoto_dual_loss()

losses = {'seg': loss, 'bound': loss, 'dist': loss_reg, 'color': loss_reg}

lossWeights = {'seg': 1.0, 'bound': 1.0, 'dist': 1.0, 'color': 1.0}

root_path = './DATASETS/patch_size=256_stride=256_norm_type=1_data_aug=True'

model_save_path = './drive/MyDrive/results/'

def compute_mcc(tp, tn, fp, fn):
    mcc = (tp*tn - fp*fn) / tf.math.sqrt((tp + fp)*(tp + fn)*(tn + fp)*(tn+fn))
    return mcc

def train_model(net, x_train_paths, y_train_paths, x_val_paths,
                y_val_paths, batch_size, epochs, x_shape_batch, y_shape_batch,
                patience, delta, path, metrics_names=None):
    # patches_train = x_train_paths
    print('Start training...')
    print(f'Training on {len(x_train_paths)} images')
    print(f'Validating on {len(x_val_paths)} images')
    print(f'Total Epochs: {epochs}')

    # Initialize as maximum possible number
    min_loss = float('inf')
    cont = 0
    x_train_b = np.zeros(x_shape_batch, dtype=np.float32)
    y_train_h_b_seg = np.zeros(y_shape_batch, dtype=np.float32)
    x_val_b = np.zeros(x_shape_batch, dtype=np.float32)
    y_val_h_b_seg = np.zeros(y_shape_batch, dtype=np.float32)

    # Bounds
    y_train_h_b_bound = np.zeros(y_shape_batch, dtype=np.float32)
    y_val_h_b_bound = np.zeros(y_shape_batch, dtype=np.float32)
    # Dists
    y_train_h_b_dist = np.zeros(y_shape_batch, dtype=np.float32)
    y_val_h_b_dist = np.zeros(y_shape_batch, dtype=np.float32)
    # Colors
    y_train_h_b_color = np.zeros((y_shape_batch[0],
                                      y_shape_batch[1],
                                      y_shape_batch[2], 3),
                                     dtype=np.float32)
    y_val_h_b_color = np.zeros((y_shape_batch[0],
                                    y_shape_batch[1],
                                    y_shape_batch[2], 3),
                                   dtype=np.float32)

    for epoch in range(epochs):
        metrics_len = len(metrics_names)
        loss_tr = np.zeros((1, metrics_len))
        loss_val = np.zeros((1, metrics_len))
        # Computing the number of batchs on training
        n_batchs_tr = len(x_train_paths)//batch_size
        # Random shuffle the data

        (x_train_paths_rand, y_train_paths_rand_seg,
        y_train_paths_rand_bound, y_train_paths_rand_dist,
        y_train_paths_rand_color) \
        = shuffle(x_train_paths, y_train_paths[0], y_train_paths[1],
                       y_train_paths[2], y_train_paths[3])

        # Training the network per batch
        for batch in tqdm(range(n_batchs_tr), desc="Train"):
            x_train_paths_b = x_train_paths_rand[batch * batch_size:(batch + 1) * batch_size]
            y_train_paths_b_seg = y_train_paths_rand_seg[batch * batch_size:(batch + 1) * batch_size]

            for b in range(batch_size):
                x_train_b[b] = np.load(x_train_paths_b[b])
                y_train_h_b_seg[b] = np.load(y_train_paths_b_seg[b]).astype(np.float32)

            # Get paths per batch on multitasking labels
            y_train_paths_b_bound = y_train_paths_rand_bound[batch * batch_size:(batch + 1) * batch_size]
            y_train_paths_b_dist = y_train_paths_rand_dist[batch * batch_size:(batch + 1) * batch_size]
            y_train_paths_b_color = y_train_paths_rand_color[batch * batch_size:(batch + 1) * batch_size]
            
            # Load multitasking labels
            for b in range(batch_size):
                y_train_h_b_bound[b] = np.load(y_train_paths_b_bound[b]).astype(np.float32)
                y_train_h_b_dist[b] = np.load(y_train_paths_b_dist[b]).astype(np.float32)
                y_train_h_b_color[b] = np.load(y_train_paths_b_color[b]).astype(np.float32)

            y_train_b = {"seg": y_train_h_b_seg}
            y_train_b['bound'] = y_train_h_b_bound
            y_train_b['dist'] = y_train_h_b_dist
            y_train_b['color'] = y_train_h_b_color

            loss_tr += net.train_on_batch(x=x_train_b, y=y_train_b, return_dict=False)

        loss_tr /= n_batchs_tr

        # Computing the number of batchs on validation
        n_batchs_val = len(x_val_paths)//batch_size

        # Evaluating the model in the validation set
        for batch in tqdm(range(n_batchs_val), desc="Validation"):
            x_val_paths_b = x_val_paths[batch * batch_size:(batch + 1) * batch_size]
            y_val_paths_b_seg = y_val_paths[0][batch * batch_size:(batch + 1) * batch_size]

            for b in range(batch_size):
                x_val_b[b] = np.load(x_val_paths_b[b])
                y_val_h_b_seg[b] = np.load(y_val_paths_b_seg[b]).astype(np.float32)


            y_val_paths_b_bound = y_val_paths[1][batch * batch_size:(batch + 1) * batch_size]
            y_val_paths_b_dist = y_val_paths[2][batch * batch_size:(batch + 1) * batch_size]
            y_val_paths_b_color = y_val_paths[3][batch * batch_size:(batch + 1) * batch_size]
                
            # Load multitasking labels
            for b in range(batch_size):
                y_val_h_b_bound[b] = np.load(y_val_paths_b_bound[b]).astype(np.float32)
                y_val_h_b_dist[b] = np.load(y_val_paths_b_dist[b]).astype(np.float32)
                y_val_h_b_color[b] = np.load(y_val_paths_b_color[b]).astype(np.float32)
                # Dict template: y_val_b = {"segmentation": y_val_h_b_seg,
                # "boundary": y_val_h_b_bound, "distance":  y_val_h_b_dist,
                # "color": y_val_h_b_color}
            y_val_b = {"seg": y_val_h_b_seg}
            y_val_b['bound'] = y_val_h_b_bound
            y_val_b['dist'] = y_val_h_b_dist
            y_val_b['color'] = y_val_h_b_color

            loss_val = loss_val + net.test_on_batch(x=x_val_b, y=y_val_b)
        loss_val = loss_val/n_batchs_val

        train_metrics = dict(zip(metrics_names, loss_tr.tolist()[0]))
        val_metrics = dict(zip(metrics_names, loss_val.tolist()[0]))

        mcc = compute_mcc(val_metrics['seg_true_positives'],
                              val_metrics['seg_true_negatives'],
                              val_metrics['seg_false_positives'],
                              val_metrics['seg_false_negatives'])

        metrics_table = PrettyTable()
        metrics_table.title = f'Epoch: {epoch}'
        metrics_table.field_names = ['Task', 'Loss', 'Val Loss',
                                         'Acc %', 'Val Acc %']

        metrics_table.add_row(['Seg', round(train_metrics['seg_loss'], 5),
                                  round(val_metrics['seg_loss'], 5),
                                  round(100*train_metrics['seg_accuracy'], 5),
                                  round(100*val_metrics['seg_accuracy'], 5)])


        metrics_table.add_row(['Bound',
                                   round(train_metrics['bound_loss'], 5),
                                  round(val_metrics['bound_loss'], 5),
                                  0, 0])


        metrics_table.add_row(['Dist',
                                   round(train_metrics['dist_loss'], 5),
                                   round(val_metrics['dist_loss'], 5),
                                   0, 0])


        metrics_table.add_row(['Color',
                                   round(train_metrics['color_loss'], 5),
                                   round(val_metrics['color_loss'], 5),
                                   0, 0])


        metrics_table.add_row(['Total', round(train_metrics['loss'], 5),
                                  round(val_metrics['loss'], 5),
                                  0, 0])

        val_loss = val_metrics['loss']
        print(metrics_table)
        # Early stop
        # Save the model when loss is minimum
        # Stop the training if the loss don't decreases after patience epochs
        if val_loss >= min_loss + delta:
            cont += 1
            print(f'EarlyStopping counter: {cont} out of {patience}')
            if cont >= patience:
                print("Early Stopping! \t Training Stopped")
                return net
        else:
            cont = 0
            min_loss = val_loss
            print("Saving best model...")
            net.save(os.path.join(path, 'best_model.h5'))



train_path = os.path.join(root_path, 'train')
patches_tr = [os.path.join(train_path, name)
                  for name in os.listdir(train_path)]

ref_path = os.path.join(root_path, 'labels/seg')
patches_tr_lb_h = [os.path.join(ref_path, name) for name
                       in os.listdir(ref_path)]

ref_bound_path = os.path.join(root_path, 'labels/bound')
patches_bound_labels = [os.path.join(ref_bound_path, name) for name
                                in os.listdir(ref_bound_path)]

ref_dist_path = os.path.join(root_path, 'labels/dist')
patches_dist_labels = [os.path.join(ref_dist_path, name) for name
                               in os.listdir(ref_dist_path)]

ref_color_path = os.path.join(root_path, 'labels/color')
patches_color_labels = [os.path.join(ref_color_path, name) for name
                                in os.listdir(ref_color_path)]

patches_tr, patches_val, patches_tr_lb_h, patches_val_lb_h, patches_bound_labels_tr, patches_bound_labels_val, patches_dist_labels_tr, patches_dist_labels_val, patches_color_labels_tr, patches_color_labels_val   = train_test_split(patches_tr, patches_tr_lb_h, patches_bound_labels, patches_dist_labels, patches_color_labels,  test_size=0.2, random_state=42)

y_paths = [patches_tr_lb_h, patches_bound_labels_tr,
                   patches_dist_labels_tr, patches_color_labels_tr]

val_paths = [patches_val_lb_h, patches_bound_labels_val,
                     patches_dist_labels_val, patches_color_labels_val]


resunet_a = ResUnet(num_classes, (patch_size,patch_size,channels))
model = resunet_a.build_model()
# model.summary()

metrics_dict = {'seg': ['accuracy', tf.keras.metrics.TruePositives(),
                                  tf.keras.metrics.FalsePositives(),
                                  tf.keras.metrics.TrueNegatives(),
                                  tf.keras.metrics.FalseNegatives()]}
              
model.compile(optimizer=Adam(), loss=losses, metrics=metrics_dict)

x_shape_batch = (batch_size, patch_size, patch_size, channels)
y_shape_batch = (batch_size, patch_size, patch_size, num_classes)

metrics_names = ['loss', 'seg_loss', 'bound_loss', 'dist_loss',
                         'color_loss', 'seg_accuracy', 'seg_true_positives',
                         'seg_false_positives', 'seg_true_negatives',
                         'seg_false_negatives']


train_model(model, patches_tr, y_paths, patches_val, val_paths,
                    batch_size, epochs,
                    x_shape_batch=x_shape_batch, y_shape_batch=y_shape_batch, patience=10, delta=0.001,
                    metrics_names=metrics_names, path=model_save_path)


