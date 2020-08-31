import tensorflow as tf
import argparse
import utils
import numpy as np
# import wandb

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

parser = argparse.ArgumentParser()
parser.add_argument('--nets', type=str, required=True)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--ops', type=str, default='train')
parser.add_argument('--saved_model', type=str, default='./saved_model')
parser.add_argument('--dataset', type=str, default='cifar10') # cifar10 | cifar100
args = parser.parse_args()

print(args)

saved_model_path = '{}/{}/{}.h5'
path = saved_model_path.format(args.saved_model, args.dataset, args.nets)
# wandb.init(project="conv-nets", name=args.nets.lower())

dataset_type = args.dataset

if dataset_type == 'cifar10':
    dataset = tf.keras.datasets.cifar10
    num_classes = 10
elif dataset_type == 'cifar100':
    dataset = tf.keras.datasets.cifar100
    num_classes = 100
else:
    raise ValueError('Please use cifar10 or cifar100 dataset')

print('No. of classes', num_classes)

model = utils.choose_nets(args.nets, num_classes)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

(x_train, y_train), (x_test, y_test) = dataset.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

print('Unique training labels : ',np.unique(y_train))

if args.ops == 'train':
    model.fit(x_train, y_train,
            validation_split=0.1,
            epochs=args.epochs,
            batch_size=args.batch_size)

    _, baseline_model_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print('Baseline test accuracy:', baseline_model_accuracy)
    model.save_weights(path)

if args.ops == 'test':
    print("Trained model is present at: ", path)
    #model = tf.keras.models.load_model(path, compile=True) 
    model.train_on_batch(x_train[:1], y_train[:1])

    model.load_weights(path)
    print("Loaded model details:", args.nets, "for", args.dataset, "dataset")
    print(model.summary())
    
    # Test model
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print("-" * 50)
    print('Restored model, accuracy: {:5.2f}%'.format(100*test_acc))

    #for layer in model.layers: 
        #print(layer.get_weights())

    last_layer_weights = model.layers[-1].get_weights()[0]
    last_layer_biases  = model.layers[-1].get_weights()[1]
    print ('The last layer weight shape is: ', np.shape(last_layer_weights))
    print ('The last layer bias shape is: ', np.shape(last_layer_biases))

    # wandb.log({
    #     "TrainLoss": train_loss.result(),
    #     "TestLoss": test_loss.result(),
    #     "TrainAcc": train_accuracy.result()*100,
    #     "TestAcc": test_accuracy.result()*100
    # })
