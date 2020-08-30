import tensorflow as tf
import argparse
import utils

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

saved_model_path = '{}/{}/{}'
path = saved_model_path.format(args.saved_model, args.dataset, args.nets)
# wandb.init(project="conv-nets", name=args.nets.lower())

model = utils.choose_nets(args.nets)

dataset_type = args.dataset
if dataset_type == 'cifar10':
    dataset = tf.keras.datasets.cifar10
elif dataset_type == 'cifar100':
    dataset = tf.keras.datasets.cifar100
else:
    raise ValueError('Please use cifar10 or cifar100 dataset')

(x_train, y_train), (x_test, y_test) = dataset.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(args.batch_size)
test_ds = tf.data.Dataset.from_tensor_slices(
    (x_test, y_test)).batch(args.batch_size)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(args.lr)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='train_accuracy')

validation_loss = tf.keras.metrics.Mean(name='validation_loss')
validation_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='validation_accuracy')


@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_accuracy(labels, predictions)
    return loss

@tf.function
def test_step(images, labels):   
    predictions = model(images)
    v_loss = loss_object(labels, predictions)
    validation_loss(v_loss)
    validation_accuracy(labels, predictions)

if args.ops == 'train':
    for epoch in range(args.epochs):
        for step, (images, labels) in enumerate(train_ds):
            loss = train_step(images, labels)
            # https://keras.io/guides/writing_a_training_loop_from_scratch/
            # Log every 200 batches.
            if step % 200 == 0:
                print("Training loss (for one batch) at step %d: %.4f" % (step, float(loss)))
                print("Seen so far: %d samples" % ((step + 1) * args.batch_size))

        model.save(path)
        
        train_template = 'Epoch: [{}/{}], Training Loss: {}, Training Accuracy: {}'
        print(train_template.format(epoch+1,
              args.epochs,
              train_loss.result(),
              train_accuracy.result()*100))   

        for step, (test_images, test_labels) in enumerate(test_ds):
            test_step(test_images, test_labels)

            if step % 200 == 0:
                print("Validated so far: %d samples" % ((step + 1) * args.batch_size))

        validation_template = 'Validation Loss: {}, Validation Accuracy: {}'
        print(validation_template.format(validation_loss.result(), validation_accuracy.result()*100))

        train_loss.reset_states()
        train_accuracy.reset_states()
        validation_loss.reset_states()
        validation_accuracy.reset_states()



if args.ops == 'test':
    print("Trained model is present at: ", path)
    model = tf.keras.models.load_model(path, compile=False)   
    print("Loaded model details:", args.nets, "for", args.dataset, "dataset")
    model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    print(model.summary())
    
    # Test model
    #test_loss, test_acc = model.evaluate(x_train, y_train, verbose=2)
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print("-" * 50)
    print('Restored model, accuracy: {:5.2f}%'.format(100*test_acc))

    for layer in model.layers: 
        print(layer.get_weights())

    last_layer_weights = model.layers[-1].get_weights()[0]
    last_layer_biases  = model.layers[-1].get_weights()[1]
    print ('The last layer weight is: ', last_layer_weights)
    print ('The last layer bias is: ', last_layer_biases)

    # wandb.log({
    #     "TrainLoss": train_loss.result(),
    #     "TestLoss": test_loss.result(),
    #     "TrainAcc": train_accuracy.result()*100,
    #     "TestAcc": test_accuracy.result()*100
    # })
