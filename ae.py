# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 12:57:15 2021

@author: rsaadatifard
https://towardsdatascience.com/how-to-make-an-autoencoder-2f2d99cd5103
"""

## import packages 

from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Input, Flatten,\
                                    Reshape, LeakyReLU ,\
                                    Activation, Dropout
from tensorflow.keras.models import Model, Sequential
from matplotlib import pyplot as plt
from IPython import display # If using IPython, Colab or Jupyter
import numpy as np
from sklearn.cluster import KMeans

## configuration
tr_num = int(20e3)
te_num = int(4e3)

epoch_num = 25
latent_size = 32
batch_size = 256

## loading MNIST data set 
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train/255.0
x_test = x_test/255.0





x_train = x_train[0:tr_num, :, :]
y_train = y_train[0:tr_num, ]

x_test = x_test[0:tr_num, :, :]
y_test = y_test[0:tr_num, ]


## plot baseline images
plt_baseline_img = 0
if plt_baseline_img:
    fig, axs = plt.subplots(4, 4)
    rand = x_test[np.arange(16)].reshape((4, 4, 1, 28, 28))

    display.clear_output()  # If you imported display from IPython

    for i in range(4):
        for j in range(4):
            axs[i, j].imshow(rand[i, j, 0, :, :], cmap="gray")
            axs[i, j].axis("off")

    plt.subplots_adjust(wspace=0, hspace=0)
    str_save = 'results/task1_baseline_img.png'
    plt.savefig(str_save)
    plt.show()

## hyperparameters 

alpha_lk_relu = 0.2 


## encoder structure 
encoder = Sequential([
    Flatten(input_shape = (28, 28)),
    Dense(512),
    LeakyReLU(alpha= alpha_lk_relu),
    Dropout(0.5),
    Dense(256),
    LeakyReLU(alpha= alpha_lk_relu),
    Dropout(0.5),
    Dense(128),
    LeakyReLU(alpha= alpha_lk_relu),
    Dropout(0.5),
    Dense(64),
    LeakyReLU(alpha= alpha_lk_relu),
    Dropout(0.5),
    Dense(latent_size),
    LeakyReLU(alpha= alpha_lk_relu)
])

## decoder structure 
decoder = Sequential([
    Dense(64, input_shape = (latent_size,)),
    LeakyReLU(alpha= alpha_lk_relu),
    Dropout(0.5),
    Dense(128),
    LeakyReLU(alpha= alpha_lk_relu),
    Dropout(0.5),
    Dense(256),
    LeakyReLU(alpha= alpha_lk_relu),
    Dropout(0.5),
    Dense(512),
    LeakyReLU(alpha= alpha_lk_relu),
    Dropout(0.5),
    Dense(784),
    Activation("sigmoid"),
    Reshape((28, 28))
])

## creating the model 

# This will create a placeholder tensor which we can feed into each network to get the output of the whole model.
img = Input(shape = (28, 28))


# getting encoder 
latent_vector = encoder(img)
output = decoder(latent_vector)


model_ae = Model(inputs = img, outputs = output)
model_ae.compile("nadam", loss = "binary_crossentropy")

## training the model 

loss_tr = np.empty(shape=(epoch_num, ))
loss_te = np.empty(shape=(epoch_num, ))
for epoch in range(epoch_num):
    fig, axs = plt.subplots(4, 4)
    rand = x_test[np.arange(16)].reshape((4, 4, 1, 28, 28))
    
    display.clear_output() # If you imported display from IPython
    
    for i in range(4):
        for j in range(4):
            axs[i, j].imshow(model_ae.predict(rand[i, j])[0], cmap = "gray")
            axs[i, j].axis("off")
    
    plt.subplots_adjust(wspace = 0, hspace = 0)
    str_save = 'results/task1_ae_epoch%d_latent%d.png' % (epoch, latent_size)
    plt.savefig(str_save)
    plt.show()

    print("-----------", "EPOCH", epoch, "-----------")
    history = model_ae.fit(x_train, x_train, batch_size=batch_size, validation_data=(x_test, x_test), shuffle=False)
    dict_loss  = history.history
    loss_tr[epoch] =  dict_loss['loss'][0]
    loss_te[epoch]  = dict_loss['val_loss'][0]
    k=1
    

# dict_loss  = history.history
# loss_tr = dict_loss['loss']
# loss_te = dict_loss['val_loss']

plt.figure()
plt.plot(loss_tr, label= 'train loss')
plt.plot(loss_te, label='test loss')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('loss value')
plt.title('AE  - classic AE Model')
str_save = 'results/task1_ae_loss_latent%d.png'%(latent_size)
plt.savefig(str_save)
plt.show()


latent_tr = encoder(x_train)
latent_te = encoder(x_test)

do_kmeans = 1
if do_kmeans:

    n_clusters = len(np.unique(y_train))

    km = KMeans(n_clusters=n_clusters, n_init=20).fit_predict(latent_tr)

    ##  Assigning Cluster Labels

    from sklearn.cluster import MiniBatchKMeans

    kmeans = MiniBatchKMeans(n_clusters=n_clusters)
    kmeans.fit(latent_tr)


    ## function to reassign in corect way
    def infer_cluster_labels(kmeans, actual_labels):
        inferred_labels = {}


        for i in range(kmeans.n_clusters):

            # find index of points in cluster
            labels = []
            index = np.where(kmeans.labels_ == i)

            # append actual labels for each point in cluster
            labels.append(actual_labels[index])

            # determine most common label
            if len(labels[0]) == 1:
                counts = np.bincount(labels[0])
            else:
                counts = np.bincount(np.squeeze(labels))

            # assign the cluster to a value in the inferred_labels dictionary
            if np.argmax(counts) in inferred_labels:
                # append the new number to the existing array at this slot
                inferred_labels[np.argmax(counts)].append(i)
            else:
                # create a new array in this slot
                inferred_labels[np.argmax(counts)] = [i]

            # print(labels)
            # print('Cluster: {}, label: {}'.format(i, np.argmax(counts)))

        return inferred_labels


    def infer_data_labels(X_labels, cluster_labels):
        # empty array of len(X)
        predicted_labels = np.zeros(len(X_labels)).astype(np.uint8)

        for i, cluster in enumerate(X_labels):
            for key, value in cluster_labels.items():
                if cluster in value:
                    predicted_labels[i] = key

        return predicted_labels

    # test the infer_cluster_labels() and infer_data_labels() functions

    cluster_labels = infer_cluster_labels(kmeans, y_train)
    X_clusters = kmeans.predict(latent_tr)
    predicted_labels = infer_data_labels(X_clusters, cluster_labels)

    print(predicted_labels[:20])
    print(y_train[:20])
    from sklearn import metrics

    acc_kmeans = metrics.accuracy_score(y_train, predicted_labels)
    print('accuracy kmeans = %.2f = '% (100*acc_kmeans))
    # VISUALIZATON
    if latent_size==2:
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.scatter(latent_tr[:, 0], latent_tr[:, 1], c=predicted_labels, cmap=plt.cm.get_cmap("jet", 256))
        plt.colorbar(ticks=range(256))
        plt.clim(-0.5, 9.5)
        plt.title('kmeans')
        # plt.show()

        plt.subplot(1, 2, 2)
        plt.scatter(latent_tr[:, 0], latent_tr[:, 1], c=y_train, cmap=plt.cm.get_cmap("jet", 256))
        plt.colorbar(ticks=range(256))
        plt.title('train label')
        plt.clim(-0.5, 9.5)
        str_save = 'results/task2_ae_scatter.png'
        plt.savefig(str_save)
        plt.show()

        tnse_ = 0
        if tnse_:
            from sklearn.manifold import TSNE

            tsne = TSNE().fit_transform(latent_tr)

            plt.figure()
            plt.subplot(1, 2, 1)
            plt.scatter(x=tsne[:, 0], y=tsne[:, 1], c=km, cmap=plt.cm.get_cmap("jet", 256))
            plt.colorbar(ticks=range(256))
            plt.clim(-0.5, 9.5)
            plt.title('kmeans')

            plt.subplot(1, 2, 2)
            plt.scatter(x=tsne[:, 0], y=tsne[:, 1], c=y_train, cmap=plt.cm.get_cmap("jet", 256))
            plt.colorbar(ticks=range(256))
            plt.title('train label')
            plt.clim(-0.5, 9.5)
            str_save = 'results/task2_ae_tsne_scatter.png'
            plt.savefig(str_save)
            plt.show()


    # read more about evaluation of kmeans

do_classifier = 1
if do_classifier:
    
    # one hot function 
    def to_onehot(y, num_classes=10):
        """Convert numpy array to one-hot."""
        onehot = np.zeros((len(y), num_classes))
        onehot[np.arange(len(y)), y] = 1
        return onehot


    y_train_oh = to_onehot(y_train)
    y_test_oh = to_onehot(y_test)


    classifier_units= 64
    num_classes = 10

    input_encoded = Input(shape=(latent_size,), name="input_encoding")
    clf_intermediate = Dense(classifier_units, activation='relu')(input_encoded)
    clf = Dense(num_classes, activation='softmax')(clf_intermediate)

    # Defining classifying model
    encoding_clf_model = Model(input_encoded, clf, name="encoder_classifier")
    encoding_clf_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    # encoding_clf_model.summary()

    print("Fit model on training data")
    history = encoding_clf_model.fit(
        latent_tr,
        y_train_oh,
        epochs= epoch_num,
        batch_size=batch_size,
        shuffle=False,
        validation_data=(latent_te, y_test_oh)
    )

    # encoding_clf_model.fit()

    dict_loss  = history.history
    loss_tr = dict_loss['loss']
    loss_te = dict_loss['val_loss']

    acc_tr = dict_loss['categorical_accuracy']
    acc_te = dict_loss['val_categorical_accuracy']

    plt.figure()
    plt.plot(loss_tr, label='train loss')
    plt.plot(loss_te, label='test loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss value')
    plt.title('Classifier Loss-  AE Model')
    str_save = 'results/task3_ae_loss_latent%d.png'%(latent_size)
    plt.savefig(str_save)
    plt.show()


    plt.figure()
    plt.plot(acc_tr, label='train accuracy')
    plt.plot(acc_te, label='test accuracy')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('acc value')
    str_tit = 'Classifier Loss-  AE Model - train acc=%.2f -- test acc=%.2f '%(acc_tr[-1], acc_te[-1])
    plt.title(str_tit)
    str_save = 'results/task3_ae_acc_latent%d.png'%(latent_size)
    plt.savefig(str_save)
    plt.show()



    k=1
k=1

