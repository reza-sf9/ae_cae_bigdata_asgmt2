# https://idiotdeveloper.com/building-convolutional-autoencoder-using-tensorflow-2/


import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, Activation, MaxPool2D
from tensorflow.keras.layers import BatchNormalization, Flatten, Reshape, Conv2DTranspose, LeakyReLU
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam, SGD
from IPython import display #

## configuration
tr_num = int(20e3)
te_num = int(4e3)

epoch_num = 25
latent_size = 2
batch_size = 256

## Seeding
np.random.seed(42)
tf.random.set_seed(42)

## Loading the dataset and then normalizing the images.
dataset = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = dataset.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_tr = x_train[0:tr_num, :, :]
y_tr = y_train[0:tr_num, ]

x_te = x_test[0:tr_num, :, :]
y_te = y_test[0:tr_num, ]

## Hyperparameters
H = 28
W = 28
C = 1





##########################
model_type =1
if model_type == 1: 
    units = 3136
    encoder = Sequential([
    Conv2D(32, (3, 3), padding="same"),
    BatchNormalization(),
    LeakyReLU(alpha=0.2),
    MaxPool2D((2, 2)),
    
    Conv2D(64, (3, 3), padding="same"),
    BatchNormalization(),
    LeakyReLU(alpha=0.2),
    MaxPool2D((2, 2)),
    
    Flatten(),
    
    Dense(latent_size, name="latent"),
    LeakyReLU(alpha=0.2)
    ])
    
    decoder = Sequential([
    Dense(units),
    LeakyReLU(alpha=0.2),
    Reshape((7, 7, 64)),
    
    Conv2DTranspose(64, (3, 3), strides=2, padding="same"),
    BatchNormalization(),
    LeakyReLU(alpha=0.2),
    
    Conv2DTranspose(1, (3, 3), strides=2, padding="same"),
    BatchNormalization(),
    Activation("sigmoid", name="outputs")
    ])
    
    img = Input(shape=(H, W, C), name="inputs")
    
    # getting encoder 
    latent_vector = encoder(img)
    output = decoder(latent_vector)
    
    
    model_cae = Model(inputs = img, outputs = output)
    model_cae.compile("nadam", loss = "binary_crossentropy")
    
    print("Fit CAE model on training data")
    loss_tr = np.empty(shape=(epoch_num, ))
    loss_te = np.empty(shape=(epoch_num, ))
    for epoch in range(epoch_num):
        fig, axs = plt.subplots(4, 4)
        rand = x_test[np.arange(16)].reshape((4, 4, 1, 28, 28))
        
        display.clear_output() # If you imported display from IPython
        
        for i in range(4):
            for j in range(4):
                axs[i, j].imshow(model_cae.predict(rand[i, j])[0], cmap = "gray")
                axs[i, j].axis("off")
        
        plt.subplots_adjust(wspace = 0, hspace = 0)
        str_save = 'results/task1_cae_epoch%d_latent%d.png' % (epoch, latent_size)
        plt.savefig(str_save)
        plt.show()
        print("-----------", "EPOCH", epoch, "-----------")
        history = model_cae.fit(x_tr, x_tr, batch_size=batch_size, validation_data=(x_te, x_te), shuffle=False)
        dict_loss  = history.history
        loss_tr[epoch] =  dict_loss['loss'][0]
        loss_te[epoch]  = dict_loss['val_loss'][0]
    
    
    # history = model_cae.fit(
    #     x_tr,
    #     x_tr,
    #     epochs= epoch_num,
    #     batch_size= batch_size,
    #     shuffle=False,
    #     validation_data=(x_te, x_te)
    # ) 

##########################

elif model_type == 2:
    inputs = Input(shape=(H, W, C), name="inputs")
    x = inputs
    
    x = Conv2D(32, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = MaxPool2D((2, 2))(x)
    
    x = Conv2D(64, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = MaxPool2D((2, 2))(x)
    
    x = Flatten()(x)
    units = x.shape[1]
    x = Dense(latent_size, name="latent")(x)
    x = LeakyReLU(alpha=0.2)(x)
    # outpu_latent = x 
    
    x = Dense(units)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Reshape((7, 7, 64))(x)
    
    x = Conv2DTranspose(64, (3, 3), strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    x = Conv2DTranspose(1, (3, 3), strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("sigmoid", name="outputs")(x)
    
    outputs = x
    
    model_cae = Model(inputs, outputs)
    model_cae.compile(optimizer=Adam(1e-3), loss='binary_crossentropy')
    model_cae.summary()
    
    print("Fit CAE model on training data")
    history = model_cae.fit(
        x_tr,
        x_tr,
        epochs= epoch_num,
        batch_size= batch_size,
        shuffle=False,
        validation_data=(x_te, x_te)
    )


    test_pred_y = model_cae.predict(x_te)
    
    n = 10  ## how many digits we will display
    plt.figure(figsize=(20, 4))
    for i in range(n):
        ## display original
        ax = plt.subplot(2, n, i + 1)
        ax.set_title("Original Image")
        plt.imshow(x_te[i].reshape(H, W))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    
        ## display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        ax.set_title("Predicted Image")
        plt.imshow(test_pred_y[i].reshape(H, W))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
    
    # plt.savefig("results/convolutonal_autoencoder.png")
    dict_loss  = history.history
    loss_tr = dict_loss['loss']
    loss_te = dict_loss['val_loss']
    
    plt.figure()
    plt.plot(loss_tr, label= 'train loss')
    plt.plot(loss_te, label='test loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss value')
    plt.title('CAE Model')
    plt.show()
    
    score_ = model_cae.evaluate(x_te, y_te)
    formatted_score = "{:.2f}".format(score_[1]*100)
    print('Accuracy: ', formatted_score)



plt.figure()
plt.plot(loss_tr, label= 'train loss')
plt.plot(loss_te, label='test loss')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('loss value')
plt.title('CAE Model')
str_save = 'results/task1_cae_loss_latent%d.png'%(latent_size)
plt.savefig(str_save)
plt.show()

k=1

x_tr_r = np.reshape(x_tr, (x_tr.shape[0], x_tr.shape[1], x_tr.shape[2], 1))
x_te_r = np.reshape(x_te, (x_te.shape[0], x_te.shape[1], x_te.shape[2], 1))
latent_tr = encoder(x_tr_r)
latent_te = encoder(x_te_r)

### k means 
do_kmeans = 1
if do_kmeans:
    from sklearn.cluster import KMeans
    n_clusters = len(np.unique(y_tr))

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

    cluster_labels = infer_cluster_labels(kmeans, y_tr)
    X_clusters = kmeans.predict(latent_tr)
    predicted_labels = infer_data_labels(X_clusters, cluster_labels)

    print(predicted_labels[:20])
    print(y_tr[:20])
    from sklearn import metrics

    acc_kmeans = metrics.accuracy_score(y_tr, predicted_labels)
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
        plt.scatter(latent_tr[:, 0], latent_tr[:, 1], c=y_tr, cmap=plt.cm.get_cmap("jet", 256))
        plt.colorbar(ticks=range(256))
        plt.title('train label')
        plt.clim(-0.5, 9.5)
        str_save = 'results/task2_cae_scatter.png'
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
            plt.scatter(x=tsne[:, 0], y=tsne[:, 1], c=y_tr, cmap=plt.cm.get_cmap("jet", 256))
            plt.colorbar(ticks=range(256))
            plt.title('train label')
            plt.clim(-0.5, 9.5)
            str_save = 'results/task2_cae_tsne_scatter.png'
            plt.savefig(str_save)
            plt.show()


    # read more about evaluation of kmeans

## classifier 
do_classifier = 1
if do_classifier:
    
    # one hot function 
    def to_onehot(y, num_classes=10):
        """Convert numpy array to one-hot."""
        onehot = np.zeros((len(y), num_classes))
        onehot[np.arange(len(y)), y] = 1
        return onehot


    y_tr_oh = to_onehot(y_tr)
    y_te_oh = to_onehot(y_te)


    classifier_units= 64
    num_classes = 10

    input_encoded = Input(shape=(latent_size,), name="input_encoding")
    clf_intermediate = Dense(classifier_units, activation='relu')(input_encoded)
    clf = Dense(num_classes, activation='softmax')(clf_intermediate)

    # Defining classifying model
    encoding_clf_model = Model(input_encoded, clf, name="encoder_classifier")
    encoding_clf_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    encoding_clf_model.summary()

    print("Fit model on training data")
    history = encoding_clf_model.fit(
        latent_tr,
        y_tr_oh,
        epochs= epoch_num,
        batch_size=batch_size,
        shuffle=False,
        validation_data=(latent_te, y_te_oh)
    )

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
    str_save = 'results/task3_cae_loss_latent%d.png'%(latent_size)
    plt.savefig(str_save)
    plt.show()


    plt.figure()
    plt.plot(acc_tr, label='train accuracy')
    plt.plot(acc_te, label='test accuracy')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('acc value')
    plt.title('Classifier Loss-  AE Model')
    str_save = 'results/task3_cae_acc_latent%d.png'%(latent_size)
    plt.savefig(str_save)
    plt.show()



    k=1
k=1
