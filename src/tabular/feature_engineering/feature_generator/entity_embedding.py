import numpy as np
import pandas as pd
import json
import pickle

from keras.models import Model as KerasModel
from keras.layers import Input, Reshape, Concatenate, Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split


class NN_with_EntityEmbedding():

    def __init__(self, X, y, configger):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=True)
        self.configger = configger
        self.checkpointer = ModelCheckpoint(filepath=configger["check_point_file_path"], verbose=1)
        self.__build_keras_model(X)
        self.fit(X_train, y_train, X_val, y_val)

    def preprocessing(self, X):
        X_list = []
        for i, column in enumerate(X.columns):
            X_list.append(X[column].fillna(0).values)

        return X_list

    def __build_keras_model(self, X):
        input_model = []
        output_model = []

        for embed_col in self.configger["embed_cols"].keys():
            input_dim = len(np.unique(X[embed_col]))
            output_dim = self.configger["embed_cols"][embed_col]["output_dim"]

            input_numeric = Input(shape=(1,))
            embedding = Embedding(input_dim, output_dim, input_length=1,
                                  name="{col_name}_embedding".format(col_name=embed_col))(input_numeric)
            embedding = Reshape(target_shape=(output_dim,))(embedding)
            input_model.append(input_numeric)
            output_model.append(embedding)

        x = Concatenate()(output_model)
        x = Dense(32, activation='relu')(x)
        x = Dropout(.35)(x)
        x = Dense(16, activation='relu')(x)
        x = Dropout(.15)(x)
        x = Dense(8, activation='relu')(x)
        x = Dropout(.15)(x)
        output_model = Dense(1, activation='sigmoid')(x)

        self.model = KerasModel(inputs=input_model, outputs=output_model)
        self.model.compile(loss=self.configger["loss_func"], optimizer=self.configger["optimizer"])

    def fit(self, X_train, y_train, X_val, y_val):
        epochs = self.configger["epochs"]
        batch_size = self.configger["batch_size"]
        X_train = self.preprocessing(X_train)
        self.model.fit(X_train, y_train.values, epochs=epochs, batch_size=batch_size
                       # callbacks=[self.checkpointer]
                       )

    def save_embeddings(self, saved_embeddings_fname):
        embeddings = {}
        for embed_col in self.configger["embed_cols"].keys():
            layer_name = "{col_name}_embedding".format(col_name=embed_col)
            embeddings[layer_name] = self.model.get_layer(name=layer_name).get_weights()[0]

        with open(saved_embeddings_fname, 'wb') as f:
            pickle.dump(embeddings, f, -1)


def embedding_data(X, embeddings):
    """

    Parameters
    ----------
    X: pd.DataFrame. the original features of data.
    embeddings: str or dict. the file path of embedding file or embeddings dict object.

    Returns
    -------
    embedded_feature: pd.DataFrame. the embedded features, named like {col_name}_embedding
    """

    if isinstance(embeddings, str):
        f_embeddings = open(embeddings, "rb")
        embeddings = pickle.load(f_embeddings)

    X_columns = list(X.columns)

    X_embedded = []
    for i, record in X.iterrows():
        embedded_features = []
        print(record)
        for col in X_columns:
            feat = int(record[col])
            embedded_features += embeddings["{col_name}_embedding".format(col_name=col)][feat].tolist()

        X_embedded.append(embedded_features)

    res = np.array(X_embedded)
    names = ["feature_{i}".format(i=i) for i in range(np.shape(res)[1])]
    embedded_feature = pd.DataFrame(data=res, columns=names)

    return embedded_feature


def entity_embedding(df, configger):
    """

    Parameters
    ----------
    df: pd.DataFrame, the input dataframe.
    configger: str, the Json string of embedding setting.
        {
            "target_col":"${target_col}", #target_col name
            "embedding_file_path":"${embedding_file_path}" ,# the file path of embedding file result
            "check_point_file_path":"${check_point_file_path}", # the check point file path of embedding
            "loss_func":"${loss_func}",# the loss function of keras model training.
            "optimizer":"${optimizer_func}",# the optimize function of keras model training.
            "epochs":${epochs}, # the epoch or nn training
            "batch_size": ${batch_size}, # the batch_size or nn training
            "embed_cols":
                {
                "${col_name}":{"output_dim": ${output_dim_n}}, # the output dim number of the embed column for embedding layer.The input dim number is np.unique(df[col])
                ...
                }

        }

    Returns
    -------
    df_t: the result after embedding. the new features named {col_name}_embedding.
    """
    configger = json.loads(configger)
    target_col_name = configger["target_col"]
    embedding_file_path = configger["embedding_file_path"]

    y = df[target_col_name]
    X = df.drop([target_col_name], axis=1)

    NN = NN_with_EntityEmbedding(X, y, configger)
    NN.save_embeddings(embedding_file_path)

    df_t = embedding_data(X, embedding_file_path)
    df_t = pd.concat([df, df_t], axis=1)

    return df_t




