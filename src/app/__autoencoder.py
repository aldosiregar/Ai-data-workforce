from tensorflow.keras.models import Sequential

from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model

from tensorflow.math import reduce_prod

from pandas import DataFrame

from sklearn.model_selection import train_test_split

#modifikasi class Model agar sesuai dengan pipeline autoencoder
class Autoencoder(Model):
    def __init__(self, latent_dim=int, to_shape=int, hidden_layer=[]):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        #encoder
        self.encoder = Sequential([
            layers.Flatten(),
            layers.Dense(latent_dim, activation="relu"),
            layers.Dropout(0.3)])
        
        for i in hidden_layer:
            self.encoder.add(layers.Dense(i, activation="relu")),
            self.encoder.add(layers.Dropout(0.3))
        
        self.encoder.add(layers.Dense(to_shape, activation="linear"))

        #decoder
        self.decoder = Sequential([
            layers.Dense(reduce_prod((to_shape,)).numpy(), activation="relu"),
            layers.Dropout(0.3)])
        
        index = len(hidden_layer) - 1
        for _ in range(len(hidden_layer)):
            self.decoder.add(layers.Dense(hidden_layer[index], activation="relu"))
            self.decoder.add(layers.Dropout(0.3))
            index -= 1

        self.decoder.add(layers.Dense(latent_dim, activation="linear"))
        self.decoder.add(layers.Reshape((latent_dim,)))

    """
    def call(self, x):
        encoded = self.encoder(x)
        return encoded
    """

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def check_performance(self,x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
        
    def get_result(self, x):
        encoded = self.encoder(x)
        return encoded
    
class Transformer:
    @staticmethod
    def transform(
        df=DataFrame([]), input_shape=100, 
        to_shape=1, test_size=0.3, hidden_layer=[],
        loss="", epoch=5) -> Autoencoder:
        #autoencoder ke 5 fitur
        x_train, x_test = train_test_split(df, test_size=test_size, random_state=43)

        autoencoder = Autoencoder(
            latent_dim=input_shape, to_shape=to_shape, hidden_layer=hidden_layer)

        if(loss == ""):
            autoencoder.compile(optimizer='adam', loss=losses.Huber())
        else:
            autoencoder.compile(optimizer='adam', loss=loss)

        autoencoder.fit(x_train, x_train,
                        epochs=epoch,
                        shuffle=True,
                        validation_data=(x_test, x_test))
        
        return autoencoder