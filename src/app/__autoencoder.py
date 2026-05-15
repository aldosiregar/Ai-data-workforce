from tensorflow.keras.models import Sequential

from tensorflow.keras import layers, losses, callbacks, optimizers, regularizers
from tensorflow.keras.models import Model

from tensorflow.math import reduce_prod

from pandas import DataFrame

from sklearn.model_selection import train_test_split

#modifikasi class Model agar sesuai dengan pipeline autoencoder
class Autoencoder(Model):
    def __init__(self, input_dim=int, latent_dim=int, 
                 hidden_layer=[], l2_strength=1e-5):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        reg = regularizers.l2(l2_strength)
        #encoder
        self.encoder = Sequential([
            layers.Input(shape=(input_dim,))])
        
        for i in hidden_layer:
            self.encoder.add(layers.Dense(
                i, activation="relu", kernel_regularizer=reg))

        self.encoder.add(layers.Dense(latent_dim, activation="linear"))

        #decoder
        self.decoder = Sequential([
            layers.Input(shape=(latent_dim,))])
        
        index = len(hidden_layer) - 1
        for _ in range(len(hidden_layer)):
            self.decoder.add(
                layers.Dense(
                    hidden_layer[index], activation="relu", kernel_regularizer=reg))
            index -= 1

        self.decoder.add(layers.Dense(latent_dim, activation="linear"))

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
        to_shape=1, test_size=0.25, hidden_layer=[],
        loss="", epoch=5) -> Autoencoder:
        x_train, x_test = train_test_split(df, test_size=test_size, random_state=43)

        autoencoder = Autoencoder(
            latent_dim=input_shape, to_shape=to_shape, hidden_layer=hidden_layer)

        if(loss == ""):
            autoencoder.compile(
                optimizer=optimizers.Adam(learning_rate=lr_schedule), 
                loss=losses.Huber(delta=1.0),
                metrics=["mae", "mse"])
        else:
            autoencoder.compile(
                optimizer=optimizers.Adam(learning_rate=lr_schedule), 
                loss=loss,
                metrics=["mae", "mse"])
        
        callbacks = [
            callbacks.EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True,
                verbose=1
            )
        ]

        total_steps = (x_train.shape[0] // 1024) * epoch
        lr_schedule = optimizers.schedules.CosineDecay(
            initial_learning_rate=1e-3,
            decay_steps=total_steps,
            alpha=1e-6
        )

        autoencoder.fit(
            x_train, x_train,epochs=epoch,
            shuffle=True,batch_size=1024,
            validation_split=0.1,callbacks=callbacks)
                
        return autoencoder