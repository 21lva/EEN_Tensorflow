from __future__ import division
import os, argparse
import _pickle as cpickle
import tensorflow as tf
import numpy as np

tfe=tf.contrib.eager
tf.enable_eager_execution()

class DeterministicNetwork(tf.keras.Model):
    def __init__(self,opt,parser):
        super(DeterministicNetwork,self).__init__()
        self.opt = opt
        self.parser=parser
        self.ncond = parser.ncond
        self.npred = parser.npred
        self.nfeature = parser.nfeature
        self.height = parser.height
        self.width = parser.width
        self.batch_size = parser.batch_size
        self.nc = parser.nc

        self.DEncoder = tf.keras.models.Sequential(
                    [
                        #tf.keras.layers.InputLayer(input_shape=(64,4,1,84,84)),
                        tf.keras.layers.Reshape((self.nc*self.ncond,self.height,self.width)),
                        tf.keras.layers.Permute((2,3,1)),
                        tf.keras.layers.Conv2D(self.nfeature,7,3,"same"),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.ReLU(),
                        tf.keras.layers.Conv2D(self.nfeature,5,2,"same"),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.ReLU(),
                        tf.keras.layers.Conv2D(self.nfeature,3,2,"same"),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.ReLU()
                    ]
                )
        self.DDecoder = tf.keras.models.Sequential(
                    [
                        tf.keras.layers.Conv2DTranspose(self.nfeature,7,3,"same"),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.ReLU(),
                        tf.keras.layers.Conv2DTranspose(self.nfeature,3,2,"same"),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.ReLU(),
                        tf.keras.layers.Conv2DTranspose(self.npred*self.nc,3,2,"same"),
                        tf.keras.layers.Permute((3,1,2)),
                        tf.keras.layers.Reshape((self.nc*self.npred,1,self.height,self.width))
                    ]
                )

    def predict(self,Input):
        return self.DDecoder(self.DEncoder(Input))

    def compute_loss(self,Input,Target):
        #Predict = tf.layers.Flatten()(self.predict(Input))
        Predict = self.predict(Input)
        #self.DEncoder.summary()
        #self.DDecoder.summary()
        #Target = tf.layers.Flatten()(Target)
        #los = tf.keras.metrics.mse(Target,Predict)
        #los = tf.nn.l2_loss((Target-Predict))
        los = tf.losses.mean_squared_error(Target,Predict)
        return los
    
    def compute_gradients(self,Input,Target):
        with tf.GradientTape() as tape:
            L = self.compute_loss(Input,Target)
        return tape.gradient(L,self.trainable_variables),L

    def apply_gradients(self,grad,global_step=None):
        self.opt.apply_gradients(zip(grad,self.trainable_variables),global_step=global_step)
   

class LatentNetwork(tf.keras.Model):
    def __init__(self,opt,parser):
        super(LatentNetwork,self).__init__()
        self.opt = opt
        self.parser=parser
        self.ncond = parser.ncond
        self.npred = parser.npred
        self.nfeature = parser.nfeature
        self.height = parser.height
        self.width = parser.width
        self.batch_size = parser.batch_size
        self.nc = parser.nc
        
        self.nLatent = parser.n_latent
        self.phi_fc_size = parser.phi_fc_size

        self.DEncoder = tf.keras.models.Sequential(
                    [
                        #tf.keras.layers.InputLayer(input_shape=(64,4,1,84,84)),
                        tf.keras.layers.Reshape((self.nc*self.ncond,self.height,self.width)),
                        tf.keras.layers.Permute((2,3,1)),
                        tf.keras.layers.Conv2D(self.nfeature,7,3,"same"),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.ReLU(),
                        tf.keras.layers.Conv2D(self.nfeature,5,2,"same"),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.ReLU(),
                        tf.keras.layers.Conv2D(self.nfeature,3,2,"same"),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.ReLU()
                        ],name="DeterministicEncoder"
                )
        self.DDecoder = tf.keras.models.Sequential(
                    [
                        tf.keras.layers.Conv2DTranspose(self.nfeature,7,3,"same"),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.ReLU(),
                        tf.keras.layers.Conv2DTranspose(self.nfeature,3,2,"same"),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.ReLU(),
                        tf.keras.layers.Conv2DTranspose(self.npred*self.nc,3,2,"same"),
                        tf.keras.layers.Permute((3,1,2)),
                        tf.keras.layers.Reshape((self.nc*self.npred,1,self.height,self.width))
                    ],name="DeterministicDecoder"

                )
        self.phi_conv = tf.keras.models.Sequential(
                [
                    tf.keras.layers.Reshape((self.nc*self.ncond,self.height,self.width)),
                    tf.keras.layers.Permute((2,3,1)),
                    tf.keras.layers.Conv2D(self.nfeature,7,3,"same"),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.ReLU(),
                    tf.keras.layers.Conv2D(self.nfeature,5,2,"same"),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.ReLU(),
                    tf.keras.layers.Conv2D(self.nfeature,5,2,"same"),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.ReLU(),
                    tf.keras.layers.Permute((3,1,2)),
                    tf.keras.layers.Flatten()
                ],name="phi_convolution"

            )
        self.phi_f = tf.keras.models.Sequential(
                [
                    tf.keras.layers.Dense(1000),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.ReLU(),
                    tf.keras.layers.Dense(1000),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.ReLU(),
                    tf.keras.layers.Dense(self.nLatent,activation="tanh")
                ],name="phi_feedforward"

           )
        self.LEncoder = tf.keras.models.Sequential(
                    [
                        #tf.keras.layers.InputLayer(input_shape=(64,4,1,84,84)),
                        tf.keras.layers.Reshape((self.nc*self.ncond,self.height,self.width)),
                        tf.keras.layers.Permute((2,3,1)),
                        tf.keras.layers.Conv2D(self.nfeature,7,3,"same"),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.ReLU(),
                        tf.keras.layers.Conv2D(self.nfeature,5,2,"same"),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.ReLU(),
                        tf.keras.layers.Conv2D(self.nfeature,3,2,"same"),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.ReLU()
                    ],name="LatentEncoder"

                )
        self.wz = tf.keras.layers.Dense(self.nfeature)
        self.LDecoder = tf.keras.models.Sequential(
                    [
                        tf.keras.layers.Conv2DTranspose(self.nfeature,7,3,"same"),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.ReLU(),
                        tf.keras.layers.Conv2DTranspose(self.nfeature,3,2,"same"),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.ReLU(),
                        tf.keras.layers.Conv2DTranspose(self.npred*self.nc,3,2,"same"),
                        tf.keras.layers.Permute((3,1,2)),
                        tf.keras.layers.Reshape((self.nc*self.npred,1,self.height,self.width))
                    ],name="LatentDecoder"

                )
        #self.trainable_var = self.LEncoder.trainable_variables+self.LDecoder.trainable_variables + self.phi_conv.trainable_variables + self.phi_f.trainable_variables + self.wz.trainable_variables
        self._freeze_weight(self.DDecoder)
        self._freeze_weight(self.DEncoder)
    def _freeze_weight(self,Model):
        for layer in Model.layers:
            layer.trainable = False
 

    def decode(self,Input,z):
        f1 = self.LEncoder(Input)
        Wz = self.wz(z)
        return self.LDecoder(f1+Wz)
        
    def Dpredict(self,Input):
        return self.DDecoder(self.DEncoder(Input))


    def compute_loss(self,Input,Target):
        Dpred = self.Dpredict(Input)
        #print(self.DEncoder.name)
        #print(self.DEncoder.summary())
        #print(self.DDecoder.name)
        #print(self.DDecoder.summary())
        rError = Dpred - Target
        z= self.phi_f(self.phi_conv(rError))
        #print(self.phi_conv.name)
        #print(self.phi_conv.summary())
        #print(self.phi_f.name)
        #print(self.phi_f.summary())
        f1 = self.LEncoder(Input)
        #print(self.LEncoder.name)
        #print(self.LEncoder.summary())
        Wz = self.wz(z)
        #print(self.wz.name)
        #print(self.wz.summary())
        #would need to use expand and reshape in here
        Wz = tf.expand_dims(tf.expand_dims(Wz,1),1)
        Predict = self.LDecoder(f1+Wz)
        #print(self.LDecoder.name)
        #print(self.LDecoder.summary())
        los = tf.losses.mean_squared_error(Target,Predict)
        return los,Predict,Dpred,z 
    
    def compute_gradients(self,Input,Target):
        with tf.GradientTape() as tape:
            L ,Predict,Dpredict,z= self.compute_loss(Input,Target)
        return tape.gradient(L,self.trainable_variables),L

    def apply_gradients(self,grad,global_step=None):
        print(len(self.trainable_variables))
        self.opt.apply_gradients(zip(grad,self.trainable_variables),global_step=global_step)
   

