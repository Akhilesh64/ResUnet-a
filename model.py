from tf.keras.models import Model
from tf.keras.layers import *

class ResUnet(object):
    
    def __init__(self, num_classes, input_shape):
        self.num_classes = num_classes
        self.height, self.width, self.channels = input_shape

    def residual_block(self, input, filters, dilation_rates):
        out = [input]
        for rate in dilation_rates:
            x = BatchNormalization()(input)
            x = Activation('relu')(x)
            x = Conv2D(filters, 3, dilation_rate = rate, padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Conv2D(filters, 3, dilation_rate = rate, padding='same')(x)
            out.append(x)
        out = Add()(out)
        return out

    def PSPPooling(self, input, filters, height):
        out = [input]
        for i in [1,2,4]:
            x = MaxPooling2D(pool_size = i, strides = i)(input)
            x = UpSampling2D(size = i)(x)
            x = Conv2D((filters//4), 1, padding='same')(x)
            x = BatchNormalization()(x)
            out.append(x)
        if height==448 and filters==1024:
            out[3] = ZeroPadding2D((1, 1))(out[3])
        out = Concatenate(axis=-1)([out[0],out[1],out[2],out[3]])
        out = Conv2D(filters, 1, padding = 'same')(out)
        out = BatchNormalization()(out)
        return out

    def combine(self, x, y, filters):
        x = UpSampling2D(size = 2, interpolation='bilinear')(x)
        x = Concatenate(axis=-1)([x,y])
        x = Conv2D(filters, 1, padding = 'same')(x)
        x = BatchNormalization()(x)
        return x

    def build_model(self):
        input = Input(shape=(self.height,self.width, self.channels))
        x = Conv2D(32, 1, padding= 'same')(input)               #Layer-1
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        a = x
        x = self.residual_block(x, 32, [1,3,15,31])             #Layer-2
        b = x
        x = Conv2D(64, 1, strides = 2, padding='same')(x)       #Layer-3
        x = self.residual_block(x, 64, [1,3,15,31])             #Layer-4
        c = x
        x = Conv2D(128, 1, strides = 2, padding='same')(x)      #Layer-5
        x = self.residual_block(x, 128, [1,3,15])               #Layer-6
        d = x
        x = Conv2D(256, 1, strides = 2, padding='same')(x)      #Layer-7
        x = self.residual_block(x, 256, [1,3,15])               #Layer-8
        e = x
        x = Conv2D(512, 1, strides = 2, padding='same')(x)      #Layer-9
        x = self.residual_block(x, 512, [1])                    #Layer-10
        f = x
        x = Conv2D(1024, 1, strides = 2, padding='same')(x)     #Layer-11
        x = self.residual_block(x, 1024, [1])                   #Layer-12
        g = x 

        if self.height == 448:
            x = self.PSPPooling(x, 1024, self.height)                          #ResUnet-a d6 model
        
        x = Activation('relu')(x)
        if self.height != 448:
            x = Conv2D(2048, 1, strides = 2, padding = 'same')(x)
            x = self.residual_block(x, 2048, [1])
            # x = MaxPooling1D(pool_size=2, strides=2)(x)           #ResUnet-a d7v1 model
            x = self.PSPPooling(x, 2048, self.height)                            #ResUnet-a d7v2 model
            x = self.combine(x, g, 2048)

        x = self.combine(x, f, 512)
        x = self.residual_block(x, 512, [1])
        x = self.combine(x, e, 256)
        x = self.residual_block(x, 256, [1])
        x = self.combine(x, d, 128)
        x = self.residual_block(x, 128, [1])
        x = self.combine(x, c, 64)
        x = self.residual_block(x, 64, [1])
        x = self.combine(x, b, 32)
        x = self.residual_block(x, 32, [1])
        x1 = Concatenate(axis=-1)([x,a])
        x = self.PSPPooling(x1, 32, self.height)
        x = Activation('relu')(x)
        
        dist = ZeroPadding2D(padding=1)(x1)
        dist = Conv2D(32, 3)(dist)
        dist = BatchNormalization()(dist)
        dist = Activation('relu')(dist)
        dist = ZeroPadding2D(padding=1)(dist)
        dist = Conv2D(32, 3)(dist)
        dist = BatchNormalization()(dist)
        dist = Activation('relu')(dist)
        dist = Conv2D(self.num_classes, 1, activation='softmax', name = 'distance')(dist)

        bound = Concatenate(axis=-1)([x, dist])
        bound = ZeroPadding2D(padding=1)(bound)
        bound = Conv2D(32, 3)(bound)
        bound = BatchNormalization()(bound)
        bound = Activation('relu')(bound)
        bound = Conv2D(self.num_classes, 1, activation='sigmoid', name = 'boundary')(bound)

        color = Conv2D(3, 1, activation = 'sigmoid', name = 'color')(x1)

        seg = Concatenate(axis=-1)([x,bound,dist])
        seg = ZeroPadding2D(padding=1)(seg)
        seg = Conv2D(32, 3)(seg)
        seg = BatchNormalization()(seg)
        seg = Activation('relu')(seg)
        seg = ZeroPadding2D(padding=1)(seg)
        seg = Conv2D(32, 3)(seg)
        seg = BatchNormalization()(seg)
        seg = Activation('relu')(seg)
        seg = Conv2D(self.num_classes, 1, activation='softmax', name = 'segmentation')(seg)

        model = Model(inputs = input, outputs={'seg': seg, 'bound': bound, 'dist': dist, 'color' : color})
        
        return model
