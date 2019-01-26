from keras.layers import Input, Flatten, Activation, Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D
from keras.models import Model
from keras import backend as K

def build_model(height, width, depth, num_classes):
    K.clear_session()
    
    X_input = Input((height, width, depth))
    
    X = Conv2D(32, (3, 3), padding='same', activation='relu')(X_input)
    X = Conv2D(32, (3, 3), padding='same', activation='relu')(X)
    X = MaxPooling2D((2, 2), strides=(2, 2))(X)

    X = Conv2D(64, (3, 3), padding='same', activation='relu')(X)
    X = Conv2D(64, (3, 3), padding='same', activation='relu')(X)
    X = MaxPooling2D((2, 2), strides=(2, 2))(X)
    
    X = Conv2D(128, (3, 3), padding='same', activation='relu')(X)
    X = Conv2D(128, (3, 3), padding='same', activation='relu')(X)
    X = Conv2D(128, (3, 3), padding='same', activation='relu')(X)
    X = MaxPooling2D((2, 2), strides=(2, 2))(X)
    
    X = Conv2D(256, (3, 3), padding='same', activation='relu')(X)
    X = Conv2D(256, (3, 3), padding='same', activation='relu')(X)
    X = Conv2D(256, (3, 3), padding='same', activation='relu')(X)
    X = MaxPooling2D((2, 2), strides=(2, 2))(X)
    
    X = Flatten()(X)
    X = Dense(1024, activation='relu')(X)
    X = BatchNormalization()(X)
    X = Dense(256, activation='relu')(X)
    X = BatchNormalization()(X)
    X = Dropout(0.25)(X)
    
    X = Dense(num_classes, activation='softmax')(X)
    
    model = Model(inputs=X_input, outputs=X)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

def build_model_2(height, width, depth, num_classes):
    """
    add normalization layer to prevent overfitting
    """
    
    K.clear_session()
    
    X_input = Input((height, width, depth))
    
    X = Conv2D(32, (3, 3), padding='same', activation='relu')(X_input)
    X = BatchNormalization()(X)
    X = Conv2D(32, (3, 3), padding='same', activation='relu')(X)
    X = BatchNormalization()(X)
    X = MaxPooling2D((2, 2), strides=(2, 2))(X)

    X = Conv2D(64, (3, 3), padding='same', activation='relu')(X)
    X = BatchNormalization()(X)
    X = Conv2D(64, (3, 3), padding='same', activation='relu')(X)
    X = BatchNormalization()(X)
    X = MaxPooling2D((2, 2), strides=(2, 2))(X)
    
    X = Conv2D(128, (3, 3), padding='same', activation='relu')(X)
    X = BatchNormalization()(X)
    X = Conv2D(128, (3, 3), padding='same', activation='relu')(X)
    X = BatchNormalization()(X)
    X = Conv2D(128, (3, 3), padding='same', activation='relu')(X)
    X = MaxPooling2D((2, 2), strides=(2, 2))(X)
    
    X = Conv2D(256, (3, 3), padding='same', activation='relu')(X)
    X = BatchNormalization()(X)
    X = Conv2D(256, (3, 3), padding='same', activation='relu')(X)
    X = BatchNormalization()(X)
    X = Conv2D(256, (3, 3), padding='same', activation='relu')(X)
    X = BatchNormalization()(X)
    X = MaxPooling2D((2, 2), strides=(2, 2))(X)
    
    X = Flatten()(X)
    X = Dense(1024, activation='relu')(X)
    X = BatchNormalization()(X)
    X = Dense(256, activation='relu')(X)
    X = BatchNormalization()(X)
    X = Dropout(0.25)(X)
    
    X = Dense(num_classes, activation='softmax')(X)
    
    model = Model(inputs=X_input, outputs=X)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model   
