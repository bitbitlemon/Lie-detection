# model.py
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Flatten, Dropout, Activation, BatchNormalization, Conv1D, MaxPooling1D, Concatenate

def build_model(input_shape):
    input_data = Input(shape=input_shape, name='input_data')
    input_prompt = Input(shape=input_shape, name='input_prompt')  # 仅包含性别提示
    merged_input = Concatenate(axis=2)([input_data, input_prompt])

    x = Conv1D(256, 8, padding='valid')(merged_input)
    x = Activation('relu')(x)
    x = Conv1D(256, 8, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.25)(x)
    x = MaxPooling1D(pool_size=(8))(x)
    x = Conv1D(128, 8, padding='same')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.25)(x)
    x = MaxPooling1D(pool_size=(8))(x)
    x = Flatten()(x)
    output = Dense(input_shape[0], activation='softmax')(x)

    model = Model(inputs=[input_data, input_prompt], outputs=output)
    return model
