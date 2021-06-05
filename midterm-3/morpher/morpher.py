import joblib
import os

import numpy as np
import tensorflow as tf

import tensorflow.keras as keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import InputLayer, Flatten, Reshape, Dense
from tensorflow.keras.models import Model, Sequential

import sklearn.svm

import pyglet


class DenseTranspose(keras.layers.Layer):
  def __init__(self, dense, activation, **kwargs):
    super().__init__(**kwargs)
    self.dense = dense
    self.activation = keras.activations.get(activation)
  def build(self, batch_input_shape):
    self.biases = self.add_weight(
        name = 'bias',
        shape = [self.dense.input_shape[-1]],
        trainable = True,
        initializer = 'zeros')
  def call(self, batch_input):
    z = tf.matmul(batch_input, self.dense.weights[0], transpose_b = True)
    return self.activation(z + self.biases)


class DeepDenoisingAutoEncoder:
  def __init__(self, hidden_sizes, noise_factor):
    self.noise_factor = noise_factor
    self.hidden_sizes = hidden_sizes
    self.num_layers = len(hidden_sizes)
    self.dense_layers = [Dense(hidden_sizes[0], activation = 'sigmoid')] + [Dense(c, activation = 'relu') for c in hidden_sizes[1 :]]
    self.encoder = Sequential([InputLayer((28, 28)), Flatten()] +
                              self.dense_layers,
                              name = 'encoder')
    self.transposed_layers = [DenseTranspose(self.dense_layers[0], activation = 'sigmoid')] + [DenseTranspose(l, activation = 'relu') for l in self.dense_layers[1 :]]
    self.decoder = Sequential([InputLayer(self.dense_layers[-1].output_shape[1 :])] + 
                              self.transposed_layers[: : -1] +
                              [Reshape((28, 28))],
                              name = 'decoder')
    self.autoencoder = Sequential([InputLayer((28, 28)), self.encoder, self.decoder], name = 'autoencoder')
    self.svm = sklearn.svm.SVC()
  
  def save(self, filename):
    pck = {'noise_factor' : self.noise_factor,
           'hidden_sizes' : self.hidden_sizes,
           'dense_weights' : [l.get_weights() for l in self.dense_layers],
           'transposed_weights' : [l.get_weights() for l in self.transposed_layers],
           'svm' : self.svm}
    joblib.dump(pck, filename)
  
  @staticmethod
  def load(filename):
    pck = joblib.load(filename)
    ae = DeepDenoisingAutoEncoder(pck['hidden_sizes'], pck['noise_factor'])
    ae.svm = pck['svm']
    for l, w in zip(ae.dense_layers, pck['dense_weights']):
      l.set_weights(w)
    for l, w in zip(ae.transposed_layers, pck['transposed_weights']):
      l.set_weights(w)
    return ae
    
  def make_noisy(self, a):
    b = a + self.noise_factor * tf.random.normal(shape = a.shape, dtype = a.dtype)
    return np.clip(b, 0, 1)
  
  def compile(self, *args, **kwargs):
    return self.autoencoder.compile(*args, **kwargs)
  
  def __call__(self, *args, **kwargs):
    return self.autoencoder.__call__(*args, **kwargs)
  
  def predict(self, *args, **kwargs):
    return self.autoencoder.predict(*args, **kwargs)


ae = DeepDenoisingAutoEncoder.load('model.sav')
ae.encoder.compile()
ae.decoder.compile()

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype(float) / 255
x_test = x_test.astype(float) / 255


def numpy_to_img(a):
    data = (255 * a)[: : -1, :].flatten().astype('uint8')
    tex_data = (pyglet.gl.GLubyte * data.size)( * data )
    img = pyglet.image.ImageData(28, 28, 'L', tex_data)
    img.get_texture()
    pyglet.gl.glTexParameteri(pyglet.gl.GL_TEXTURE_2D, pyglet.gl.GL_TEXTURE_MAG_FILTER, pyglet.gl.GL_NEAREST) 
    return img

digits = np.array([x_train[np.where(y_train == i)[0][0]] for i in range(10)])
encoded_digits = ae.encoder(digits)
digits = [numpy_to_img(d) for d in digits]

button_r = 10

window_size = 56 * 11 + 1 * 9 + 5
window_config = pyglet.gl.Config(sample_buffers = 1, samples = 4)
window = pyglet.window.Window(width = window_size, height = window_size + 4 * button_r, config = window_config)
pyglet.gl.glClearColor(1, 1, 1, 1)


morph_factor = 0
morphed_digits = [[None for i in range(10)] for j in range(10)]
to_update = True
highlighted = None

def update_func(dt):
    global morph_factor
    global morphed_digits
    global to_update
    
    if not to_update:
        return
    
    decoded_morphed_digits = ae.decoder(tf.reshape(encoded_digits + morph_factor * (encoded_digits[:, None, :] - encoded_digits), (100, 32))).numpy()
    for i in range(10):
        for j in range(10):
            morphed_digits[i][j] = numpy_to_img(decoded_morphed_digits[10 * i + j])
    to_update = False



factor_to_screen = lambda i: (int(2 * button_r + (window.width - 4 * button_r) * i), 2 * button_r)
screen_to_factor = lambda x: (x - 2 * button_r) / (window.width - 4 * button_r)


original_x = None
mouse_pressed = False
def update_factor(x):
    global to_update
    global morph_factor
    morph_factor = min(1, max(0, screen_to_factor(x - original_x)))
    print(morph_factor)
    to_update = True
@window.event
def on_mouse_press(x, y, button, modifiers):
    global mouse_pressed
    global original_x
    global highlighted
    
    print('pressed')
    bx, by = factor_to_screen(morph_factor)
    if button & pyglet.window.mouse.LEFT and (x - bx) ** 2 + (y - by) ** 2 <= button_r ** 2:
        print('pressed here')
        mouse_pressed = True
        original_x = x - factor_to_screen(morph_factor)[0]
    i = (x - 56 - 5) // 58
    j = (window.height - y - 56 - 5) // 58
    print(f'highlight {i} {j}')
    if 0 <= i <= 9 and 0 <= j <= 9:
        highlighted = (i, j)
        
@window.event
def on_mouse_release(x, y, button, modifiers):
    global mouse_pressed
    if button & pyglet.window.mouse.LEFT:
        mouse_pressed = False
@window.event
def on_mouse_drag(x, y, dx, dy, button, modifiers):
    if mouse_pressed:
        print('dragged')
        update_factor(x)
        

@window.event
def on_draw():
    global morphed_digits
    
    window.clear()
    batch = pyglet.graphics.Batch()
    sprites = []
    if highlighted is not None:
        i, j = highlighted
        pyglet.shapes.Rectangle(5 + (i + 1) * 57 - 1, window.height - 5 - (j + 2) * 57, 58, 58, color = (255, 0, 0)).draw()
    for i in range(10):
        sprites.append(pyglet.sprite.Sprite(digits[i], 0, window.height - 5 - (i + 1) * 57 - 56, batch = batch))
        sprites.append(pyglet.sprite.Sprite(digits[i], 5 + (i + 1) * 57, window.height - 56, batch = batch))
        for j in range(10):
            if morphed_digits[i][j] is not None:
                sprites.append(pyglet.sprite.Sprite(morphed_digits[i][j], 5 + (i + 1) * 57, window.height - 5 - (j + 1) * 57 - 56, batch = batch))
    for s in sprites:
        s.scale = 2
    
    slider = pyglet.shapes.Rectangle(2 * button_r, 2 * button_r - 3, window.width - 4 * button_r, 6, color = (128, 128, 128), batch = batch)
    button_outline = pyglet.shapes.Circle(*factor_to_screen(morph_factor), button_r, color = (0, 0, 0), batch = batch)
    button_inside = pyglet.shapes.Circle(*factor_to_screen(morph_factor), button_r - 4, color = (255, 255, 255), batch = batch)
    
    batch.draw()

pyglet.clock.schedule(update_func)

pyglet.app.run()
    
