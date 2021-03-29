import pyglet
import pyaudio
from collections import deque
import math
import numpy as np


def autocorrelogram(y):
    a = np.convolve(y, y[: : -1], 'same')
    a = a[a.size // 2 :]
    return a / np.dot(y, y)

def find_pitch(y, sr):
    if np.linalg.norm(y, 1) / y.size < 2e-2:
        return None
    a = autocorrelogram(y)
    peaks = []
    for b in np.split(np.arange(a.size), np.nonzero(a < 0)[0].tolist())[1 :]:
        if b:= [i for i in b if a[i] > .01]:
            peaks.append(max(b, key = lambda i: a[i]))
    if not peaks:
        return None
    highest_peak = max(a[p] for p in peaks)
    f = np.array([p for p in peaks if a[p] >= .95 * highest_peak][: 10])
    tau = np.average(f / np.arange(1, f.size + 1), 0, a[f])
    return sr / tau

def bytes_to_wav(b):
    return np.frombuffer(b, dtype = np.int16).astype(float) / (2 ** 15)


class MicStream:
    def __init__(self, fps = 30, rate = 44100):
        self.audio = pyaudio.PyAudio()
        self.rate = rate
        self.chunk = int(rate / fps)
        self.stream = None
    def start(self):
        self.stream = self.audio.open(
            format = pyaudio.paInt16,
            channels = 1,
            rate = self.rate,
            input = True,
            frames_per_buffer = self.chunk)
    def read(self):
        return bytes_to_wav(self.stream.read(self.chunk))
    def stop(self):
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()

class AudioManager:
    def __init__(self, stream, time_window):
        self.stream = stream
        self.rate = stream.rate
        self.chunk = stream.chunk
        self.time_window = time_window
        queue_len = int(time_window * self.rate / self.chunk) + 1
        self.audio_chunks = deque([], maxlen = queue_len)
    def open(self):
        self.stream.start()
    def close(self):
        self.stream.stop()
    def update(self):
        self.audio_chunks.append(self.stream.read())
    def get_audio(self):
        return np.concatenate(self.audio_chunks)[-int(self.time_window * self.rate) :]

class MusicalStaff:
    def __init__(self, max_samples):
        self.max_samples = max_samples
        self.queue = deque([], maxlen = max_samples)
        self.yshift = 0
        self.labels = {
            f'{n}{j}' :
                pyglet.text.Label(
                    f'{n}{j}',
                    font_name = 'Coolvetica',
                    font_size = 8,
                    color = (0, 0, 0, 255),
                    anchor_x = 'left', anchor_y = 'center')
            for n in 'ABCDEFG' for j in range(1, 10) }
    def add_sample(self, sample):
        self.queue.append(sample)
    def draw(self, x1, y1, x2, y2):
        x3 = x2 - 25
        pyglet.shapes.Rectangle(x = x1, y = y1, width = x2 - x1, height = y2 - y1, color = (255, 255, 255)).draw()
        ymap = lambda y: y1 + (y2 - y1) / 27 * (y + 4)
        ynotes = [0, 2, 4, 5, 7, 9, 11, 12, 14, 16, 17, 19, 21]
        j = self.yshift + 5
        for i, y in enumerate(ynotes):
            pyglet.graphics.draw(2, pyglet.gl.GL_LINES, ('v2f', [x1 + 5, int(ymap(y)), x3 + 3, int(ymap(y))]), ('c3B', [0, 0, 0] * 2))
            n = 'CDEFGAB'[i % 7]
            l = self.labels[f'{n}{j}']
            l.x, l.y = x3 + 5, ymap(y)
            l.draw()
            if n == 'B':
                j += 1
        for y in range(22):
            if y not in ynotes:
                pyglet.graphics.draw(2, pyglet.gl.GL_LINES, ('v2f', [x1 + 5, int(ymap(y)), x3, int(ymap(y))]), ('c3B', [160, 160, 160] * 2))
        
        x = np.array([i for i, v in enumerate(self.queue) if v is not None], dtype = int)
        y = np.array([v for v in self.queue if v is not None], dtype = float)
        y = np.round(np.log2(y / 440) * 12) - 3
        take = np.zeros(x.size, dtype = bool)
        take[1 : -1] = (y[: -2] == y[1 : -1]) & (y[1 : -1] == y[2 :])
        x = x[take]
        y = y[take]
        if y.size < 2:
            return
        ylast = y[len(self.queue) - x <= self.max_samples * .06]
        if ylast.size >= .03 * self.max_samples:
            self.yshift = min(range(-3, 4), key = lambda i: sum(np.maximum(0, abs(ylast - 11 - 12 * i) - 11)) + abs(i - self.yshift))
        y = y - 12 * self.yshift
        take = (-2 <= y) & (y <= 21)
        x = x[take]
        y = y[take]
        x = x1 + 5 + (x + self.max_samples - len(self.queue)) / self.max_samples * (x3 - x1 - 5)
        y = ymap(y)
        coords = np.empty(8 * x.size, dtype = float)
        coords[: : 8] = x - 2
        coords[1 : : 8] = y - 3
        coords[2: : 8] = x - 2
        coords[3 : : 8] = y + 3
        coords[4: : 8] = x + 2
        coords[5 : : 8] = y + 3
        coords[6: : 8] = x + 2
        coords[7 : : 8] = y - 3
        cols = np.tile(np.array([31, 119, 180], dtype = np.uint8), coords.size // 2)
        pyglet.graphics.draw(coords.size // 2, pyglet.gl.GL_QUADS, ('v2f', coords), ('c3B', cols))


window_config = pyglet.gl.Config(sample_buffers = 1, samples = 4)
window = pyglet.window.Window(width = 800, height = 600, config = window_config)

mic = MicStream(fps = 60)
audio = AudioManager(stream = mic, time_window = .04)
audio_track = np.array([], dtype = float)
staff = MusicalStaff(max_samples = 1000)

def update_func(dt):
    global audio_track
    audio.update()
    audio_track = audio.get_audio()
    pitch = find_pitch(audio_track, audio.rate)
    staff.add_sample(pitch)

pyglet.font.add_file('font.ttf')

def draw_audio_track(track, x1, y1, x2, y2, l):
    x = np.linspace(x1, x2, l)
    if track.size >= l:
        y = track[-l :]
    else:
        y = np.concatenate([np.zeros((l - track.size,), dtype = float), track])
    y = y * (y1 - y2) / 2 + (y1 + y2) / 2
    coords = np.empty(2 * l, dtype = float)
    coords[: : 2] = x
    coords[1 : : 2] = y
    pyglet.graphics.draw(l, pyglet.gl.GL_LINE_STRIP, ('v2f', coords))

pitch_label = pyglet.text.Label('',
    font_name = 'Coolvetica',
    font_size = 16,
    color = (255, 255, 255, 255),
    anchor_x = 'right', anchor_y = 'baseline')
@window.event
def on_draw():
    window.clear()
    draw_audio_track(audio_track, 0, 20, window.width, window.height / 2 - 20, int(audio.time_window * audio.rate))
    staff.draw(0, window.height / 2, window.width, window.height)
    if staff.queue and staff.queue[-1] is not None:
        pitch_label.text = f'{ staff.queue[-1]:.0f} Hz'
        pitch_label.x = window.width - 5
        pitch_label.y = 5
        pitch_label.draw()

pyglet.clock.schedule(update_func)

audio.open()
pyglet.app.run()
audio.close()
