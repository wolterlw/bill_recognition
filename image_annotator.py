import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import cv2
from matplotlib.widgets import Button
from scipy.io import savemat

img_names = glob("/home/wolterlw/practice/diploma/bills_cropped/*")
get_name = lambda x: img_names[x][46:-4]
imgs = [cv2.imread(x,0) for x in img_names]

class ImageProcessor:
    def __init__(self,ax,fig):
        self.ax = ax
        self.fig = fig
        self.line_data = []
        self.idx = 0
        self.ax.imshow(imgs[self.idx], cmap='Greys_r')
        self.act_line = None
        self.press = None
        self.ax.set_title(str(self.idx) + " / " +str(len(imgs)))
        self.view_width = 200
        self.view_center = 100
        self.ax.set_ylim(
            self.view_center + self.view_width/2,
            self.view_center - self.view_width/2)
        self.step = 10

    def on_press(self, event):
        'on button press we will see if the mouse is over us and store some data'
        if event.inaxes != self.ax: return
        self.press = event.xdata, event.ydata
        self.act_line = self.ax.plot([self.press[0]],[self.press[1]],c='r')[0]

    def on_motion(self, event):
        'on motion we will move the rect if the mouse is over us'
        if self.press is None: return
        if event.inaxes != self.ax: return
        
        self.act_line.set_data( [self.press[0], event.xdata], [self.press[1], event.ydata])
        self.fig.canvas.draw()

    def on_scroll(self, event):
        if event.button == 'up':
            self.view_center -= self.step
            self.ax.set_ylim(
                self.view_center + self.view_width/2,
                self.view_center - self.view_width/2)
        if event.button == 'down':
            self.view_center += self.step
            self.ax.set_ylim(
                self.view_center + self.view_width/2,
                self.view_center - self.view_width/2)
        self.fig.canvas.draw()

    def on_release(self, event):
        if event.inaxes != self.ax: return
        self.press = None
        data = np.array(self.act_line.get_data()).astype('uint8')
        if len(data[0]) == 2:
            self.line_data.append(
                np.array(self.act_line.get_data())
            )
        self.act_line = None

    def undo(self):
        y = self.line_data.pop()
        lines = self.ax.get_lines()
        self.ax.get_lines()[-1].remove()
        self.fig.canvas.draw()

    def redraw(self,idx_prev):

        self.ax.clear()
        savemat("./annotations/" + get_name(idx_prev) + ".mat", {"lines": np.r_[self.line_data]})
        self.act_line = None
        self.ax.set_title(str(self.idx) + " / " +str(len(imgs)))
        self.ax.imshow(imgs[self.idx],cmap='Greys_r')
        self.line_data = []

    def on_key(self, event):
        if event.key == 'ctrl+z':
            self.undo()
        if event.key == 'right':
            idx_prev = self.idx
            self.idx = min(len(imgs)-1,self.idx+1)
            self.redraw(idx_prev)
        if event.key == 'left':
            idx_prev = self.idx
            self.idx = max(0,self.idx-1)
            self.redraw(idx_prev)

    def connect(self):
        'connect to all the events we need'
        self.cidpress = self.fig.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = self.fig.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmotion = self.fig.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)
        self.cidscroll = self.fig.canvas.mpl_connect(
            'scroll_event', self.on_scroll)
        self.cidundo = self.fig.canvas.mpl_connect(
            'key_press_event', self.on_key)


fig,ax = plt.subplots()

img_proc = ImageProcessor(ax,fig)
img_proc.connect()

plt.show()