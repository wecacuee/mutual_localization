import sys, os
import pygame
from pygame import camera, surfarray
import inspect
import numpy as np
import scipy

DEBUG = True
_init_done = False
def init():
    global _init_done
    if _init_done:
        return
    pygame.init()
    camera.init()
    _init_done = True

def display_cams(cams, screen, onmouseclick, resolutions):
    running = True
    while running:
        event = pygame.event.poll()
        keyinput = pygame.key.get_pressed()
        # exit on corner 'x' click or escape key press
        if keyinput[pygame.K_ESCAPE]:
            running = False
        elif event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            onmouseclick()
        screen.fill([0, 0, 0])
        pos = (0, 0)
        for i, cam in enumerate(cams):
            surf = cam.get_image()
            screen.blit(surf, pos)
            pos = (pos[0], pos[1] + resolutions[i][1])
        pygame.display.flip()
    for c in cams:
        c.stop()
    pygame.display.quit()

class SaveImgEventHandler(object):
    def __init__(self, file_formats=['/tmp/img%04d_0.png', '/tmp/img%04d_1.png']):
        self.file_formats = file_formats

    def onmouseclick(self, multicap):
        imgs = list()
        for cam in multicap.cams:
            img = multicap.capture(cam)
            imgs.append(img)
        for ff, img in zip(self.file_formats, imgs):
            fname = ff % (multicap.count)
            while os.path.exists(fname):
                multicap.count += 1
                fname = ff % (multicap.count)
            print("Saved image to {0}".format(fname))
            scipy.misc.imsave(fname, img)
        multicap.count += 1
        return imgs, fname

class MultiCapture(object):
    def __init__(self, configs):
        init()
        self.cams = list()
        self.resolutions = list()
        self.count = 0
        self.screen = pygame.display.set_mode(
            (max([conf['resolution'][0] for conf in configs]),
             sum([conf['resolution'][1] for conf in configs])))
        pos = [0, 0]
        for conf in configs:
            cam = camera.Camera(conf['dev'], conf['resolution'], "RGB")
            cam.start()

            surf = cam.get_image()
            self.cams.append(cam)
            self.resolutions.append(conf['resolution'])
    def start(self):
        display_cams(self.cams, self.screen, self.onmouseclick,
                     self.resolutions)

    def capture(self, cam):
        surf = cam.get_image()
        imgt = surfarray.array3d(surf)
        img = np.transpose(imgt, (1, 0, 2)) # swap rows and cols
        return img

    def onmouseclick(self):
        if hasattr(self, 'event_handler'):
            self.event_handler.onmouseclick(self)

def main(**kwargs):
    mc = MultiCapture([dict(dev='/dev/video0',
                       resolution=(1024, 576)),
                   dict(dev='/dev/video1',
                       resolution=(1024, 576)),
                  ])
    mc.event_handler = SaveImgEventHandler(**kwargs)
    mc.start()

if __name__ == '__main__':
    import sys
    kwargs = dict()
    if len(sys.argv) > 1:
        kwargs['file_formats'] = sys.argv[1:3]
    main(**kwargs)
