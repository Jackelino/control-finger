import pygame as pg
from setup import Setup


class Player(pg.sprite.Sprite):
    def __init__(self, name='Palyer 1', score=0):
        pg.sprite.Sprite.__init__(self)
        self.name = name
        self.score = score



    def walking(self):
        pass

    def jumping(self):
        pass

    def repose(self):
        pass

    def handleState(self):
        pass

    def update(self):
        pass