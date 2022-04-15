import pygame as pg


class Player(pg.sprite.Sprite):
    def __init__(self, name='Palyer 1', score=0):
        pg.sprite.Sprite.__init__(self)
        self.name = name
        self.score = score
