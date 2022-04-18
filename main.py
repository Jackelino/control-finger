import sys
import os
import pygame as pg
from setup.Setup import Setup
from player.Player import Player

running = True

player = Player('Jack')
setup = Setup()

pg.init()
pg.mixer.init()
screen = pg.display.set_mode((setup.screenWidth, setup.screenHeight))
pg.display.set_caption(setup.tittle)
font = pg.font.Font(setup.font['Fixedsys500'], setup.sizeFont)
textScore = font.render("score", True, setup.colors['WHITE'])

clock = pg.time.Clock()

# Game Loop
while running:
    # Keep loop running at the right speed
    clock.tick(60)
    # Process input (events)
    for event in pg.event.get():
        # check for closing window
        if event.type == pg.QUIT:
            running = False
    # Draw / Render
    screen.fill(setup.colors['BLACK'])
    screen.blit(textScore, (10, 10))
    pg.display.flip()
pg.quit()