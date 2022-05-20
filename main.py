import pygame as pg
from setup.Setup import Setup
from player.Player import Player
from control.Control import Control

running = True
player = Player('Jack')
setup = Setup()

pg.init()
pg.mixer.init()
screen = pg.display.set_mode((setup.screenWidth, setup.screenHeight))
pg.display.set_caption(setup.tittle)
font = pg.font.Font(setup.font['Fixedsys500'], setup.sizeFont)
textScore = font.render("score: " + str(player.score), True, setup.colors['WHITE'])
textPlayer = font.render("player: " + player.name, True, setup.colors['WHITE'])
clock = pg.time.Clock()
textTime = font.render("Time: " + str(clock.tick()), True, setup.colors['WHITE'])

def draw_windows():
    screen.fill(setup.colors['BLACK'])
    screen.blit(textScore, (10, 10))
    screen.blit(textPlayer, (600, 10))
    screen.blit(textTime, (150, 10))
    pg.display.flip()


def main(flag):
    # Game Loop
    while flag:
        # Keep loop running at the right speed
        clock.tick(setup.fps)
        # Process input (events)
        for event in pg.event.get():
            # check for closing window
            if event.type == pg.QUIT:
                flag = False
        # Draw / Render
        draw_windows()
    pg.quit()


main(running)
