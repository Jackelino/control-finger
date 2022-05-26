import pygame

from setup.Setup import Setup
from platforms.Platform import *


class Level:
    """ This is a generic super-class used to define a level.
        Create a child class for each level with level-specific
        info. """

    # Lists of sprites used in all levels. Add or remove
    # lists as needed for your game. """
    platform_list = None
    enemy_list = None

    # Background image
    background = None

    # How far this world has been scrolled left/right
    world_shift = 0
    level_limit = -1000
    # setup
    setup = Setup()

    def __init__(self, player):
        """ Constructor. Pass in a handle to player. Needed for when moving platforms
            collide with the player. """
        self.platform_list = pygame.sprite.Group()
        self.enemy_list = pygame.sprite.Group()
        self.player = player

    # Update everythign on this level
    def update(self):
        """ Update everything in this level."""
        self.platform_list.update()
        self.enemy_list.update()

    def draw(self, screen):
        """ Draw everything on this level. """

        # Draw the background
        # We don't shift the background as much as the sprites are shifted
        # to give a feeling of depth.
        screen.fill(self.setup.colors['BLUE'])
        screen.blit(self.background, (self.world_shift // 3, 0))

        # Draw all the sprite lists that we have
        self.platform_list.draw(screen)
        self.enemy_list.draw(screen)

    def shift_world(self, shift_x):
        """ When the user moves left/right and we need to scroll everything: """

        # Keep track of the shift amount
        self.world_shift += shift_x

        # Go through all the sprite lists and shift
        for platform in self.platform_list:
            platform.rect.x += shift_x

        for enemy in self.enemy_list:
            enemy.rect.x += shift_x


# Create platforms for the level
class Level_01(Level):
    """ Definition for level 1. """
    setup = Setup()

    def __init__(self, player):
        """ Create level 1. """

        # Call the parent constructor
        Level.__init__(self, player)

        self.background = pygame.image.load(self.setup.images['background1']).convert()
        self.background.set_colorkey(self.setup.colors['WHITE'])
        self.level_limit = -2500

        # Array with type of platform, and x, y location of the platform.
        level = [[self.setup.platforms['GRASS_LEFT'], 500, 500],
                 [self.setup.platforms['GRASS_MIDDLE'], 570, 500],
                 [self.setup.platforms['GRASS_RIGHT'], 640, 500],
                 [self.setup.platforms['GRASS_LEFT'], 800, 400],
                 [self.setup.platforms['GRASS_MIDDLE'], 870, 400],
                 [self.setup.platforms['GRASS_RIGHT'], 940, 400],
                 [self.setup.platforms['GRASS_LEFT'], 1000, 500],
                 [self.setup.platforms['GRASS_MIDDLE'], 1070, 500],
                 [self.setup.platforms['GRASS_RIGHT'], 1140, 500],
                 [self.setup.platforms['STONE_PLATFORM_LEFT'], 1120, 280],
                 [self.setup.platforms['STONE_PLATFORM_MIDDLE'], 1190, 280],
                 [self.setup.platforms['STONE_PLATFORM_RIGHT'], 1260, 280],
                 ]

        # Go through the array above and add platforms
        for platform in level:
            block = Platform(platform[0])
            block.rect.x = platform[1]
            block.rect.y = platform[2]
            block.player = self.player
            self.platform_list.add(block)

        # Add a custom moving platform
        block = MovingPlatform(self.setup.platforms['STONE_PLATFORM_MIDDLE'])
        block.rect.x = 1350
        block.rect.y = 280
        block.boundary_left = 1350
        block.boundary_right = 1600
        block.change_x = 1
        block.player = self.player
        block.level = self
        self.platform_list.add(block)


# Create platforms for the level
class Level_02(Level):
    """ Definition for level 2. """
    setup = Setup()

    def __init__(self, player):
        """ Create level 1. """

        # Call the parent constructor
        Level.__init__(self, player)

        self.background = pygame.image.load(self.setup.images['background2']).convert()
        self.background.set_colorkey(self.setup.colors['WHITE'])
        self.level_limit = -1000

        # Array with type of platform, and x, y location of the platform.
        level = [[self.setup.platforms['STONE_PLATFORM_LEFT'], 500, 550],
                 [self.setup.platforms['STONE_PLATFORM_MIDDLE'], 570, 550],
                 [self.setup.platforms['STONE_PLATFORM_RIGHT'], 640, 550],
                 [self.setup.platforms['GRASS_LEFT'], 800, 400],
                 [self.setup.platforms['GRASS_MIDDLE'], 870, 400],
                 [self.setup.platforms['GRASS_RIGHT'], 940, 400],
                 [self.setup.platforms['GRASS_LEFT'], 1000, 500],
                 [self.setup.platforms['GRASS_MIDDLE'], 1070, 500],
                 [self.setup.platforms['GRASS_RIGHT'], 1140, 500],
                 [self.setup.platforms['STONE_PLATFORM_LEFT'], 1120, 280],
                 [self.setup.platforms['STONE_PLATFORM_MIDDLE'], 1190, 280],
                 [self.setup.platforms['STONE_PLATFORM_RIGHT'], 1260, 280],
                 ]

        # Go through the array above and add platforms
        for platform in level:
            block = Platform(platform[0])
            block.rect.x = platform[1]
            block.rect.y = platform[2]
            block.player = self.player
            self.platform_list.add(block)

        # Add a custom moving platform
        block = MovingPlatform(self.setup.platforms['STONE_PLATFORM_MIDDLE'])
        block.rect.x = 1500
        block.rect.y = 300
        block.boundary_top = 100
        block.boundary_bottom = 550
        block.change_y = -1
        block.player = self.player
        block.level = self
        self.platform_list.add(block)
