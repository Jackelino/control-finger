class Setup:
    def __init__(self, tittle="Control finger", screenWidth=800, screenHeight=600):
        # TITLE
        self._tittle = tittle
        # SCREEN GAME
        self._screenWidth = screenWidth
        self._screenHeight = screenHeight
        # SCREEN CAMERA
        self._screenWidthCamera = 500
        self._screenHeightCamera = 500
        # COLORS
        self._colors = {
            'WHITE': (255, 255, 255),
            'BLACK': (0, 0, 0),
            'BLUE': (0, 0, 255),

        }
        # FONTS
        self._font = {
            'Fixedsys500': './resources/fonts/Fixedsys500c.ttf'
        }
        self._sizeFont = 20
        # MUSIC
        self._musics = {
            'musicBackground': './resources/music/music_main_theme.ogg'
        }

        # SOUND
        self._sounds = {
            'musicBackground': './resources/sound'

        }
        # image
        self._images = {
            'background1': './resources/images/background_01.png',
            'background2': './resources/images/background_02.png'
        }
        # SPRITE_SHEET
        self._spriteSheet = {
            'spitePlay': './resources/sprites/p1_walk.png',
            'spriteTiles': './resources/sprites/tiles_spritesheet.png',
            'spriteBlocks': './resources/sprites/sheet.png'
        }
        # plataforms
        # These constants define our platform types:
        #   Name of file
        #   X location of sprite
        #   Y location of sprite
        #   Width of sprite
        #   Height of sprite
        self._platforms = {
            'GRASS_LEFT': (576, 720, 70, 70),
            'GRASS_RIGHT': (576, 576, 70, 70),
            'GRASS_MIDDLE': (504, 576, 70, 70),
            'STONE_PLATFORM_LEFT': (432, 720, 70, 40),
            'STONE_PLATFORM_MIDDLE': (648, 648, 70, 40),
            'STONE_PLATFORM_RIGHT': (792, 648, 70, 40),
        }
        # FPS
        self._fps = 60

        # MODELS
        self._models = {
            'model1': './recognition/models/model1.h5'
        }

        # weights
        self._weights = {
            'weight1': './recognition/weights/weight1.h5'
        }

    # GETTERS

    @property
    def screenHeight(self):
        return self._screenHeight

    @property
    def platforms(self):
        return self._platforms

    @property
    def screenWidthCamera(self):
        return self._screenWidthCamera

    @property
    def screenHeightCamera(self):
        return self._screenHeightCamera

    @property
    def tittle(self):
        return self._tittle

    @property
    def screenWidth(self):
        return self._screenWidth

    @property
    def font(self):
        return self._font

    @property
    def sizeFont(self):
        return self._sizeFont

    @property
    def colors(self):
        return self._colors

    @property
    def musics(self):
        return self._musics

    @property
    def sounds(self):
        return self._sounds

    @property
    def images(self):
        return self._images

    @property
    def spriteSheet(self):
        return self._spriteSheet

    @property
    def fps(self):
        return self._fps

    # SETTERS
    @screenHeight.setter
    def screenHeight(self, screenHeight):
        self._screenHeight = screenHeight
