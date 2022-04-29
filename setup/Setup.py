
class Setup:
    def __init__(self, tittle="Control finger", screenWidth=800, screenHeight=600):
        # TITLE
        self._tittle = tittle
        # SCREEN
        self._screenWidth = screenWidth
        self._screenHeight = screenHeight
        # COLORS
        self._colors = {
            'WHITE': (255, 255, 255),
            'BLACK': (0, 0, 0)
        }
        # FONTS
        self._font = {
            'Fixedsys500': './resources/fonts/Fixedsys500c.ttf'
        }
        self._sizeFont = 20
        # MUSIC
        self._musics = {

        }

        # SOUND
        self._sounds = {

        }
        # image
        self._images = {

        }
        # SPRITE_SHEET
        self._spriteSheet = {

        }
        # FPS
        self._fps = 60
    # GETTERS

    @property
    def screenHeight(self):
        return self._screenHeight

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
