"""
Run the learned model to connect to client with ros messages
"""
import pygame
from pygame.locals import K_a
from pygame.locals import K_q
import time
import fire

# ros packages
import rospy
from geometry_msgs.msg import Twist

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

class Timer(object):
    def __init__(self):
        self.step = 0
        self._lap_step = 0
        self._lap_time = time.time()

    def tick(self):
        self.step += 1

    def lap(self):
        self._lap_step = self.step
        self._lap_time = time.time()

    def ticks_per_second(self):
        return float(self.step - self._lap_step) / self.elapsed_seconds_since_lap()

    def elapsed_seconds_since_lap(self):
        return time.time() - self._lap_time

class Controller(object):
    tele_twist = Twist()
    def __init__(self, mode, max_x, max_z):
        self._mode = mode
        self._max_x = max_x
        self._max_z = max_z
        self._timer = Timer()

    def _on_loop(self):
        """
        Logical loop
        """
        pass

    def _on_render(self):
        """
        render loop
        """
        pass

    def _initialize_game(self):
        self._display = pygame.display.set_mode(
                (WINDOW_WIDTH, WINDOW_HEIGHT),
                pygame.HWSURFACE | pygame.DOUBLEBUF)

    def execute(self):
        pygame.init()
        self._initialize_game()
        try:
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
            self._on_loop()
            self._on_render()
        finally:
            pygame.quit()

# wrapper for fire to get command arguments
def run_wrapper(mode, max_x=1, max_z=1):
    controller = Controller(mode, max_x, max_z)
    controller.execute()

def main():
    fire.Fire(run_wrapper)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print ('\nCancelled by user! Bye Bye!')
