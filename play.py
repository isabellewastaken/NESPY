"""Windowed runner.  Usage:  python play.py smb.nes [max_frames]

Needs:  pip install pygame numpy

Keys: arrows = D-pad, X = A, Z = B, Enter = Start, Tab/Right Shift = Select.
The window title shows the achieved frame rate (target is 60).
"""
import sys

import pygame

import pad as P
from main import build, run_frame

SCALE = 3
KEYS = {
    pygame.K_x: P.A, pygame.K_z: P.B,
    pygame.K_RETURN: P.START, pygame.K_TAB: P.SELECT, pygame.K_RSHIFT: P.SELECT,
    pygame.K_UP: P.UP, pygame.K_DOWN: P.DOWN, pygame.K_LEFT: P.LEFT, pygame.K_RIGHT: P.RIGHT,
}


def main():
    rom = sys.argv[1] if len(sys.argv) > 1 else "smb.nes"
    max_frames = int(sys.argv[2]) if len(sys.argv) > 2 else None

    cpu, ppu, pad = build(rom)
    pygame.init()
    screen = pygame.display.set_mode((256 * SCALE, 240 * SCALE))
    pygame.display.set_caption("NES in Python")
    clock = pygame.time.Clock()

    target = cpu.cycles
    frame = 0
    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type in (pygame.KEYDOWN, pygame.KEYUP):
                bit = KEYS.get(e.key)
                if bit:
                    if e.type == pygame.KEYDOWN:
                        pad.buttons |= bit
                    else:
                        pad.buttons &= ~bit

        target = run_frame(cpu, ppu, target)

        surf = pygame.image.frombuffer(ppu.frame_rgb().tobytes(), (256, 240), "RGB")
        screen.blit(pygame.transform.scale(surf, screen.get_size()), (0, 0))
        pygame.display.flip()
        clock.tick(60)

        frame += 1
        if frame % 30 == 0:
            pygame.display.set_caption(f"NES in Python - {clock.get_fps():.1f} fps")
        if max_frames and frame >= max_frames:
            running = False
    pygame.quit()


if __name__ == "__main__":
    main()