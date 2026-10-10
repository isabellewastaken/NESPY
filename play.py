"""Windowed runner.  Usage:  python play.py smb.nes [max_frames]

Needs:  pip install pygame-ce numpy   (or pygame on older Pythons)

Keys: arrows = D-pad, X = A, Z = B, Enter = Start, Tab/Right Shift = Select.
The window title shows the achieved frame rate (a real NTSC NES runs at 60.1).
"""
import sys
import time

import pygame

import pad as P
from main import build, run_frame

SCALE = 3
FRAME_TIME = 1.0 / 60.0988          # NTSC NES frame period
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

    perf = time.perf_counter
    target = cpu.cycles
    next_frame = last_report = perf()
    frame = frames_since_report = 0
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

        # Precise limiter: sleep for most of the wait, spin for the last ~1.5 ms.
        # (pygame's clock.tick(60) truncates to 16 ms = 62.5 fps, a bit too fast.)
        next_frame += FRAME_TIME
        slack = next_frame - perf() - 0.0015
        if slack > 0.001:
            pygame.time.wait(int(slack * 1000))
        while perf() < next_frame:
            pass
        now = perf()
        if now - next_frame > 0.1:           # big hitch (window drag etc.): don't sprint
            next_frame = now

        frame += 1
        frames_since_report += 1
        if now - last_report >= 0.5:
            pygame.display.set_caption(
                f"NES in Python - {frames_since_report / (now - last_report):.1f} fps")
            last_report, frames_since_report = now, 0
        if max_frames and frame >= max_frames:
            running = False
    pygame.quit()


if __name__ == "__main__":
    main()