"""Controller 1 on $4016: write 1 then 0 to latch, then read 8 times.

Read order is A, B, Select, Start, Up, Down, Left, Right (bit 0 first).
"""

A, B, SELECT, START, UP, DOWN, LEFT, RIGHT = (1 << i for i in range(8))


class Pad:
    def __init__(self):
        self.buttons = 0   # currently held buttons, set by the input handler
        self.shift = 0
        self.strobe = 0

    def write(self, val):
        self.strobe = val & 1
        if self.strobe:
            self.shift = self.buttons

    def read(self):
        if self.strobe:                       # while strobe is high, always reports A
            return 0x40 | (self.buttons & 1)
        bit = self.shift & 1
        self.shift = (self.shift >> 1) | 0x80  # after 8 reads, returns 1s
        return 0x40 | bit