"""Param2Image - field-map surrogate for impact damage.

The model predicts four *physical* fields on the plate's own finite-element
grid (200 x 200 cells, 3.000 mm per cell):

    ch 0  von Mises stress   (MPa)
    ch 1  PEEQ               (-)
    ch 2  |U3| deflection    (mm)
    ch 3  surviving-material mask (0/1)

This is the native resolution of the FE solution, so the representation is
exact and invertible: a pixel maps to one element and a channel value maps to a
physical quantity through a stored linear range.
"""

__version__ = "3.6.0"
