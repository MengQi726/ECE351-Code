# Part2-9.
import numpy as np
import numpy

cRect = 2 + 3j
print ( cRect )

cPol = abs( cRect ) * np . exp (1j * np . angle ( cRect ) )
print ( cPol ) # notice Python will store this in rectangular form

cRect2 = np . real ( cPol ) + 1j * np . imag ( cPol )
print ( cRect2 ) # converting from polar to rectangular


print (numpy.sqrt (3*5 - 5*5))


print (numpy.sqrt (3*5 - 5*5 + 0j ))