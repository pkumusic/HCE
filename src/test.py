import math
 
import matplotlib.pyplot
 
import numpy
 
xs = numpy.arange(-math.pi, math.pi, 0.1)
ys1 = numpy.sin(xs)
ys2 = numpy.cos(xs)
 
figure = matplotlib.pyplot.figure(figsize=(8.0, 5.0))
# The size of the figure is specified as (width, height) in inches
 
axes = figure.add_axes([0.1, 0.25, 0.8, 0.65])
# [left, bottom, width, height], all the quantities are fractions of figure
# width and height
 
axes.grid(True)
 
plot1 = axes.plot(xs, ys1, "r-", lw=2)
plot2 = axes.plot(xs, ys2, "g-", lw=2)
 
axes.set_title("Trigonometric functions: $\\sin$ and $\\cos$")
 
axes.set_xlabel("$x$")
axes.set_ylabel("$y$")
 
figure.legend((plot1, plot2), ("$\\sin(x)$", "$\\cos(x)$"), 'lower center')
# This is a *figure* legend, so it can appear outside the axes
 
# Or you could use an axes legend, within the axes:
# axes.legend((plot1, plot2), ("$\\sin(x)$", "$\\cos(x)$"), 'lower center')
 
matplotlib.pyplot.show()
figure.savefig("foo.eps", format="svg")