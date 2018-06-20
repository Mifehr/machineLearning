
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import Bbox


def my_plot(x, y, options=dict(), name='current_Plot'):
    'Unform, centered plots with exportation to the folder of the running file.'

    fig = plt.figure(figsize = options.get('figsize', (6, 3.5)))
    ax = fig.subplots()

    # options are given as dictionary in the form: 
    # options = {'marker': 'o', 'color': 'C1', ...}
    ax.plot(x, y, 
            marker = options.get('marker', 'o'),
        markersize = options.get('markersize', 2),
             color = options.get('color', 'C1'),
         linestyle = options.get('linestyle', 'None'))
    
    ax.set_xlabel(options.get('xlabel', 'x'))
    ax.set_ylabel(options.get('ylabel', 'y'))
    ax.set_title(options.get('title', 'Data'))
    plt.tight_layout()

    axpo = Bbox.get_points(ax.get_position())
    axponew = Bbox.from_extents(axpo[0,0], axpo[0,1], 1-axpo[0,0], axpo[1,1])

    ax.set_position(axponew)
    plt.show()
    
    if name == 'current_Plot':
        if options.get('title', 'Data') == 'Data':
            print('Neither export name nor plot title set. '
                'Exported as <current_Plot.pdf>.')
        else:
            print('No export name set. Exported as <' + options.get('title') +
            '.pdf>.')
            name = options.get('title')
    fig.savefig(name + '.pdf')

