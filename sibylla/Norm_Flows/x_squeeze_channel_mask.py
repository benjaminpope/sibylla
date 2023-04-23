# experiment to see what happens when masks are squeezed

from Image_masks import *
from custom_bijectors import Squeeze

starting_mask = ChannelMask((14,14,4))

squeeze_layer = Squeeze(3,3)

final_mask = squeeze_layer.inverse(starting_mask._mask)

final_mask = ImageMask(final_mask)

plt.figure()
final_mask.visualise()
plt.show()
# conclusion: can get away without using squeeze by intelligently constructing ignorance masks such 
# that each column is ignored


starting_mask = ChannelMask((7,7,16))

squeeze_layer = Squeeze(3,3)
final_mask = squeeze_layer.inverse(squeeze_layer.inverse(starting_mask._mask))
final_mask = ImageMask(final_mask)

plt.figure()
final_mask.visualise()
plt.show()