import tarfile

import cv2
import numpy as np
import io
from PIL import Image


from fractions import Fraction


a=Fraction(1,4)
print(a)
b=Fraction(0.25)
print(b)
exit()
with tarfile.open('tar-sample.tar') as tf:
    print(tf.getnames())
    for tarinfo in tf.getmembers():

        im = tf.extractfile(tarinfo)
        print(im, tarinfo,tarinfo.name)


        # if im:
        #     i = im.read()
        #     img = Image.open(io.BytesIO(i))
        #     img2 = cv2.imdecode(np.fromstring(i, np.uint8), cv2.IMREAD_COLOR)
        #
        #     print(im, member, img, img2.shape,member.offset_data,member.offset,member.size)
