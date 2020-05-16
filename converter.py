import sys
import numpy as np
import cv2

def convert(blue,green,red):
    color = np.uint8([[[blue,green,red]]])
    hsv_color = cv2.cvtColor(color,cv2.COLOR_BGR2HSV)

    hue = hsv_color[0][0][0]

    lower_range = np.array([hue-10,100,100],dtype = np.uint8)
    upper_range = np.array([hue+10,255,255],dtype = np.uint8)

    return lower_range,upper_range

def main():
    blue = sys.argv[1]
    green = sys.argv[2]
    red = sys.argv[3]
    lower,upper = convert(blue,green,red)
    print(f'{lower}\n{upper}')


if __name__ == '__main__':
    main()


