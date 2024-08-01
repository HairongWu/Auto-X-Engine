#include "autox_nn_ansi.h"

void autox_crop_image(uint8_t *rgb888, uint8_t *cropped, uint16_t im_w, uint16_t im_h, uint16_t x1, uint16_t y1, uint16_t x2, uint16_t y2)
 {
     uint8_t r, g, b;
	 int index = 0;
     for (int h = 0; h < im_h; ++h)
     {
		 if (h < y1 || h > y2-1)
			 continue;

         for (int w = 0; w < im_w*3; w+=3)
         {
			 if (w/3 < x1 || w/3 > x2-1)
				 continue;

			 cropped[index] = rgb888[h*im_w * 3 + w];
			 index++;
			 cropped[index] = rgb888[h*im_w * 3 + w+1];
			 index++;
			 cropped[index] = rgb888[h*im_w * 3 + w+2];
			 index++;
         }
     }
 }