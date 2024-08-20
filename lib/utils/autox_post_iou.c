#include "../include/autox_nn.h"

float cal_overlap(const float* box1, const float* box2)
{
	float xmin1 = box1[0];
	float ymin1 = box1[1];
	float xmax1 = box1[2];
	float ymax1 = box1[3];
	float xmin2 = box2[0];
	float ymin2 = box2[1];
	float xmax2 = box2[2];
	float ymax2 = box2[3];

	float s1 = (xmax1 - xmin1) * (ymax1 - ymin1);
	float s2 = (xmax2 - xmin2) * (ymax2 - ymin2);

	float xmin = max(xmin1, xmin2);
	float ymin = max(ymin1, ymin2);
	float xmax = min(xmax1, xmax2);
	float ymax = min(ymax1, ymax2);

	float w = max(0, xmax - xmin);
	float h = max(0, ymax - ymin);
	float a1 = w * h;
	return a1 / max(s1, s2);
}
