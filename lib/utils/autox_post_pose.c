#include "../include/autox_nn.h"

float get_max(float* arr, int n)
{
	float res = arr[0];
	for (int i = 1; i < n; i++) {
		if (res < arr[i])
			res = arr[i];
	}
	return res;
}
void gaussian_blur(float* heatmap, int kernel, int num_joints, int height, int width)
{
	int border = (kernel - 1) % 2;
	for (int j = 0; j < num_joints; j++)
	{
		float origin_max = get_max(heatmap + j * height * width, height * width);
		//float* dr = (float*)calloc((height + 2 * border) * (width + 2 * border), sizeof(float));
		//memcpy(dr + border + (j + border) * (height * width + 2 * border), heatmap + j * height * width, height * width * sizeof(float));
		//dr = blur_cpu(dr, 0, width + 2 * border, height + 2 * border, 1);
		//memcpy(heatmap + j * height * width, dr + border + (j + border) * (height * width + 2 * border), height * width * sizeof(float));
		//float new_max = get_max(heatmap + j * height * width, height * width);
		/*for (int i = 0; i < height * width; i++)
			heatmap[j * height * width + i] *= origin_max / new_max;*/
	}
}

void dark_parse(float* hm, float* coord, int heatmap_height, int heatmap_width)
{
	int px = (int)(coord[0]);
	int py = (int)(coord[1]);
	if (1 < px < heatmap_width - 2 && 1 < py < heatmap_height - 2)
	{
		float dx = 0.5 * (hm[py * heatmap_width + (px + 1)] - hm[py * heatmap_width + px - 1]);
		float dy = 0.5 * (hm[(py + 1) * heatmap_width + px] - hm[py - 1 + px]);
		float dxx = 0.25 * (hm[py * heatmap_width + px + 2] - 2 * hm[py * heatmap_width + px] + hm[py * heatmap_width + px - 2]);
		float dxy = 0.25 * (hm[(py + 1) * heatmap_width + px + 1] - hm[(py - 1) * heatmap_width + px + 1] - hm[(py + 1) * heatmap_width + px - 1] \
			+ hm[(py - 1) * heatmap_width + px - 1]);
		float dyy = 0.25 * (
			hm[(py + 2 * 1) * heatmap_width + px] - 2 * hm[py * heatmap_width + px] + hm[(py - 2 * 1) * heatmap_width + px]);
		float derivative[2] = { dx,dy };
		//float hessian[4] = np.matrix([[dxx, dxy], [dxy, dyy]]);
		if (dxx * dyy - powf(dxy, 2) != 0)
		{
			float coef = 1 / (dxx * dyy - powf(dxy, 2));
			float hessianinv[4] = { dyy * coef,-dxy * coef,-dxy * coef,dxx * coef };
			float offset_x = -(hessianinv[0] * derivative[0] + hessianinv[1] * derivative[1]);
			float offset_y = -(hessianinv[2] * derivative[0] + hessianinv[3] * derivative[1]);
			coord[0] += offset_x;
			coord[1] += offset_y;
		}
	}
}

void dark_postprocess(float* hm, float* coords, int kernelsize, int num_joints, int height, int width)
{
	gaussian_blur(hm, kernelsize, num_joints, height, width);
	for (int i = 0; i < num_joints * height * width; i++)
	{
		hm[i] = hm[i] > 1e-10 ? hm[i] : 1e-10;
		hm[i] = logf(hm[i]);
	}
	for (int p = 0; p < num_joints; p++)
		dark_parse(hm + p * height * width, coords + p * 2, height, width);
}
int get_max_preds(float* heatmaps, float* preds, uint16_t num_joints, uint16_t height, uint16_t width)
{
	float* idx = preds;
	float* maxvals = (float*)calloc(num_joints, sizeof(float));
	for (int i = 0; i < num_joints; i++)
	{
		maxvals[i] = heatmaps[i * width * height];
		idx[i] = 0;
		for (int j = 1; j < width * height; j++)
		{
			float temp = heatmaps[i * width * height + j];
			if (temp > maxvals[i])
			{
				maxvals[i] = temp;
				idx[i] = j;
			}
		}
	}

	float* idx2 = preds + num_joints;
	memcpy(idx2, idx, num_joints * sizeof(uint32_t));

	for (int i = 0; i < num_joints; i++)
	{
		float pred_mask = 0;
		if (maxvals[i] > 0.0)
			pred_mask = 1;

		idx[i] = ((int)idx[i] % width) * pred_mask;
		idx2[i] = floor(idx2[i] / width) * pred_mask;
	}

	return 0;
}

//void affine_transform(pt, t)
//{
//	new_pt = np.array([pt[0], pt[1], 1.]).T
//	new_pt = np.dot(t, new_pt)
//}
//
//void transform_preds(float* coords, float* center, float* scale, uint16_t* output_size)
//{
//	get_affine_transform(coords, center, scale * 200, 0, output_size, 1);
//	for(int p=0; p<coords.shape[0];p++)
//		target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans);
//}

void post_pose(float* heatmap, float* coord, uint16_t num_joints, uint16_t height, uint16_t width)
{
	get_max_preds(heatmap, coord, num_joints, height, width);
	dark_postprocess(heatmap, coord, 3, num_joints, height, width);
}