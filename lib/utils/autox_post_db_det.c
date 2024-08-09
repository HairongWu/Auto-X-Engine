#include "../include/autox_nn.h"

//void order_points_clockwise(pts) {
//    rect = np.zeros((4, 2), dtype = "float32")
//        s = pts.sum(axis = 1)
//        rect[0] = pts[np.argmin(s)]
//        rect[2] = pts[np.argmax(s)]
//        tmp = np.delete(pts, (np.argmin(s), np.argmax(s)), axis = 0)
//        diff = np.diff(np.array(tmp), axis = 1)
//        rect[1] = tmp[np.argmin(diff)]
//        rect[3] = tmp[np.argmax(diff)]
//        return rect
//}
//void clip_det_res(points, img_height, img_width){
//        for pno in range(points.shape[0]) :
//            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
//            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
//            return points
//}
//void filter_tag_det_res(float *dt_boxes, uint16_t img_height, uint16_t img_width) {
//        dt_boxes_new = [];
//    for box in dt_boxes :
//        if type(box) is list :
//        box = np.array(box)
//            box = self.order_points_clockwise(box)
//            box = self.clip_det_res(box, img_height, img_width)
//            rect_width = int(np.linalg.norm(box[0] - box[1]))
//            rect_height = int(np.linalg.norm(box[0] - box[3]))
//            if rect_width <= 3 or rect_height <= 3:
//        continue
//            dt_boxes_new.append(box)
//            dt_boxes = np.array(dt_boxes_new)
//}

void autox_db_postprocess(const float* pred, float* dt_boxes, uint16_t frame_h, uint16_t frame_w, uint16_t height, uint16_t width, float thresh)
{
	int8_t* segmentation = (int8_t*)calloc(height * width, sizeof(int8_t));
	for (int i = 0; i < height * width; i++)
		if (pred[i] > thresh)
			segmentation[i] = 1;

	// float* boxes = boxes_from_bitmap(pred, segmentation,frame_w, frame_h);
	free(segmentation);

	// filter_tag_det_res(boxes, frame_w, frame_h);
}