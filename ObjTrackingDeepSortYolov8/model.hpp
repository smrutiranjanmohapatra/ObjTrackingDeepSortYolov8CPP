#ifndef MODEL_HPP
#define MODEL_HPP

#include <algorithm>
#include "datatype.h"


// * Each rect's data structure.
// * tlwh: topleft point & (w,h)
// * confidence: detection confidence.
// * feature: the rect's 256d feature.
// */

const float kRatio = 0.5;
enum DETECTBOX_IDX { IDX_X = 0, IDX_Y, IDX_W, IDX_H };

//// Assuming you have a bounding box in the format (x1, y1, x2, y2)
//DETECTBOX convert_to_tlwh(const BoundingBox& bbox) {
//    DETECTBOX tlwh;
//
//    // Extract coordinates from the BoundingBox
//    float x1 = bbox.x1;
//    float y1 = bbox.y1;
//    float x2 = bbox.x2;
//    float y2 = bbox.y2;
//
//    // Calculate width and height
//    float width = x2 - x1;
//    float height = y2 - y1;
//
//    // Set the values into the DETECTBOX format (x, y, w, h)
//    tlwh(0, 0) = x1;          // x-coordinate of the top-left corner
//    tlwh(0, 1) = y1;          // y-coordinate of the top-left corner
//    tlwh(0, 2) = width;       // width of the bounding box
//    tlwh(0, 3) = height;      // height of the bounding box
//
//    return tlwh;
//}

class DETECTION_ROW {
public:
    DETECTBOX tlwh;
    float confidence;
    FEATURE feature;
    DETECTBOX to_xyah() const {
        //(centerx, centery, aspect ratio, height)
        DETECTBOX ret = tlwh;
        ret(0, IDX_X) += (ret(0, IDX_W) * kRatio);
        ret(0, IDX_Y) += (ret(0, IDX_H) * kRatio);
        ret(0, IDX_W) /= ret(0, IDX_H);
        return ret;
    }
    DETECTBOX to_tlbr() const {
        //(top-left x, top-left y, bottom-right x, bottom-right y))
        DETECTBOX ret = tlwh;
        ret(0, IDX_X) += ret(0, IDX_W);
        ret(0, IDX_Y) += ret(0, IDX_H);
        return ret;
    }
};

typedef std::vector<DETECTION_ROW> DETECTIONS;
typedef std::pair<std::vector<CLSCONF>, DETECTIONS> DETECTIONSV2;

#endif // MODEL_HPP