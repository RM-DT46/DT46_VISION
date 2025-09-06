#ifndef ARMOR_DETECTOR_OPENCV_HPP
#define ARMOR_DETECTOR_OPENCV_HPP

#include <opencv2/opencv.hpp>
#include <tuple>
#include <iostream>
#include <vector>
#include <cmath>
#include <utility>
#include <set>
#include "rm_detector/number_classifier.hpp"

using namespace cv;
using namespace std;

namespace DT46_VISION {

    double calculate_distance(const cv::Point2f& p1, const cv::Point2f& p2);
    std::pair<cv::Size2f, double> adjust(const cv::Size2f& w_h, double angle);
    double angle_to_slope(double angle_degrees);

    class Light {
    public:
        int cx;
        int cy;
        double height;
        cv::Point2f up;
        cv::Point2f down;
        int color;

        Light(const cv::Point2f& up, const cv::Point2f& down, int color);
    };

    class Armor {
    public:

        float height_multiplier;

        cv::Point2f light1_up;
        cv::Point2f light1_down;
        cv::Point2f light2_up;
        cv::Point2f light2_down;

        int color;
        int armor_id;
        NumberClassifier::Result res;

        Armor(float height_multiplier, const Light& light1, const Light& light2, NumberClassifier::Result res);
        int get_id() const;
    };

    // 定义 Params 结构体
    struct Params {
        int light_area_min;
        double light_h_w_ratio;
        int light_angle_min;
        int light_angle_max;
        double light_red_ratio;
        double light_blue_ratio;
        double height_rate_tol;
        double height_multiplier_min;
        double height_multiplier_max;
        int bin_offset;
    };

    class ArmorDetector {
    public:
        cv::Mat img;
        cv::Mat img_binary;
        cv::Mat img_armor;
        cv::Mat img_armor_processed;
        cv::Mat img_drawn;

        std::vector<Light> lights;
        std::vector<Armor> armors;

        int binary_val;
        int color;
        int display_mode;
        
        Params params;

        cv::Point2f dst_armor_pts[4] = {
            cv::Point2f(0, 0),
            cv::Point2f(0, int(57 / 0.45f)),
            cv::Point2f(133, int(57 / 0.45f)),
            cv::Point2f(133, 0)
        };
        // 134mm x 58 (125)

        ArmorDetector(int detect_color, int display_mode, int binary_val, const Params& params);
        void set_classifier(std::shared_ptr<NumberClassifier> cls){
            classifier_ = std::move(cls);
        }

        void update_light_area_min(int new_light_area_min);
        void update_light_h_w_ratio(double new_light_h_w_ratio);
        void update_light_angle_min(int new_light_angle_min);
        void update_light_angle_max(int new_light_angle_max);
        void update_light_red_ratio(float new_light_red_ratio);
        void update_light_blue_ratio(float new_light_blue_ratio);

        void update_height_rate_tol(float new_height_rate_tol);
        void update_height_multiplier_min(float new_height_multiplier_min);
        void update_height_multiplier_max(float new_height_multiplier_max);
        void update_bin_offset(int new_bin_offset);

        void update_binary_val(int new_binary_val);
        void update_detect_color(int new_color);
        void update_display_mode(int new_display_mode);

        cv::Mat process(const cv::Mat& img_input);
        std::vector<Light> find_lights(const cv::Mat& img_binary_input);
        std::pair<cv::Point2f, cv::Point2f> stretch_point(const cv::Point2f& up, const cv::Point2f& down, float ratio = 0.45f) const;
        NumberClassifier::Result get_armor_result(const Light& light1, const Light& light2);
        std::pair<NumberClassifier::Result, float> is_close(const Light& light1, const Light& light2);
        std::vector<Armor> is_armor(const std::vector<Light>& lights);
        cv::Mat draw_rect(cv::Mat img_draw);
        cv::Mat draw_lights(cv::Mat img_draw);
        cv::Mat draw_armors(cv::Mat img_draw);
        cv::Mat draw_img();
        std::tuple<cv::Mat, cv::Mat, cv::Mat, cv::Mat> display();
        std::vector<Armor> detect_armors(const cv::Mat& img_input);
        
        std::shared_ptr<NumberClassifier> classifier_; 
    };
    
}

#endif // ARMOR_DETECTOR_OPENCV_HPP