// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "movenet.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

#include "cpu.h"

const int num_joints = 13;

template <class ForwardIterator>
inline static size_t argmax(ForwardIterator first, ForwardIterator last)
{
    return std::distance(first, std::max_element(first, last));
}

MoveNet::MoveNet()
{
    blob_pool_allocator.set_size_compare_ratio(0.f);
    workspace_pool_allocator.set_size_compare_ratio(0.f);
}

int MoveNet::load(const char *modeltype, int _target_size, const float *_mean_vals, const float *_norm_vals, bool use_gpu)
{

    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    poseNet.opt = ncnn::Option();

#if NCNN_VULKAN
    poseNet.opt.use_vulkan_compute = use_gpu;
#endif

    poseNet.opt.num_threads = ncnn::get_big_cpu_count();

    char parampath[256];
    char modelpath[256];
    sprintf(parampath, "../models/%s.param", modeltype);
    sprintf(modelpath, "../models/%s.bin", modeltype);

    poseNet.load_param(parampath);
    poseNet.load_model(modelpath);

    target_size = _target_size;
    mean_vals[0] = _mean_vals[0];
    mean_vals[1] = _mean_vals[1];
    mean_vals[2] = _mean_vals[2];
    norm_vals[0] = _norm_vals[0];
    norm_vals[1] = _norm_vals[1];
    norm_vals[2] = _norm_vals[2];
    if (target_size == 192)
    {
        feature_size = 48;
        kpt_scale = 0.02083333395421505;
    }
    else
    {
        feature_size = 64;
        kpt_scale = 0.015625;
    }
    for (int i = 0; i < feature_size; i++)
    {
        std::vector<float> x, y;
        for (int j = 0; j < feature_size; j++)
        {
            x.push_back(j);
            y.push_back(i);
        }
        dist_y.push_back(y);
        dist_x.push_back(x);
    }
    return 0;
}

void MoveNet::detect_pose(cv::Mat &bgr, std::vector<keypoint> &points)
{
    int w = bgr.cols;
    int h = bgr.rows;
    float scale = 1.f;

    cv::resize(bgr, bgr, cv::Size(192, 192));

    ncnn::Mat in = ncnn::Mat::from_pixels(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, bgr.cols, bgr.rows);

    ncnn::Extractor ex = poseNet.create_extractor();

    ex.input("in0", in);

    ncnn::Mat regress, center, heatmap, offset;

    ex.extract("regress", regress); // 26*48*48
    ex.extract("offset", offset);   // 26*48*48
    ex.extract("heatmap", heatmap); // 13*48*48
    ex.extract("center", center);   // 1*48*48

    float *regress_data = (float *)regress.data;
    float *center_data = (float *)center.data;
    float *heatmap_data = (float *)heatmap.data;
    float *offset_data = (float *)offset.data;

    int top_index = 0;

    top_index = int(argmax(center_data, center_data + center.h * center.w));

    int ct_y = (top_index / feature_size);
    int ct_x = top_index - ct_y * feature_size;

    std::vector<float> y_regress(num_joints), x_regress(num_joints);

    for (size_t i = 0; i < num_joints; i++)
    {
        int x_regress_origin = regress_data[2 * i * feature_size * feature_size + ct_y * feature_size + ct_x] + 0.5;
        int y_regress_origin = regress_data[(2 * i + 1) * feature_size * feature_size + ct_y * feature_size + ct_x] + 0.5;
        x_regress[i] = x_regress_origin + ct_x;
        y_regress[i] = y_regress_origin + ct_y;
    }

    ncnn::Mat kpt_scores = ncnn::Mat(feature_size * feature_size, num_joints, sizeof(float));

    float *scores_data = (float *)kpt_scores.data;
    for (int c = 0; c < num_joints; c++)
    {
        for (int i = 0; i < feature_size; i++)
        {
            for (int j = 0; j < feature_size; j++)
            {
                float y = (dist_y[i][j] - y_regress[c]) * (dist_y[i][j] - y_regress[c]);
                float x = (dist_x[i][j] - x_regress[c]) * (dist_x[i][j] - x_regress[c]);
                float dist_weight = std::sqrt(y + x) + 1.8;
                scores_data[c * feature_size * feature_size + i * feature_size + j] = heatmap_data[c * feature_size * feature_size + i * feature_size + j] / dist_weight;
            }
        }
    }
    std::vector<int> kpts_ys, kpts_xs;
    for (int i = 0; i < num_joints; i++)
    {
        top_index = 0;
        top_index = int(argmax(scores_data + feature_size * feature_size * i, scores_data + feature_size * feature_size * (i + 1)));

        int top_y = (top_index / feature_size);
        int top_x = top_index - top_y * feature_size;
        kpts_ys.push_back(top_y);
        kpts_xs.push_back(top_x);
    }

    points.clear();
    for (int i = 0; i < num_joints; i++)
    {
        float kpt_offset_x = offset_data[2 * i * feature_size * feature_size + kpts_ys[i] * feature_size + kpts_xs[i]];
        float kpt_offset_y = offset_data[(2 * i + 1) * feature_size * feature_size + kpts_ys[i] * feature_size + kpts_xs[i]];

        float x = (kpts_xs[i] + kpt_offset_x) / (target_size / 4);
        float y = (kpts_ys[i] + kpt_offset_y) / (target_size / 4);

        keypoint kpt;
        kpt.x = x * target_size;
        kpt.y = y * target_size;
        kpt.score = heatmap_data[(int)(i * feature_size * feature_size + y_regress[i] * feature_size + x_regress[i])];
        points.push_back(kpt);
    }
}

int MoveNet::draw(cv::Mat &bgr, std::vector<keypoint> &points)
{
    // std::vector<keypoint> points;
    // detect_pose(bgr, points);

    // int skele_index[][2] = {{0, 1}, {0, 2}, {1, 3}, {2, 4}, {0, 5}, {0, 6}, {5, 6}, {5, 7}, {7, 9}, {6, 8}, {8, 10}, {11, 12}, {5, 11}, {11, 13}, {13, 15}, {6, 12}, {12, 14}, {14, 16}};
    int skele_index[][2] = {{0, 1}, {3, 5}, {1, 2}, {1, 3}, {2, 4}, {4, 6}, {7, 8}, {1, 7}, {7, 9}, {9, 11}, {2, 8}, {8, 10}, {10, 12}};
    int color_index[][3] = {
        {255, 0, 0},
        {0, 0, 255},
        {255, 0, 0},
        {0, 0, 255},
        {255, 0, 0},
        {0, 0, 255},
        {0, 255, 0},
        {255, 0, 0},
        {255, 0, 0},
        {0, 0, 255},
        {0, 0, 255},
        {0, 255, 0},
        {255, 0, 0},
        {255, 0, 0},
        {255, 0, 0},
        {0, 0, 255},
        {0, 0, 255},
        {0, 0, 255},
    };
    // cv::resize(bgr, bgr, cv::Size(192, 192));

    for (int i = 0; i < num_joints; i++)
    {
        if (points[skele_index[i][0]].score > 0.1 && points[skele_index[i][1]].score > 0.1)
            cv::line(bgr, cv::Point(points[skele_index[i][0]].x, points[skele_index[i][0]].y),
                     cv::Point(points[skele_index[i][1]].x, points[skele_index[i][1]].y), cv::Scalar(color_index[i][0], color_index[i][1], color_index[i][2]), 2);
    }
    for (int i = 0; i < num_joints; i++)
    {
        if (points[i].score > 0.1)
            cv::circle(bgr, cv::Point(points[i].x, points[i].y), 3, cv::Scalar(100, 255, 150), -1);
    }
    return 0;
}
