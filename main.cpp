#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <net.h>
#include <movenet.h>
#include <benchmark.h>

static int draw_unsupported(cv::Mat &rgb)
{
    const char text[] = "unsupported";

    int baseLine = 0;
    cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 1.0, 1, &baseLine);

    int y = (rgb.rows - label_size.height) / 2;
    int x = (rgb.cols - label_size.width) / 2;

    cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                  cv::Scalar(255, 255, 255), -1);

    cv::putText(rgb, text, cv::Point(x, y + label_size.height),
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 0));

    return 0;
}

static int draw_fps(cv::Mat &rgb)
{
    // resolve moving average
    float avg_fps = 0.f;
    {
        static double t0 = 0.f;
        static float fps_history[10] = {0.f};

        double t1 = ncnn::get_current_time();
        if (t0 == 0.f)
        {
            t0 = t1;
            return 0;
        }

        float fps = 1000.f / (t1 - t0);
        t0 = t1;

        for (int i = 9; i >= 1; i--)
        {
            fps_history[i] = fps_history[i - 1];
        }
        fps_history[0] = fps;

        if (fps_history[9] == 0.f)
        {
            return 0;
        }

        for (int i = 0; i < 10; i++)
        {
            avg_fps += fps_history[i];
        }
        avg_fps /= 10.f;
    }

    char text[32];
    sprintf(text, "FPS=%.2f", avg_fps);

    int baseLine = 0;
    cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

    int y = 0;
    int x = rgb.cols - label_size.width;

    cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                  cv::Scalar(255, 255, 255), -1);

    cv::putText(rgb, text, cv::Point(x, y + label_size.height),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));

    return 0;
}

int image_demo(MoveNet &posetracker, const char *imagepath)
{
    std::vector<cv::String> filenames;
    cv::glob(imagepath, filenames, false);

    for (auto img_name : filenames)
    {
        std::vector<keypoint> points;
        cv::Mat image = cv::imread(img_name);
        if (image.empty())
        {
            fprintf(stderr, "cv::imread %s failed\n", img_name);
            return -1;
        }
        posetracker.detect_pose(image, points);
        posetracker.draw(image, points);
        cv::imwrite("../output/result.png", image);
    }
    return 0;
}
int webcam_demo(MoveNet &posetracker, int cam_id)
{
    cv::Mat bgr;
    cv::VideoCapture cap(cam_id, cv::CAP_V4L2);

    int image_id = 0;
    while (true)
    {
        cap >> bgr;
        std::vector<keypoint> points;
        posetracker.detect_pose(bgr, points);
        posetracker.draw(bgr, points);
        cv::imshow("test", bgr);
        cv::waitKey(1);
    }
    return 0;
}

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        fprintf(stderr, "usage: %s [mode] [path]. \n For webcam mode=0, path is cam id; \n For image demo, mode=1, path=xxx/xxx/*.jpg; \n For video, mode=2; \n For benchmark, mode=3 path=0.\n", argv[0]);
        return -1;
    }
    const char *modeltypes[] =
        {
            "movenet",
        };

    const int target_sizes[] =
        {
            192,
        };

    const float mean_vals[][3] =
        {
            {127.5f, 127.5f, 127.5f},
        };

    const float norm_vals[][3] =
        {
            {1 / 127.5f, 1 / 127.5f, 1 / 127.5f},
        };

    int modelid = 0;
    const char *modeltype = modeltypes[(int)modelid];
    std::cout << modeltype << std::endl;
    int target_size = target_sizes[(int)modelid];
    bool use_gpu = false;

    MoveNet posetracker;
    posetracker.load(modeltype, target_size, mean_vals[(int)modelid], norm_vals[(int)modelid], use_gpu);

    int mode = atoi(argv[1]);
    switch (mode)
    {
    case 0:
    {
        int cam_id = atoi(argv[2]);
        webcam_demo(posetracker, cam_id);
        break;
    }
    case 1:
    {
        const char *images = argv[2];
        image_demo(posetracker, images);
        break;
    }

    default:
    {
        fprintf(stderr, "usage: %s [mode] [path]. \n For webcam mode=0, path is cam id; \n For image demo, mode=1, path=xxx/xxx/*.jpg; \n", argv[0]);
        break;
    }
    }
}
