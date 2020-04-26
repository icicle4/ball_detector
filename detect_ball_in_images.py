import json
import cv2

import ball_detector


def filter_above_net_ball(ball_results, net_heigt):

    filtered_results = list()
    for ball_result in ball_results:
        xmin, ymin, xmax, ymax, score = ball_result

        if ymin < net_heigt:
            filtered_results.append(
                ball_result
            )
    return filtered_results


def detect_ball_above_net_in_time(video, time, ball_detector, video_mode, cfg):
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, time)
    _, frame = cap.read()
    ball_results = ball_detector.detection_ball(frame)

    net_height = cfg.net_heights[video_mode]
    final_results = filter_above_net_ball(ball_results, net_height)

    cap.release()
    return final_results


if __name__ == '__main__':
    import Configs
    cfg = Configs.Config()

    with open(cfg.pass_time_file, 'r') as f:
        time_and_poses = json.load(f)

    my_ball_detector = ball_detector.BallDetector(cfg)

    different_view_results = {}
    for video_mode, video in cfg.videos.items():
        different_view_results[video_mode] = {}

    for time, pos in time_and_poses:
        for video_mode, video in cfg.videos.items():
            poses = detect_ball_above_net_in_time(video, time, my_ball_detector, video_mode, cfg)

            different_view_results[video_mode].update({time: poses})

    with open('result.json', 'w') as f:
        json.dump(different_view_results, f)

