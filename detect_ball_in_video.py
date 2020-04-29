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


def detect_ball_above_net_in_video(video, ball_detector, video_mode, cfg):
    cap = cv2.VideoCapture(video)

    for_return = {}
    f = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        ball_results = ball_detector.detection_ball(frame)
        net_height = cfg.net_heights[video_mode]
        final_results = filter_above_net_ball(ball_results, net_height)
        for_return[f] = final_results
        f += 1

    cap.release()
    return for_return


if __name__ == '__main__':
    import Configs
    cfg = Configs.Config()

    my_ball_detector = ball_detector.BallDetector(cfg)

    different_view_results = {}
    for video_mode, video in cfg.videos.items():
        different_view_results[video_mode] = {}

        for video_mode, video in cfg.videos.items():
            poses = detect_ball_above_net_in_video(video, my_ball_detector, video_mode, cfg)

            different_view_results[video_mode] = poses
        
    with open('video_result.json', 'w') as f:
        json.dump(different_view_results, f)

