import json
import cv2

import Configs
cfg = Configs.Config()


result_path = 'result.json'

with open(result_path, 'r') as f:
    different_view_results = json.load(f)


for video_mode, video in cfg.videos.items():
    cap = cv2.VideoCapture(video)

    this_view_results = different_view_results[video_mode]

    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if this_view_results.get(i):
            poses = this_view_results[i]

            for pos in poses:
                xmin, ymin, xmax, ymax, score = pos

                cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)),
                              (0, 255, 0))
                cv2.imshow(f'{video_mode}', frame)
                cv2.waitKey(0)
        cv2.imshow(f'{video_mode}', frame)
        i += 1

    cap.release()
