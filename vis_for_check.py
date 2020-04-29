import json
import cv2

import Configs
cfg = Configs.Config()

result_path = 'result.json'

with open(result_path, 'r') as f:
    different_view_results = json.load(f)

print(different_view_results)

for video_mode, video in cfg.videos.items():
    cap = cv2.VideoCapture(video)

    this_view_results = different_view_results[str(video_mode)]

    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if this_view_results.get(str(i)):
            poses = this_view_results[str(i)]

            for pos in poses:
                xmin, ymin, xmax, ymax, score = pos

                cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)),
                              (0, 255, 0))
                cv2.putText(frame, str(i), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), thickness=2)
                frame = cv2.resize(frame, (1080, 720))
                cv2.imshow(f'{video_mode}', frame)
                cv2.imwrite(f'{i}.jpg', frame)
                cv2.waitKey(0)

        #frame = cv2.resize(frame, (1080, 720))
        #cv2.putText(frame, str(i), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), thickness=2)
        #cv2.imshow(f'{video_mode}', frame)
        #cv2.waitKey(1)
        i += 1

    cap.release()
