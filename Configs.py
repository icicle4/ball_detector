import os


class Config:
    def __init__(self):
        self.root = os.path.join(os.getcwd(), 'src')
        self.task_name = 'LaoZi'
        self.task_root = os.path.join(self.root,  self.task_name)

        self.ball_vis_dir = os.path.join(self.root, 'ball_vis')

        self.camera_ids = [0, 4]
        self.ball_camera_ids = [0, 4]

        self.use_gpu = True

        # ball_detector config
        self.ball_config_file = '/content/ball_detector/faster_rcnn_r50_fpn_1x_ball_blur.py'
        self.ball_checkpoint_file = '/content/drive/My Drive/latest.pth'

        self.pass_time_file = '/content/drive/My Drive/pass_time.json'

        self.videos = {
            0: "/content/drive/My Drive/0.mp4",
            4: "/content/drive/My Drive/4.mp4"
        }

        self.net_heights = {
            0: 278,
            4: 395
        }




