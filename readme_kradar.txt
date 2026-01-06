alien3:
~/Documents/datasets/k-radar/
https://github.com/kaist-avelab/K-Radar.git

(kradar) student@alien3:~/Documents/K-Radar$ python datasets/kradar_detection_v2_1.py 


------- polar (range-angle) bev from drea -------------------
(kradar) student@alien3:~/Documents/K-Radar$ python radar_drae_bev.py
bev.shape (256, 107)
img.shape (256, 107)
Saved BEV image to: tesseract_00417_bev.png, arr_drea.shape (64, 256, 37, 107) arr_drea[0] []
(kradar) student@alien3:~/Documents/K-Radar$ eog tesseract_00417_bev.png


------- cartesian bev from zyx -------------------
(kradar) student@alien3:~/Documents/K-Radar$ python radar_zyx_bev.py
Saved BEV image to: /tmp/cube_00012_bev.png, arr_zyx.shape (150, 400, 250) arr_zyx[0] [[-1.00000000e+00 -1.00000000e+00 -1.00000000e+00 -1.00000000e+00
	eog /tmp/cube_00012_bev.png
