alien3:
~/Documents/datasets/k-radar/
https://github.com/kaist-avelab/K-Radar.git

(kradar) student@alien3:~/Documents/K-Radar$ python datasets/kradar_detection_v2_1.py 
	generate: ~/Documents/datasets/k-radar/RadarTensor/rdr_polar_3d/new_all/1/*.npy
	raw power, wide range, normalized by e11,

	use radar_rdr_polar.py to convert to png. use_log true

	get_cube_polar take non static slices: tesseract = dict_item['tesseract'][1:,:,:,:]/normalizer
	so non-moving object radar return is kind of filtered?



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

----------- polar 3d data -----------
(kradar) student@alien3:~/Documents/K-Radar$ python radar_rdr_polar.py --polar_file ~/Documents/datasets/k-radar/RadarTensor/rdr_polar_3d/new_all/1
	convert npy power measure to png image, use_log=True, otherwise signal range too wide and you only see the strongest return.

/Documents/datasets/k-radar/RadarTensor/rdr_polar_3d/1
(2, 256, 107, 37)
first 256,107,37): pw measure over rae but only take radar return from moving object. see datasets/kradar_detection_v2_1.py
eog polar3d_00319_bev.png
