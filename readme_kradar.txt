
---------------4/7/26 radar_corp data ----------------
/data/jwang/datasets/datar/radarstuff/20190813_scorp_dataset/20190813_icmim_dataset/2019-08-13-14-04-36/angle_range_numpy

>>> x.shape
(128, 256)
>>> np.max(x)
np.float32(103528.86)
>>> np.min(x)
np.float32(36.0052)

/data/jwang/Documents/K-Radar$ python3 radar_rdr_polar.py --polar_file /data/jwang/datasets/k-radar/18/rdr_polar_3d
	converting rdr_polar_3d 3d npy to bev npy and save to angle_range_numpy/, needed by the training script of rad_segformer/

---------4/2/26 code repo ---------------------
alien3, i9ub22, gpuhead2

----- 4/2/26 dataset ---------------------------
alien3:
~/Documents/datasets/k-radar/
https://github.com/kaist-avelab/K-Radar.git

i9ub22:
~/gpudata/jwang/datasets/k-radar
pguhead2:
/data/jwang/datasets/k-radar/RadarTensor
   rdr_polar_3d/*.zip: (2, 256, 107, 37) npy files for all sequence, converted from raw data
   from_rdr_polar_3d/
   from_rdr_cube_xyz/
   polar3d_bev/
        samples local converted to png from rdr_polar_3d data

raw data from:
	/home/student/Documents/datasets/k-radar/1/radar_tesseract/tesseract_00012.mat

	k-radar/RadarTensor/:  processed data from the raw data, avia google share
			rdr_polar_3d/: (2, 256, 107, 37), ch 1 avg pw, ch2 avg doppler

	unfortunately only seq 1, 58 raw data avialabe at google share

(kradar) student@alien3:~/Documents/K-Radar$ python datasets/kradar_detection_v2_1.py 
	input: /home/student/Documents/datasets/k-radar/1/radar_tesseract/tesseract_00012.mat
		
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

	computer  polar3d_00624.npy avg power measure to png image and bev npy, use_log=True, otherwise signal range too wide and you only see the strongest return.
	output: angle_range_png/ angle_range_numpy/
	mv angle_range_png/polar3d*.png polar3d_bev/
	eog polar3d_bev/polar3d_00065_63_bev.png
	new_all/1/: regenerated from raw tensor, add individual file per dopplar channel.

/Documents/datasets/k-radar/RadarTensor/rdr_polar_3d/1
(2, 256, 107, 37)
first 256,107,37): pw measure over rae but only take radar return from moving object. see datasets/kradar_detection_v2_1.py
 polar3d_00319_bev.png

