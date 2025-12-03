------------12/3/25 rtsp stream segmentation ------------
@deeplabseg

video src:
rtsp://150.174.3.15:80/mystream

segment and localview
python3 rtsp_seg.py | ffplay     -f rawvideo     -pixel_format bgr24     -video_size 640X360     -framerate 25     -i -

segment and push to rtsp server
python3 rtsp_seg.py | ffmpeg   -f rawvideo   -pix_fmt bgr24   -s 640x360   -r 15   -i -   -c:v libvpx   -b:v 1M   -pix_fmt yuv420p   -f rtsp   -rtsp_transport tcp   rtsp://150.174.3.15:80/overlay

segmented overlay
ffplay rtsp://150.174.3.15:80/overlay
