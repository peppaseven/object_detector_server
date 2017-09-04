NUM_WORKER=2
BIND_ADDR=0.0.0.0:30001
gunicorn -w $NUM_WORKER -b $BIND_ADDR -p gunicorn.pid object_detector_svr:app --log-level=debug> engine_detect.log&
