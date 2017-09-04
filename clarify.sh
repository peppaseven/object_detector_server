NUM_WORKER=2
BIND_ADDR=0.0.0.0:30000
gunicorn -w $NUM_WORKER -b $BIND_ADDR -p gunicorn.pid object_classify_svr:app --log-level=debug> engine_classify.log &
