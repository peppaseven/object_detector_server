NUM_WORKER=1
BIND_ADDR=127.0.0.1:30000
gunicorn -w $NUM_WORKER -b $BIND_ADDR -p gunicorn.pid object_classify_svr:app --log-level=debug> engine_classify.log &
