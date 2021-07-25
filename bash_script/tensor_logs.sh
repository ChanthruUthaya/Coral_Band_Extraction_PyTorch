PORT=$((($UID-6025) % 65274))
echo "Port number = $PORT"
hostname -s
module load "languages/anaconda3/2019.07-3.6.5-tflow-1.14"
echo "modules loaded"
tensorboard --logdir logs --port "$PORT"
