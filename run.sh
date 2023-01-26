CHECKPOINT_DIR='./ckpt_4sq' 
dataset='4sq'       # [gowalla, 4sq]
if [ ! -d $CHECKPOINT_DIR ]
then
    mkdir $CHECKPOINT_DIR
fi

mode='train'
# mode='test'   # Uncomment this line if you test model

gpu_id=0
layer_num=2
max_len=50
input_dim=64    # Do not change the input dimension
hidden_channel=32
batch_size=128
lr=0.001
weight_decay=0.0001
ubias_num=400   # 400 for 4sq, 180 for gow
ibias_num=400   # 400 for 4sq, 180 for gow
neg_sample_num=99

eval_per_n=30000    # Do not change
epoch_num=20

ssl_temp=0.5
mtl_coef_1=0
mtl_coef_2=0

model='TADGAT_full_v2'     # To train HST-HGAT, 'TADGAT' must be contained in model name
# model='MF'
# model='STP_UDGAT'
# model='LSTM'
# model='LightGCN'

python -u train.py \
    --dataset=$dataset \
    --mode=$mode \
    --model=$model \
    --checkpoint_dir=$CHECKPOINT_DIR \
    --gpu_id=$gpu_id \
    --layer_num=$layer_num \
    --max_len=$max_len \
    --input_dim=$input_dim \
    --hidden_channel=$hidden_channel \
    --ubias_num=$ubias_num \
    --ibias_num=$ibias_num \
    --weight_decay=$weight_decay \
    --lr=$lr \
    --batch_size=$batch_size \
    --neg_sample_num=$neg_sample_num \
    --eval_per_n=$eval_per_n \
    --epoch_num=$epoch_num \
    --ssl_temp=$ssl_temp \
    --mtl_coef_1=$mtl_coef_1 \
    --mtl_coef_2=$mtl_coef_2 \
    