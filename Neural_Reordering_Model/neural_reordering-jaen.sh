#!/bin/sh
#-*-coding:utf-8-*-

export ALAGIN_HOME=/work/kanouchi/ALAGIN/alagin2015_binary.x86_64-Linux
export SCRIPTS_ROOTDIR=${ALAGIN_HOME}/scripts
export KFTT_DATA=/work/kanouchi/ALAGIN/kftt-data-1.0
export LD_LIBRARY_PATH=${ALAGIN_HOME}/gcc/lib64:${ALAGIN_HOME}/gcc/lib:${ALAGIN_HOME}/lib
SHIN=/work/kanouchi/git/shin/MT
TRAVATARDIR=/work/kanouchi/ALAGIN/alagin2015_binary.x86_64-Linux/bin
CORPUS=/work/kanouchi/git/shin/MT/exp_KFTT

model=$1 #Output name
msd=$2 #msd or msrl or rl
epoch=4

#作成したエポックごとのモデルを使いmosesのn-bestを入力として並び替えモデルのところだけを書き換える（test-data）
echo "python rae_decoding-py3.py -f $model -m $msd -t ${CORPUS}/kyoto-test.1000best.ja-en -s ${CORPUS}/test/kyoto-test.en -o ${SHIN}/result/kyoto-test.1000best.ja-en.$model"
python rae_decoding-py3.py -n NOTnull -u No -f $model -e 1 -m $msd -t ${CORPUS}/kyoto-test.1000best.ja-en -s ${CORPUS}/test/kyoto-test.ja -o ${SHIN}/result/kyoto-test.1000best.ja-en.$model.1 &
python rae_decoding-py3.py -n NOTnull -u No -f $model -e 2 -m $msd -t ${CORPUS}/kyoto-test.1000best.ja-en -s ${CORPUS}/test/kyoto-test.ja -o ${SHIN}/result/kyoto-test.1000best.ja-en.$model.2 &
python rae_decoding-py3.py -n NOTnull -u No -f $model -e 3 -m $msd -t ${CORPUS}/kyoto-test.1000best.ja-en -s ${CORPUS}/test/kyoto-test.ja -o ${SHIN}/result/kyoto-test.1000best.ja-en.$model.3 &
python rae_decoding-py3.py -n NOTnull -u No -f $model -e 4 -m $msd -t ${CORPUS}/kyoto-test.1000best.ja-en -s ${CORPUS}/test/kyoto-test.ja -o ${SHIN}/result/kyoto-test.1000best.ja-en.$model.4
wait

#作成したエポックごとのモデルを使いmosesのn-bestを入力として並び替えモデルのところだけを書き換える（tune-data）
echo "python python rae_decoding-py3.py -f $model -m $msd -t ${CORPUS}/kyoto-dev.1000best.ja-en -s ${CORPUS}/test/kyoto-dev.ja -o ${SHIN}/result/kyoto-dev.1000best.ja-en.$model.1-4"
python rae_decoding-py3.py -n NOTnull -u No -f $model -e 1 -m $msd -t ${CORPUS}/kyoto-dev.1000best.ja-en -s ${CORPUS}/test/kyoto-dev.ja -o ${SHIN}/result/kyoto-dev.1000best.ja-en.$model.1 &
python rae_decoding-py3.py -n NOTnull -u No -f $model -e 2 -m $msd -t ${CORPUS}/kyoto-dev.1000best.ja-en -s ${CORPUS}/test/kyoto-dev.ja -o ${SHIN}/result/kyoto-dev.1000best.ja-en.$model.2 &
python rae_decoding-py3.py -n NOTnull -u No -f $model -e 3 -m $msd -t ${CORPUS}/kyoto-dev.1000best.ja-en -s ${CORPUS}/test/kyoto-dev.ja -o ${SHIN}/result/kyoto-dev.1000best.ja-en.$model.3 &
python rae_decoding-py3.py -n NOTnull -u No -f $model -e 4 -m $msd -t ${CORPUS}/kyoto-dev.1000best.ja-en -s ${CORPUS}/test/kyoto-dev.ja -o ${SHIN}/result/kyoto-dev.1000best.ja-en.$model.4
wait

for i in `seq 1 $epoch`; do
    #書き換えたチューニングデータを使い，mertを回して素性の重みを調整し直す
    echo "rerank-tune epoch.$i"
    ${TRAVATARDIR}/batch-tune -threads 4 -weight_in ${SHIN}/tune/moses.rae.$msd.weights.ja-en -nbest ${SHIN}/result/kyoto-dev.1000best.ja-en.$model.$i ${CORPUS}/test/kyoto-dev.en > ${SHIN}/tune/shin.tune.weights.ja-en.$model.$i

    #調整した重みを使ってリランキングし直す
    echo "rerank-test epoch.$i"
    ${TRAVATARDIR}/rescorer -weight_in ${SHIN}/tune/shin.tune.weights.ja-en.$model.$i -nbest ${SHIN}/result/kyoto-test.1000best.ja-en.$model.$i > ${SHIN}/result/kyoto-test.1best.ja-en.$model.$i

    #BLEUの計算
    echo 'evaluate'
    ${TRAVATARDIR}/mt-evaluator -ref ${CORPUS}/test/kyoto-test.en -eval 'bleu ribes' ${SHIN}/result/kyoto-test.1best.ja-en.$model.$i > ${SHIN}/score/score.$model.$i
done
