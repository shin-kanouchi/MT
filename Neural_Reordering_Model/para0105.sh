
for u in 25; do
    for w in 1; do
        for d in 0; do
            for p in "NOpre"; do
                for k in "NOTkana"; do
                    for c in "w2v"; do
                        for opt in "Adam"; do
                            size="1000k"
                            epoch=50
                            jaen="en-ja"
                            msrl="msrl"
                            phrase="phrase"
                            null="null"
                            python rae-py3.py -g data0105 -f ${size}-${jaen}-${phrase}-${null}-${p}-c${c}-u${u}-w${w}-d${d}-e${epoch}-${opt}-${msrl} -j ${jaen} -x ${phrase} -n ${null} -p ${p} -c ${c} -w ${w} -u ${u} -d ${d} -a 0.12 -b 25 -e ${epoch} -o ${opt} -m ${msrl} -r ../data/${jaen}/phrase_mslr.over2.pre2.align.ph-lex-score.rmD10.head5000k.${size} -t ../data/${jaen}/phrase_mslr.over2.pre2.align.ph-lex-score.rmD10.tail500k.10k &
                            phrase="NOTphrase"
                            python rae-py3.py -g data0105 -f ${size}-${jaen}-${phrase}-${null}-${p}-c${c}-u${u}-w${w}-d${d}-e${epoch}-${opt}-${msrl} -j ${jaen} -x ${phrase} -n ${null} -p ${p} -c ${c} -w ${w} -u ${u} -d ${d} -a 0.12 -b 25 -e ${epoch} -o ${opt} -m ${msrl} -r ../data/${jaen}/phrase_mslr.over2.pre2.align.ph-lex-score.rmD10.head5000k.${size} -t ../data/${jaen}/phrase_mslr.over2.pre2.align.ph-lex-score.rmD10.tail500k.10k &
                            phrase="NOTphrase"
                            null="NOTnull"
                            python rae-py3.py -g data0105 -f ${size}-${jaen}-${phrase}-${null}-${p}-c${c}-u${u}-w${w}-d${d}-e${epoch}-${opt}-${msrl} -j ${jaen} -x ${phrase} -n ${null} -p ${p} -c ${c} -w ${w} -u ${u} -d ${d} -a 0.12 -b 25 -e ${epoch} -o ${opt} -m ${msrl} -r ../data/${jaen}/phrase_mslr.over2.pre2.align.ph-lex-score.rmD10.head5000k.${size} -t ../data/${jaen}/phrase_mslr.over2.pre2.align.ph-lex-score.rmD10.tail500k.10k &
                            phrase="phrase"
                            null="NOTnull"
                            python rae-py3.py -g data0105 -f ${size}-${jaen}-${phrase}-${null}-${p}-c${c}-u${u}-w${w}-d${d}-e${epoch}-${opt}-${msrl} -j ${jaen} -x ${phrase} -n ${null} -p ${p} -c ${c} -w ${w} -u ${u} -d ${d} -a 0.12 -b 25 -e ${epoch} -o ${opt} -m ${msrl} -r ../data/${jaen}/phrase_mslr.over2.pre2.align.ph-lex-score.rmD10.head5000k.${size} -t ../data/${jaen}/phrase_mslr.over2.pre2.align.ph-lex-score.rmD10.tail500k.10k &

                            jaen="ja-en"
                            phrase="phrase"
                            null="null"
                            python rae-py3.py -g data0105 -f ${size}-${jaen}-${phrase}-${null}-${p}-c${c}-u${u}-w${w}-d${d}-e${epoch}-${opt}-${msrl} -j ${jaen} -x ${phrase} -n ${null} -p ${p} -c ${c} -w ${w} -u ${u} -d ${d} -a 0.12 -b 25 -e ${epoch} -o ${opt} -m ${msrl} -r ../data/${jaen}/phrase_mslr.over2.pre2.align.ph-lex-score.rmD10.head5000k.${size} -t ../data/${jaen}/phrase_mslr.over2.pre2.align.ph-lex-score.rmD10.tail500k.10k &
                            phrase="NOTphrase"
                            python rae-py3.py -g data0105 -f ${size}-${jaen}-${phrase}-${null}-${p}-c${c}-u${u}-w${w}-d${d}-e${epoch}-${opt}-${msrl} -j ${jaen} -x ${phrase} -n ${null} -p ${p} -c ${c} -w ${w} -u ${u} -d ${d} -a 0.12 -b 25 -e ${epoch} -o ${opt} -m ${msrl} -r ../data/${jaen}/phrase_mslr.over2.pre2.align.ph-lex-score.rmD10.head5000k.${size} -t ../data/${jaen}/phrase_mslr.over2.pre2.align.ph-lex-score.rmD10.tail500k.10k &
                            phrase="NOTphrase"
                            null="NOTnull"
                            python rae-py3.py -g data0105 -f ${size}-${jaen}-${phrase}-${null}-${p}-c${c}-u${u}-w${w}-d${d}-e${epoch}-${opt}-${msrl} -j ${jaen} -x ${phrase} -n ${null} -p ${p} -c ${c} -w ${w} -u ${u} -d ${d} -a 0.12 -b 25 -e ${epoch} -o ${opt} -m ${msrl} -r ../data/${jaen}/phrase_mslr.over2.pre2.align.ph-lex-score.rmD10.head5000k.${size} -t ../data/${jaen}/phrase_mslr.over2.pre2.align.ph-lex-score.rmD10.tail500k.10k &
                            phrase="phrase"
                            null="NOTnull"
                            python rae-py3.py -g data0105 -f ${size}-${jaen}-${phrase}-${null}-${p}-c${c}-u${u}-w${w}-d${d}-e${epoch}-${opt}-${msrl} -j ${jaen} -x ${phrase} -n ${null} -p ${p} -c ${c} -w ${w} -u ${u} -d ${d} -a 0.12 -b 25 -e ${epoch} -o ${opt} -m ${msrl} -r ../data/${jaen}/phrase_mslr.over2.pre2.align.ph-lex-score.rmD10.head5000k.${size} -t ../data/${jaen}/phrase_mslr.over2.pre2.align.ph-lex-score.rmD10.tail500k.10k 
                            wait
                        done
                    done
                done
            done
        done
    done
done

