#!/usr/bin/env bash

# Usage: plot_training_loss.sh (<log file(s)>)
#
#        if a log file is missed, it will pick a most recent one
#        (that matches *_log*.txt *.log *_train.txt)
#
#        multiple log files can be specified if training has been paused and
#        resumed, hence recording to multiple logs
#
#        it calculates #iter/epoch, based on #training samples, batch_size and
#        #gpu's being used. it assumes solver file being used is the most
#        recent one.
#
#        it does a bit of hand-waving to find right files, so make sure #iter
#        per epoch is correct.

# check if a log file has been changed this frequently
#REFRESHSECOND=10
REFRESHSECOND=3600

# get a unique ID for new filenames
UUID=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 32 | head -n 1)
DIR="$(pwd)"
if [[ $# -ge 1 && -f $1 ]]
then
  LOGFILE="$@"
else
  # latest file matches *_log*.txt, *.log, *_train.txt
  LOGFILE=$(ls -t1 ${DIR}/*_log*.txt ${DIR}/*.log ${DIR}/*_train.txt 2>/dev/null | head -1)
fi

LOSSFILE=/tmp/iter_loss_${UUID}.txt
PLOTFILE=/tmp/iter_loss_${UUID}.png
ACCURACYTMPFILE=/tmp/iter_accuracy_tmp_${UUID}.txt
ACCURACYFILE=/tmp/iter_accuracy_${UUID}.txt
ACCURACYTOP5FILE=/tmp/iter_accuracy_top5_${UUID}.txt

if [ -z "${LOGFILE}" ];
then
  echo "[Error] Can not find a log file."
  echo "[Error] A valid filename should match *_{log|train}.txt or *.log."
  echo "[Error] Alternatively, specify log file(s) as input arg (wildcard permitted). Exitting."
  exit -1
fi

echo --------------------------------
echo "[Info] log file(s)=${LOGFILE}"

# try to guess num of sample sizes (1 epoch)
SOLVERFILE=$(ls -t1 ${DIR}/*solver*.*txt | egrep -v -i temp | head -1)
echo "[Info] solver file=${SOLVERFILE}"
TRAINFILE=$(egrep -i "net.*:" "${SOLVERFILE}" | egrep -v "#" | sed -e 's/^[^"]*"//' -e 's/"$//')
# train file path could be relative to Caffe root
if [ ! -f ${TRAINFILE} ]; then
  # try a parent
  TRAINFILE=../${TRAINFILE}
  if [ ! -f ${TRAINFILE} ]; then
    # try a parent
    TRAINFILE=../${TRAINFILE}
    if [ ! -f ${TRAINFILE} ]; then
      TRAINFILE_FOUND=false
    else
      TRAINFILE_FOUND=true
    fi
  else
    TRAINFILE_FOUND=true
  fi
else
  TRAINFILE_FOUND=true
fi

if ${TRAINFILE_FOUND}; then
  # assuming the first batch_size encountered is batch_size for "training"
  BATCHSIZE=$(egrep "batch_size" ${TRAINFILE} | egrep -v "#" | head -1 | awk '{print $2}')
  SOURCE=$(egrep "source.*:" ${TRAINFILE} | egrep -v "#" | head -1 | awk '{print $2}')
  SOURCE=$(echo ${SOURCE} | sed -e 's/^[^"]*"//' -e 's/"$//')
  if [ ! -f ${SOURCE} ]; then
    # try a parent
    SOURCE=../${SOURCE}
    if [ ! -f ${SOURCE} ]; then
      # try a parent
      SOURCE=../${SOURCE}
      if [ ! -f ${SOURCE} ]; then
        SOURCE_FOUND=false
      else
        SOURCE_FOUND=true
      fi
    else
      SOURCE_FOUND=true
    fi
  else
    SOURCE_FOUND=true
  fi
  if ${SOURCE_FOUND}; then
    NUMTRAINSAMPLES=$(cat "${SOURCE}" | wc -l)

    # find number of gpu's
    TRAINSCRIPT=$(ls -t1 ${DIR}/*train*.sh ${DIR}/*finetune*.sh 2>/dev/null | head -1)
    # train file path could be relative to Caffe root
    if [ ! -f ${TRAINSCRIPT} ]; then
      # try a parent
      TRAINSCRIPT=../${TRAINSCRIPT}
      if [ ! -f ${TRAINSCRIPT} ]; then
        # try a parent
        TRAINSCRIPT=../${TRAINSCRIPT}
        if [ ! -f ${TRAINSCRIPT} ]; then
          TRAINSCRIPT_FOUND=false
        else
          TRAINSCRIPT_FOUND=true
        fi
      else
        TRAINSCRIPT_FOUND=true
      fi
    else
      TRAINSCRIPT_FOUND=true
    fi

    NUMGPU=$(cat "${TRAINSCRIPT}" | egrep "\-gpu" | egrep -v "#" | grep -o "," | wc -l)
    NUMGPU=$((NUMGPU + 1))

    if [ ${NUMTRAINSAMPLES} -gt 0 ]; then
      echo "[Info] NUMTRAINSAMPLES=${NUMTRAINSAMPLES}, BATCHSIZE=${BATCHSIZE}, NUMGPU=${NUMGPU}"
      export EPOCH=$(( NUMTRAINSAMPLES / BATCHSIZE / NUMGPU))
    fi
  fi
fi

if [ -n "${EPOCH}" ]; then
  echo "[Info] 1 epoch=${EPOCH} iters"
fi
echo --------------------------------

### Set initial time of file
LASTLOGFILE="${LOGFILE##* }"
LASTTIME=""

echo -n "Last iter:"
while true
do
  NEWTIME=`stat -c %Z "${LASTLOGFILE}"`
  if [[ "${LASTTIME}" != "${NEWTIME}" ]]
  then
    (killall eog 2> /dev/null)
    cat ${LOGFILE} | egrep "Iteration.* loss = " | sed -e 's/.*Iteration //' -e 's/, loss = / /' > "${LOSSFILE}"
    cat ${LOGFILE} | egrep -B 2 --no-group-separator ": accuracy\/top\-1" | egrep " solver\.cpp" | sed -e 's/.*Iteration //' -e 's/, Testing.*//' -e 's/.*= //' > "${ACCURACYTMPFILE}"
    paste - - < "${ACCURACYTMPFILE}" | awk '!seen[$1]++' > "${ACCURACYFILE}"
    cat ${LOGFILE} | egrep -B 3 --no-group-separator ": accuracy\/top\-5" | egrep -v "top\-1" | egrep " solver\.cpp" | sed -e 's/.*Iteration //' -e 's/, Testing.*//' -e 's/.*= //' > "${ACCURACYTMPFILE}"
    paste - - < "${ACCURACYTMPFILE}" |awk '!seen[$1]++' > "${ACCURACYTOP5FILE}"
    py_plot_training_loss.py "${LOSSFILE}" "${PLOTFILE}" "${ACCURACYFILE}" "${ACCURACYTOP5FILE}" > /dev/null 2>&1
    (eog "${PLOTFILE}" 2> /dev/null) &
    LASTITER=$(tail -n 1 "${LOSSFILE}" | awk '{print $1}')
    echo -n " [${LASTITER}]"
    LASTTIME="${NEWTIME}"
  fi
  sleep ${REFRESHSECOND}
done 2> /dev/null
echo " "
echo --------------------------------
