### please change the corresponding path prefix ${PATH}


import sys, os, errno
import numpy as np
import csv
import json
import copy

assert len(sys.argv) >= 2, "Usage: python log_analysis.py <test_log>"
num_logs = len(sys.argv) - 1
lines = []
for i in xrange(num_logs):
  log = sys.argv[i+1]
  with open(log, 'r') as f:
    lines += f.read().splitlines()

split='test'
with open('${PATH}/Charades/Charades_v1_%s.csv'%split, 'r') as csvfile:
  reader = csv.reader(csvfile, delimiter=',')
  data = [row for row in reader][1:]

vid_length={}
for i, row in enumerate(data):
  vid = row[0]
  length= float(row[10])
  vid_length[vid]=length


def nms(dets, thresh=0.4):
    """Pure Python NMS baseline."""
    if len(dets) == 0: return []
    x1 = dets[:, 0]
    x2 = dets[:, 1]
    scores = dets[:, 2]
    lengths = x2 - x1
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1)
        ovr = inter / (lengths[i] + lengths[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep

def get_segments(data):
    segments = []
    vid = 'Background'
    find_next = False
    tmp = {'label' : 'c0', 'segment': [0, 0, 0]}
    for l in data:
      # video name and sliding window length
      if "fg_name :" in l:
         vid = l.split('/')[-1]
         continue    

      # frame index, time, confident score
      elif "frames :" in l:
         start_frame=int(l.split()[4])
         stride = int(l.split()[6].split(']')[0])

      elif "activity:" in l:
         label = int(l.split()[1])
         tmp['label'] ='c%03d' % (label-1)  
         find_next = True

      elif "im_detect" in l:
         return vid, segments

      elif find_next:
         left  = ( float(l.split()[1])*stride + start_frame) / 25.0
         right = ( float(l.split()[2])*stride + start_frame) / 25.0
         score = float(l.split()[3].split(']')[0])
         tmp1 = copy.deepcopy(tmp)
         tmp1['segment'] = [left, right, score]
         segments.append(tmp1)
  

segmentations = {}
predict_data = []
for l in lines:
  if "gt_classes :" in l:
    predict_data = []
  predict_data.append(l)
  if "im_detect:" in l:
    vid, segments = get_segments(predict_data)
    if vid not in segmentations:
      segmentations[vid] = []
    segmentations[vid] += segments

res = {}
for vid, vinfo in segmentations.iteritems():
  labels = list(set([d['label'] for d in vinfo]))
  res[vid] = []
  for lab in labels:
    nms_in = [d['segment'] for d in vinfo if d['label'] == lab]
    keep = nms(np.array(nms_in), thresh=0.4)
    for i in keep:
      tmp = {'label':lab, 'segment': nms_in[i]}
      res[vid].append(tmp)

#SAMPLE = 25
SAMPLE = 150
SELECT_SAMPLE = 25

text_file = open("results.txt", "w")
for vid, vinfo in res.iteritems():
  length = len(os.listdir('../../../preprocess/charades/frames/'+vid))
  score_75=[]
  for i in xrange(SAMPLE):
    t = i *vid_length[vid] * 1.0 / SAMPLE
    select = [d for d in vinfo if d['segment'][0] <= t and d['segment'][1] >= t]
    scores = {}
    for d in select:
      if d['label'] not in scores:
         scores[d['label']] = d['segment'][2]
      else:
         if d['segment'][2] > scores[d['label']]:
            scores[d['label']] = d['segment'][2]
    score_75_tmp=[]
    for j in xrange(157):
      lab = 'c%03d'%j
      score_75_tmp.append((scores[lab] if lab in scores else 0))
    score_75.append(score_75_tmp)
  score_75=np.array(score_75)
  for m in xrange(0, SAMPLE-1, SAMPLE/SELECT_SAMPLE):
    matrix_tmp=score_75[max(0,m-15):min(m+15,SAMPLE),:]
    smooth_tmp=np.mean(matrix_tmp,axis=0)
    label_tmp = '%s %d' % (vid, m/(SAMPLE/SELECT_SAMPLE))
    for n in xrange(smooth_tmp.shape[0]):
      label_tmp += ' '+str(smooth_tmp[n])
    text_file.write(label_tmp + '\n')
  
text_file.close()

