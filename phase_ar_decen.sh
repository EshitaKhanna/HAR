#!/bin/bash

#Table 3
nohup python -u phase_ar_decen.py unihar merge 20_120 -g 0 -sd 1 -e unihar_bert_fed_d1 -s unihar_decen_d1 > log/phase_ar_decen_unihar_d1.log &
nohup python -u phase_ar_decen.py unihar merge 20_120 -g 1 -sd 2 -e unihar_bert_fed_d2 -s unihar_decen_d2 > log/phase_ar_decen_unihar_d2.log &
nohup python -u phase_ar_decen.py unihar merge 20_120 -g 0 -sd 3 -e unihar_bert_fed_d3 -s unihar_decen_d3 > log/phase_ar_decen_unihar_d3.log &
nohup python -u phase_ar_decen.py unihar merge 20_120 -g 1 -sd 4 -e unihar_bert_fed_d4 -s unihar_decen_d4 > log/phase_ar_decen_unihar_d4.log &

#Figure 9
#nohup python -u phase_ar_decen.py unihar merge 20_120 -g 0 -sd 1 -n 50 -e unihar_bert_fed_d1 -s unihar_decen_d1 > log/phase_ar_decen_unihar_d1.log &
#nohup python -u phase_ar_decen.py unihar merge 20_120 -g 1 -sd 2 -n 50 -e unihar_bert_fed_d2 -s unihar_decen_d2 > log/phase_ar_decen_unihar_d2.log &
#nohup python -u phase_ar_decen.py unihar merge 20_120 -g 0 -sd 3 -n 50 -e unihar_bert_fed_d3 -s unihar_decen_d3 > log/phase_ar_decen_unihar_d3.log &
#nohup python -u phase_ar_decen.py unihar merge 20_120 -g 1 -sd 4 -n 50 -e unihar_bert_fed_d4 -s unihar_decen_d4 > log/phase_ar_decen_unihar_d4.log &

#Figure 10
#nohup python -u phase_ar_decen.py unihar merge 20_120 -g 0 -sd 1 -rda 1 -e unihar_bert_fed_d1 -s unihar_decen_d1 > log/phase_ar_decen_unihar_d1.log &
#nohup python -u phase_ar_decen.py unihar merge 20_120 -g 1 -sd 2 -rda 1 -e unihar_bert_fed_d2 -s unihar_decen_d2 > log/phase_ar_decen_unihar_d2.log &
#nohup python -u phase_ar_decen.py unihar merge 20_120 -g 0 -sd 3 -rda 1 -e unihar_bert_fed_d3 -s unihar_decen_d3 > log/phase_ar_decen_unihar_d3.log &
#nohup python -u phase_ar_decen.py unihar merge 20_120 -g 1 -sd 4 -rda 1 -e unihar_bert_fed_d4 -s unihar_decen_d4 > log/phase_ar_decen_unihar_d4.log &