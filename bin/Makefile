

# A Makefile to generate transcription files out of charter images. The target patterns match the
# output file pattern of the involved Python apps. Therefore, image filenames that should be the parameters
# of a Python app should be rewritten first to match the pattern of their respective output files. 
#
# Example on the CLI:
#   
#   ~$ echo ~/tmp/data/1000CV/AT-StiAM/*/*/*.jpg | sed 's/img.jpg/htr.pred.json/g' | xargs make 
# 
# Within a wrapper shell script that takes a sequence of filenames as parameters:
#
#   make $(echo "${@/.img.jpg/.htr.pred.json}")
#
#

ROOT=$(HOME)/graz/htr/vre
DATA=$(HOME)/tmp/data/1000CV

HTR_APP_DIR=$(ROOT)/ddpa_htr
LINESEG_APP_DIR=$(ROOT)/ddpa_lines
REGSEG_APP_DIR=$(ROOT)/yolov5 # 'seals' app

SEG_OUT_FORMAT="xml"
HTR_OUT_FORMAT="stdout"


main:
	echo $(ROOT)

clean_all: clean_htr clean_seg

clean_htr:
	find ${DATA} -name "*.htr.pred.*" | xargs rm -f

clean_seg:
	find ${DATA} -name "*.line.pred.*" | xargs rm -f

%.htr.pred.json: %.img.jpg %.lines.pred.xml
	cd $(HTR_APP_DIR); PYTHONPATH=$(HTR_APP_DIR) bin/ddp_htr_inference -output_format $(HTR_OUT_FORMAT) -img_paths $<


%.lines.pred.xml: %.seals.pred.json
	cd $(LINESEG_APP_DIR); PYTHONPATH=$(LINESEG_APP_DIR) bin/ddp_line_detect -output_format $(LINESEG_OUT_FORMAT) -img_paths $<

%.seals.pred.json: %.img.jpg
	cd $(REGSEG_APP_DIR); PYTHONPATH=$(REGSEG_APP_DIR) bin/ddp_seals_detect -img_paths $<





%.OldText.jpg: %.img.jpg
	echo ""

