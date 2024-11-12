

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
# Models (weights) are retrieved from Google drive---a provisory location---until we come up with a better solution.
#

ROOT=$(HOME)/graz/htr/vre
DATA=$(HOME)/tmp/data/1000CV

HTR_APP_DIR=$(ROOT)/ddpa_htr
LINESEG_APP_DIR=$(ROOT)/ddpa_lines
REGSEG_APP_DIR=$(ROOT)/yolov5 # 'seals' app

LINESEG_OUT_FORMAT="json"
HTR_OUT_FORMAT="json"

seals_model=seals.20241024.pt
# NPR Google drive
seals_model_url="https://drive.google.com/uc?id=17_bhsxtQL28mVFpSYkQ9vMYErbcgKsIk"

line_segmentation_model=blla.mlmodel
# NPR Google drive
line_segmentation_model_url="https://drive.google.com/uc?id=160PxpPdnkBAzrpSpzW-dJn8Vd5CHBAom"

htr_model=model_monasterium-2024-10-28.mlmodel
htr_model_url="https://drive.google.com/uc?id=1GOKgGWvhO7ugWw0tevzXhQa2cVx09iLu"




main:
	echo $(ROOT)


clean_all: clean_htr clean_lineseg clean_regseg

clean_htr:
	find ${DATA} -name "*.htr.pred.*" | xargs rm -f

clean_lineseg:
	find ${DATA} -name "*.line.pred.*" | xargs rm -f

clean_regseg:
	find ${DATA} -name "*.seals.pred.*" | xargs rm -f

.SECONDARY: $(%.lines.pred.json) $(%.seals.pred.json)

%.htr.pred.json: %.img.jpg %.lines.pred.json
	cd $(HTR_APP_DIR);\
		test -f "$(htr_model)" || gdown -O $(htr_model) $(htr_model_url);\
		PYTHONPATH=$(HTR_APP_DIR) bin/ddp_htr_inference -output_format $(HTR_OUT_FORMAT) -model_path $(htr_model) -img_paths $<


%.lines.pred.json: %.img.jpg %.seals.pred.json
	cd $(LINESEG_APP_DIR);\
	       test -f "$(line_segmentation_model)" || gdown -O $(line_segmentation_model) $(line_segmentation_model_url);\
       	       PYTHONPATH=$(LINESEG_APP_DIR) bin/ddp_line_detect -output_format $(LINESEG_OUT_FORMAT) -model_path $(line_segmentation_model) -region_segmentation_suffix '.seals.pred.json' -img_paths $<


%.seals.pred.json: %.img.jpg
	cd $(REGSEG_APP_DIR);\
		test -f "$(seals_model)" || gdown -O $(seals_model) $(seals_model_url);\
		PYTHONPATH=$(REGSEG_APP_DIR) bin/ddp_seals_detect -weights $(seals_model) -img_paths $<




%.OldText.jpg: %.img.jpg
	echo ""

