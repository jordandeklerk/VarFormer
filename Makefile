.PHONY: all clean m3 m4

DATASETS_DIR := datasets

all: clean m3 clean m4

$(DATASETS_DIR):
	mkdir -p $(DATASETS_DIR)

m3: $(DATASETS_DIR)
	python -m datasets.main build --dataset=m3

m4: $(DATASETS_DIR)
	python -m datasets.main build --dataset=m4

clean:
	rm -rf $(DATASETS_DIR)/M3C.xls
	rm -rf $(DATASETS_DIR)/Monthly-train.csv
	rm -rf $(DATASETS_DIR)/Monthly-test.csv
	python3 -c "import gc; gc.collect()"
