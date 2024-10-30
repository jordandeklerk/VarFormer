.PHONY: all clean m3 m4

DATASETS_DIR := datasets

all: m3 m4

$(DATASETS_DIR):
	mkdir -p $(DATASETS_DIR)

m3: $(DATASETS_DIR)
	@echo "Downloading M3 dataset..."
	wget -P $(DATASETS_DIR) https://forecasters.org/data/m3comp/M3C.xls
	@echo "M3 dataset downloaded successfully"

m4: $(DATASETS_DIR)
	@echo "Downloading M4 dataset..."
	wget -P $(DATASETS_DIR) https://github.com/Mcompetitions/M4-methods/raw/master/Dataset/Train/Monthly-train.csv
	wget -P $(DATASETS_DIR) https://github.com/Mcompetitions/M4-methods/raw/master/Dataset/Test/Monthly-test.csv
	@echo "M4 dataset downloaded successfully"

clean:
	rm -f $(DATASETS_DIR)/M3C.xls
	rm -f $(DATASETS_DIR)/Monthly-train.csv
	rm -f $(DATASETS_DIR)/Monthly-test.csv
